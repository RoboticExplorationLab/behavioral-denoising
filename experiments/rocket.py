"""Rocket Landing benchmark with the paper's full saved MPC bank."""
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np

from src.benchmarks.outcomes import EpisodeAudit, benchmark_outcomes
from src.benchmarks.policies import build_deepc, build_imitation
from src.evaluation import record_applied_action, load_experiment_config
from src.data import TrajectoryData, TrajectoryDataSet, prepare_dataset
from src.policies import DeePC, SelectDPC, WindowedDeePC
from src.policies.bd_mean_chi2 import BDMeanChi2Policy
from src.policies.behavioral_denoising import BehavioralDenoisingPolicy
from src.policies.bd_kl import BDKLPolicy, HistoryConstraintError
from src.policies.bd_chi2 import BDChi2Policy

ROOT = Path(__file__).resolve().parents[1]
POLICIES = {
    "bd": BehavioralDenoisingPolicy, "bd_kl": BDKLPolicy,
    "bd_mean_chi2": BDMeanChi2Policy, "bd_chi2": BDChi2Policy,
    "deepc": DeePC, "select_dpc": SelectDPC, "windowed_deepc": WindowedDeePC,
}
DEEPC_FAMILIES = {"deepc", "windowed_deepc", "select_dpc"}

from src.tasks.rocket import RocketTask



def make_task(start, steps, settings=None):
    task_options = dict((settings or {}).get("task", {"mode": "landing"}))
    task_options["max_episode_steps"] = steps
    if start is None:
        task_options["random_initial_position"] = True
    else:
        task_options["initial_condition"] = start
    return RocketTask(**task_options)


def bounds(task):
    environment = task.dynamics.env.unwrapped
    limits = [environment.cfg.main_engine_thrust_limits, environment.cfg.side_engine_thrust_limits,
              environment.cfg.nozzle_angle_limits]
    return np.array([limit[0] for limit in limits], float), np.array([limit[1] for limit in limits], float)


def hover_input(task):
    environment = task.dynamics.env.unwrapped
    mass, _ = environment.get_mass_properties()
    return np.array([-environment.cfg.gravity * mass / environment.cfg.main_engine_thrust, 0.0, 0.0])



def episode(make, seed, start, steps, uses_raw=False, noise=0.0, rng=None,
            record=False, trace=False, settings=None, recorder=None):
    """Run one episode with shared touchdown control and landing metrics."""
    settings = settings or load_experiment_config("rocket")
    state_cost = np.diag(settings["state_weights"])
    action_cost = np.diag(settings["action_weights"])
    task = make_task(start, steps, settings)
    times, observations, actions = [], [], []
    states_trace, actions_trace = [], []
    cost, cost_hover = 0.0, 0.0
    crashed, touched, settled, ended = False, False, False, False
    benchmark = benchmark_outcomes(settings)
    audit = EpisodeAudit()
    terminal_reason, controller_error = "horizon", None
    completed_steps = 0
    try:
        np.random.seed(seed)  # The simulator samples starts from NumPy's global stream.
        raw_state = task.reset(seed)
        if recorder is not None:
            recorder.start(task, raw_state)
        policy = make(task)
        policy.reset(seed)
        lower, upper = bounds(task)
        hover = hover_input(task)
        if trace:
            states_trace.append(task.observation(raw_state).copy())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for step in range(steps):
                reference = task.reference(raw_state, step)
                if recorder is not None:
                    recorder.reference(reference, step)
                observation = task.observation(raw_state)
                override = task.override_action(raw_state)
                if override is None:
                    started = perf_counter()
                    try:
                        proposed = np.asarray(policy.act(raw_state if uses_raw else observation, reference), float).reshape(3)
                    except HistoryConstraintError as error:
                        if not benchmark:
                            raise
                        times.append(perf_counter() - started)
                        if recorder is not None:
                            recorder.decision(policy, observation, reference, None, times[-1], error)
                        terminal_reason, controller_error = "history_infeasible", str(error)
                        cost = cost_hover = np.inf
                        break
                    times.append(perf_counter() - started)
                    if recorder is not None:
                        recorder.decision(policy, observation, reference, proposed, times[-1])
                    if benchmark:
                        audit.decision(policy)
                        if not np.isfinite(proposed).all():
                            terminal_reason = "nonfinite_action"
                            cost = cost_hover = np.inf
                            break
                    perturbed = proposed + noise * (upper - lower) / 2 * rng.standard_normal(3) if noise else proposed
                    action = np.clip(perturbed, lower, upper)
                else:
                    proposed = perturbed = action = np.asarray(override, float)
                applied = np.clip(action, lower, upper)
                if benchmark:
                    if not np.isfinite(applied).all():
                        terminal_reason = "nonfinite_action"
                        cost = cost_hover = np.inf
                        break
                    audit.applied(proposed, applied, clipped=not np.array_equal(perturbed, applied),
                                  noisy=bool(noise and override is None), override=override is not None)
                record_applied_action(policy, applied)
                error = observation - reference
                cost += float(error @ state_cost @ error + action @ action_cost @ action)
                applied_relative_to_hover = np.clip(action, lower, upper) - hover
                cost_hover += float(error @ state_cost @ error
                                    + applied_relative_to_hover @ action_cost @ applied_relative_to_hover)
                keep_sample = task.include_demonstration(raw_state)
                transition = task.dynamics.step(action)
                if recorder is not None:
                    recorder.transition(action, applied, reference, transition, override=override is not None)
                raw_state = transition.observation
                completed_steps += 1
                touched |= bool(np.any(raw_state[6:]))
                if record and keep_sample and override is None:
                    actions.append(action)
                    observations.append(task.observation(raw_state))
                if trace:
                    actions_trace.append(action.copy())
                    states_trace.append(task.observation(raw_state).copy())
                if benchmark and not np.isfinite(raw_state).all():
                    terminal_reason = "nonfinite_state"
                    cost = cost_hover = np.inf
                    break
                if transition.terminated:
                    crashed = transition.reward < 0
                    settled = transition.reward > 0
                    ended = True
                    terminal_reason = "crashed" if crashed else "settled" if settled else "terminated"
                    break
                if transition.truncated:
                    terminal_reason = "truncated"
                    break
    finally:
        task.close()
    result = dict(success=float(touched and not crashed), crashed=float(crashed), steps=completed_steps,
                  cost=cost, settled=float(touched and settled), timeout=float(not ended), cost_hover=cost_hover,
                  ms=1e3 * float(np.mean(times)) if times else 0.0)
    if benchmark:
        result.update(audit.fields(terminal_reason, controller_error))
        result.update(contact_success=result["success"], success=result["settled"],
                      timeout=float(terminal_reason in {"horizon", "truncated"}))
    if record:
        result["trajectory"] = TrajectoryData(np.asarray(observations), np.asarray(actions)) if actions else None
    if trace:
        result["states"] = np.asarray(states_trace).tolist()
        result["actions"] = np.asarray(actions_trace).tolist()
    return result


def build_policy(method, parameters, data, task, cache=None):
    policy_class = POLICIES[method["family"]]
    if method["family"] in DEEPC_FAMILIES:
        return build_deepc(policy_class, {**parameters, "family": method["family"]}, data, task, cache)
    return build_imitation(policy_class, method, parameters, data, 6, 3, cache)


def evaluation_starts(settings, select_subset=True):
    """Load the simulator reset arguments for each evaluation start."""
    import json
    saved = json.loads((ROOT / settings.get("test_starts", settings.get("test_starts_file", "data/rocket/test_starts.json"))).read_text())
    starts = [(record["seed"], record["initial_condition"]) for record in saved["starts"]]
    return [], starts, [record["test_index"] for record in saved["starts"]]


def run_experiment(config_path=None, methods=None, overrides=None, output=None, *, seeds=None, steps=None):
    """Run fixed configurations and write resumable, per-start paper records."""
    from src.benchmarks.runner import run_task
    return run_task("rocket", config_path=config_path, methods=methods, seeds=seeds,
                    steps=steps, output=output, overrides=overrides)


if __name__ == "__main__":
    from src.benchmarks.runner import main
    main("rocket")
