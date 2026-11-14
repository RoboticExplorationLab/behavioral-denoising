"""Reacher benchmark: five references, ten fixed starts, and saved MPC demonstrations."""
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

from src.tasks.reacher import ReacherTask




def episode(act, reset, seed, targets, steps, augment, noise=0.0, rng=None,
            record=False, trace=False, settings=None, recorder=None):
    """Run one episode and record costs, reference completion, and violations."""
    settings = settings or load_experiment_config("reacher")
    state_cost = np.diag(settings["state_weights"])
    action_cost = np.diag(settings["action_weights"])
    action_limit = settings["action_limit"]
    height_limit = settings["height_limit"]
    stop_after_targets = settings.get("evaluation", {}).get("stop_after_targets")
    if (stop_after_targets is not None
            and (isinstance(stop_after_targets, bool)
                 or not isinstance(stop_after_targets, (int, np.integer))
                 or stop_after_targets < 1)):
        raise ValueError("evaluation.stop_after_targets must be a positive integer or None")
    task_options = {**settings.get("task", {}), "max_episode_steps": steps, "targets": targets}
    task = ReacherTask(**task_options)
    observations, actions, times = [], [], []
    states_trace, actions_trace = [], []
    cost, reached, violations, strict_violations = 0.0, 0, 0, 0
    previous_target = 0
    benchmark = benchmark_outcomes(settings)
    audit = EpisodeAudit()
    terminal_reason, controller_error = "horizon", None
    completed_steps = 0
    try:
        raw_state = task.reset(seed)
        if recorder is not None:
            recorder.start(task, raw_state)
        reset(seed)
        if trace:
            states_trace.append(task.observation(raw_state).copy())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for step in range(steps):
                # When stopping at a reference count, reuse the reference
                # updated immediately after the preceding action.
                if stop_after_targets is None or step == 0:
                    reference = task.reference(raw_state, step)
                    if recorder is not None:
                        recorder.reference(reference, step)
                if task.target_index != previous_target:
                    reached += 1
                    previous_target = task.target_index
                if stop_after_targets is not None and reached >= stop_after_targets:
                    terminal_reason = "targets_completed"
                    break
                observation = task.observation(raw_state)
                if record and observations and augment:
                    observations[-1] = np.r_[observations[-1][:8], reference[4:6]]
                policy_observation = np.r_[observation, reference[4:6]] if augment else observation
                started = perf_counter()
                try:
                    proposed = np.asarray(act(policy_observation, reference), dtype=float).reshape(2)
                except HistoryConstraintError as error:
                    if not benchmark:
                        raise
                    times.append(perf_counter() - started)
                    if recorder is not None:
                        recorder.decision(getattr(act, "__self__", None), policy_observation, reference, None, times[-1], error)
                    terminal_reason, controller_error = "history_infeasible", str(error)
                    cost = np.inf
                    break
                times.append(perf_counter() - started)
                if recorder is not None:
                    recorder.decision(getattr(act, "__self__", None), policy_observation, reference, proposed, times[-1])
                if benchmark:
                    audit.decision(getattr(act, "__self__", None))
                    if not np.isfinite(proposed).all():
                        terminal_reason, cost = "nonfinite_action", np.inf
                        break
                perturbed = proposed + noise * action_limit * rng.standard_normal(2) if noise else proposed
                action = np.clip(perturbed, -action_limit, action_limit)
                if benchmark:
                    audit.applied(proposed, action, clipped=not np.array_equal(perturbed, action), noisy=bool(noise))
                record_applied_action(getattr(act, "__self__", None), action)
                error = observation - reference
                cost += float(error @ state_cost @ error + action @ action_cost @ action)
                transition = task.dynamics.step(action)
                if recorder is not None:
                    recorder.transition(action, action, reference, transition)
                raw_state = transition.observation
                completed_steps += 1
                following = task.observation(raw_state)
                violations += int(following[5] > height_limit + 1e-3)
                strict_violations += int(following[5] > height_limit)
                if record:
                    actions.append(action)
                    observations.append(np.r_[following, reference[4:6]] if augment else following)
                if trace:
                    actions_trace.append(action.copy())
                    states_trace.append(following.copy())
                if not np.isfinite(following).all():
                    cost = np.inf
                    terminal_reason = "nonfinite_state"
                    break
                if stop_after_targets is not None:
                    # Check the state just reached, including the last allowed
                    # transition. Reuse this reference if another action follows.
                    reference = task.reference(raw_state, completed_steps)
                    if recorder is not None:
                        recorder.reference(reference, completed_steps)
                    completed_targets = reached + int(task.target_index != previous_target)
                    if completed_targets >= stop_after_targets:
                        terminal_reason = "targets_completed"
                        break
                if transition.terminated or transition.truncated:
                    terminal_reason = "terminated" if transition.terminated else "truncated"
                    break
        if benchmark and terminal_reason in {"history_infeasible", "nonfinite_action", "nonfinite_state"}:
            final_reference = reference
            reached_final = reached
        else:
            # Stopping mode has already inspected the final state. A second
            # reference call could advance again when two targets overlap.
            final_reference = (task.reference(raw_state, steps)
                               if stop_after_targets is None else reference)
            reached_final = reached + int(task.target_index != previous_target)
        if record and observations and augment:
            observations[-1] = np.r_[observations[-1][:8], final_reference[4:6]]
    finally:
        task.close()
    result = dict(cost=cost, reached=reached, violations=violations,
                  reached_final=reached_final, violations_strict=strict_violations,
                  steps=completed_steps if benchmark else len(times), ms=1e3 * float(np.mean(times)) if times else 0.0)
    if benchmark:
        result.update(audit.fields(terminal_reason, controller_error))
        result.update(targets_completed=reached_final, tracking_cost=cost,
                      success=bool(reached_final >= 5 and not result["controller_failure"]),
                      all_targets_completed=bool(reached_final >= 5 and not result["controller_failure"]))
    if record:
        result["trajectory"] = TrajectoryData(np.asarray(observations), np.asarray(actions))
    if trace:
        result["states"] = np.asarray(states_trace).tolist()
        result["actions"] = np.asarray(actions_trace).tolist()
    return result





def strip_targets(data):
    trajectories = [TrajectoryData(item.state_trajectory[:, :8], item.input_trajectory) for item in data.dataset]
    return TrajectoryDataSet(trajectories, data.dataset_name, data.timing)


def build_policy(method, parameters, data, settings, task=None, cache=None):
    """Construct a policy with the selected method parameters."""
    family = method["family"]
    policy_class = POLICIES[family]
    if family in DEEPC_FAMILIES:
        if task is None:
            task = ReacherTask(**settings.get("task", {}))
            task.close()
        return build_deepc(policy_class, {**parameters, "family": family}, data, task, cache)
    return build_imitation(policy_class, method, parameters, data, 10, 2, cache)


def evaluation_starts(settings):
    """Load the evaluation starts and reference sequences."""
    import json
    saved = json.loads((ROOT / settings.get("test_starts", settings.get("test_starts_file", "data/reacher/test_starts.json"))).read_text())
    steps = settings["evaluation"].get("steps", saved["max_steps"])
    starts = [(record["seed"], np.asarray(record["targets"]), steps) for record in saved["starts"]]
    labels = [record["test_set"] for record in saved["starts"]]
    indices = [record["test_index"] for record in saved["starts"]]
    return [], starts, labels, indices


def run_experiment(config_path=None, methods=None, overrides=None, output=None, *, seeds=None, steps=None):
    """Run fixed configurations and write resumable, per-start paper records."""
    from src.benchmarks.runner import run_task
    return run_task("reacher", config_path=config_path, methods=methods, seeds=seeds,
                    steps=steps, output=output, overrides=overrides)


if __name__ == "__main__":
    from src.benchmarks.runner import main
    main("reacher")
