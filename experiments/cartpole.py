"""Cartpole benchmark with the paper's saved MPC bank and fixed configurations."""
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np

from src.benchmarks.outcomes import EpisodeAudit
from src.evaluation import record_applied_action
from src.data import DeePCConstraints, DeePCControllerArgs, DeePCCost, DeePCDims, prepare_dataset
from src.optimization.hankel import HankelMatrixGenerator
from src.policies.bd_mean_chi2 import BDMeanChi2Policy
from src.policies.behavioral_denoising import BehavioralDenoisingPolicy
from src.policies.bd_kl import BDKLPolicy, HistoryConstraintError
from src.policies.deepc import DeePC
from src.policies.bd_chi2 import BDChi2Policy
from src.policies.select_dpc import SelectDPC
from src.policies.windowed_deepc import WindowedDeePC
from src.selectors import LkSelector
from src.tasks.cartpole import CartpoleTask

ROOT = Path(__file__).resolve().parents[1]
POLICIES = {"bd": BehavioralDenoisingPolicy, "bd_kl": BDKLPolicy,
            "bd_mean_chi2": BDMeanChi2Policy, "bd_chi2": BDChi2Policy}



def prepare_context(data, past=1, future=1, neighbors=200, past_inputs=False):
    return {"data": prepare_dataset(data), "past": past, "future": future,
            "neighbors": neighbors, "past_inputs": past_inputs, "matrices": {}}


def dictionary(context, past, future, hankel_length=None):
    key = (past, future, hankel_length)
    if key not in context["matrices"]:
        context["matrices"][key] = HankelMatrixGenerator(past, future, hankel_length).generate_hankel_matrices(context["data"])
    return context["matrices"][key]


def build_policy(context, method, parameters):
    options = deepcopy(parameters)
    family = method["family"]
    past, future = context["past"], context["future"]
    if family == "deepc":
        past, future = options["past"], options["future"]
    dims = DeePCDims(past, future, 5, 1)
    truncate_episodes = family == "deepc" and method["policy"] == "deepc" and options.get("truncate_episodes", False)
    hankel_length = options["hankel_length"] if truncate_episodes else None
    inputs, outputs = dictionary(context, past, future, hankel_length)
    if family == "deepc":
        costs = DeePCCost(np.diag(options["Q"]), np.diag(options["R"]), options["slack"],
                          options["g_l1"], options["g_projection"])
        bounds = DeePCConstraints(*(np.asarray(options[key]) for key in ("A_u", "b_u", "A_y", "b_y")))
        reference = np.asarray(options["input_reference"])
        arguments = DeePCControllerArgs(context["data"], dims, options["hankel_length"], costs, bounds, reference, False)
        solver_options = options.get("solver")
        if method["policy"] == "select_dpc":
            if options["selector"] != "norm":
                raise ValueError("The cartpole protocol requires the configured norm selector")
            return SelectDPC(arguments, LkSelector(deepc_dims=dims, **options["selector_options"]),
                             options["columns"], options["iterations"], solver_options=solver_options,
                             hankel_matrices=(inputs, outputs))
        if method["policy"] == "deepc":
            maximum_columns = options.get("max_columns", 800)
            count = inputs.shape[1] if maximum_columns is None else min(maximum_columns, inputs.shape[1])
            subset = np.linspace(0, inputs.shape[1] - 1, count).astype(int)
            return DeePC(inputs[:, subset], outputs[:, subset], dims, costs, bounds, reference,
                         solver_options=solver_options)
        if method["policy"] == "windowed_deepc":
            count = min(options.get("window_columns", options["columns"] * 5), inputs.shape[1])
            return WindowedDeePC(inputs[:, -count:], outputs[:, -count:], arguments,
                                 solver_options=solver_options)
        raise ValueError(f"Unknown DeePC policy: {method['policy']}")
    options.setdefault("neighbors", context["neighbors"])
    options.setdefault("include_past_inputs", context["past_inputs"])
    return POLICIES[method["policy"]](inputs, outputs, dims, **options)


def balanced(states):
    angles = np.arctan2(np.sin(states[:, 1]), np.cos(states[:, 1]))
    return (np.abs(angles) <= np.deg2rad(15)) & (np.abs(states[:, 3]) <= 1) & (np.abs(states[:, 0]) <= 1)


def run_episode(policy, seed, task_options=None, steps=300, reference=None, record_trace=False,
                outcome_protocol=None, recorder=None):
    task = CartpoleTask(**(task_options or {}))
    reference = np.asarray(reference if reference is not None else [0, 0, 1, 0, 0], dtype=float)
    states, actions, durations = [], [], []
    benchmark = outcome_protocol == "benchmark_v1"
    audit = EpisodeAudit()
    terminal_reason, controller_error = "horizon", None
    try:
        state = task.reset(seed)
        initial = state.copy()
        if recorder is not None:
            recorder.start(task, state)
        policy.reset(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(steps):
                started = perf_counter()
                try:
                    proposed = np.asarray(policy.act(task.observation(state), reference), dtype=float)
                except HistoryConstraintError as error:
                    if not benchmark:
                        raise
                    durations.append(perf_counter() - started)
                    if recorder is not None:
                        recorder.decision(policy, task.observation(state), reference, None, durations[-1], error)
                    terminal_reason, controller_error = "history_infeasible", str(error)
                    break
                durations.append(perf_counter() - started)
                if recorder is not None:
                    recorder.decision(policy, task.observation(state), reference, proposed, durations[-1])
                if benchmark:
                    audit.decision(policy)
                    if not np.isfinite(proposed).all():
                        terminal_reason = "nonfinite_action"
                        break
                action = np.clip(proposed, -3, 3)
                if benchmark:
                    audit.applied(proposed, action, clipped=not np.array_equal(proposed, action))
                record_applied_action(policy, action)
                transition = task.dynamics.step(action)
                if recorder is not None:
                    recorder.transition(action, action, reference, transition)
                state = transition.observation
                states.append(state.copy())
                actions.append(np.asarray(action).copy())
                if benchmark and not np.isfinite(state).all():
                    terminal_reason = "nonfinite_state"
                    break
                if transition.terminated or transition.truncated:
                    terminal_reason = "terminated" if transition.terminated else "truncated"
                    break
    finally:
        task.close()
    stable = balanced(np.asarray(states)) if states else np.array([], dtype=bool)
    result = {"seed": int(seed), "success": bool(len(stable) == steps and stable[-50:].all()),
              "first_balanced": float(np.argmax(stable)) if stable.any() else float("nan"),
              "steps": len(states), "ms": 1e3 * float(np.mean(durations)) if durations else 0.0}
    if benchmark:
        result.update(audit.fields(terminal_reason, controller_error))
        result["success"] = result["success"] and len(stable) >= 50 and not result["controller_failure"]
    if record_trace:
        result["states"] = np.asarray([initial, *states])
        result["actions"] = np.asarray(actions)
    return result


def run_experiment(config_path=None, methods=None, overrides=None, output=None, *, seeds=None, steps=None):
    """Run fixed configurations and write resumable, per-start paper records."""
    from src.benchmarks.runner import run_task
    return run_task("cartpole", config_path=config_path, methods=methods, seeds=seeds,
                    steps=steps, output=output, overrides=overrides)


if __name__ == "__main__":
    from src.benchmarks.runner import main
    main("cartpole")
