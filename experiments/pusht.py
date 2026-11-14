"""State-based Push-T with released demonstrations and the fixed validation winners."""
from copy import deepcopy
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from src.data import DeePCDims, prepare_dataset
from src.optimization.hankel import HankelMatrixGenerator
from src.policies.bd_chi2 import BDChi2Policy

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("bd_chi2", "select_dpc")



def test_seeds(configuration):
    import json
    path = ROOT / configuration.get("test_starts", configuration.get("test_starts_file", "data/pusht/test_starts.json"))
    return [record["seed"] for record in json.loads(path.read_text())["starts"]]


class RecordedBDChi2(BDChi2Policy):
    """Capture returned plans; leave the numerical policy unchanged."""

    def reset(self, seed=None):
        super().reset(seed)
        self.query_index = -1
        self.latest_query = None

    def plan(self, history):
        plan = super().plan(history)
        self.query_index += 1
        self.latest_query = {"query_index": self.query_index, "history": history.copy(),
                             "planned_actions": plan.copy(), "diagnostics": self.diagnostics()}
        return plan


def build_policy(data, configuration, method, reference):
    options = deepcopy(method["parameters"])
    past, future = configuration["history"], method["prediction_horizon"]
    execute = configuration["execute_steps"]
    if method["id"] == "bd_chi2":
        inputs, outputs = HankelMatrixGenerator(past, future).generate_hankel_matrices(prepare_dataset(data))
        return RecordedBDChi2(inputs, outputs, DeePCDims(past, future, 6, 2),
                              action_horizon=future, execute_steps=execute, **options)
    if method["id"] == "select_dpc":
        from src.policies.select_dpc_chunks import SelectDPCChunks
        return SelectDPCChunks(data, reference=reference, past=past, horizon=future,
                               execute_steps=execute, **options)
    raise ValueError(f"Unknown method: {method['id']}")


def current_query(policy):
    if isinstance(policy, RecordedBDChi2):
        return policy.query_index
    return policy.diagnostics()["query_index"]


def run_episode(task, policy, environment_seed, policy_seed, maximum_steps):
    """Record one episode. Called only by the explicit run command."""
    raw = task.reset(environment_seed)
    policy.reset(policy_seed)
    states, observations = [np.asarray(raw).copy()], [task.observation(raw)]
    snapshots = [task.dynamics.snapshot()]
    actions, proposed_actions, rewards = [], [], []
    decisions, queries = [], []
    termination_reason, controller_error = "timeout", None
    for step in range(maximum_steps):
        before_query = current_query(policy)
        wall_started, cpu_started = perf_counter(), process_time()
        try:
            proposed = np.asarray(policy.act(observations[-1], task.reference(raw, step)), dtype=float).reshape(2)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            decisions.append({"step": step, "wall_seconds": perf_counter() - wall_started,
                              "cpu_seconds": process_time() - cpu_started, "error": str(error)})
            termination_reason, controller_error = "controller_error", str(error)
            break
        wall_seconds, cpu_seconds = perf_counter() - wall_started, process_time() - cpu_started
        query_index = current_query(policy)
        decision = {"step": step, "wall_seconds": wall_seconds, "cpu_seconds": cpu_seconds,
                    "query_index": query_index, "new_query": query_index != before_query,
                    "diagnostics": policy.diagnostics(), "proposed_action": proposed.copy()}
        decisions.append(decision)
        if decision["new_query"]:
            queries.append({"step": step, "wall_seconds": wall_seconds, "cpu_seconds": cpu_seconds,
                            **deepcopy(policy.latest_query)})
        if not np.isfinite(proposed).all():
            termination_reason = "nonfinite_action"
            controller_error = "Controller returned a nonfinite target position"
            break
        action = np.clip(proposed, 0.0, 512.0)
        policy.record_applied_action(action)
        decision["proposed_action"] = proposed.copy()
        decision["applied_action"] = action.copy()
        decision["diagnostics"] = policy.diagnostics()
        transition = task.dynamics.step(action)
        raw = np.asarray(transition.observation).copy()
        proposed_actions.append(proposed)
        actions.append(action)
        rewards.append(float(transition.reward))
        states.append(raw)
        observations.append(task.observation(raw) if np.isfinite(raw).all() else np.full(6, np.nan))
        snapshots.append(task.dynamics.snapshot())
        if not np.isfinite(raw).all():
            termination_reason, controller_error = "nonfinite_state", "Simulator returned a nonfinite state"
            break
        if transition.terminated:
            termination_reason = "success" if snapshots[-1]["coverage"] > 0.95 else "environment_termination"
            break
        if transition.truncated:
            termination_reason = "timeout"
            break
    success = termination_reason == "success"
    result = {
        "seed": int(environment_seed), "policy_seed": int(policy_seed), "success": success,
        "score": float(max(rewards, default=0.0)),
        "max_coverage": float(max((s["coverage"] for s in snapshots[1:]), default=0.0)),
        "final_coverage": float(snapshots[-1]["coverage"]),
        "steps": len(actions), "time_to_success_seconds": len(actions) / 10 if success else None,
        "termination_reason": termination_reason, "controller_error": controller_error,
        "queries": len(queries), "decisions": len(decisions),
        "clipped_actions": sum(not np.array_equal(a, b) for a, b in zip(actions, proposed_actions)),
        "wall_ms_per_decision": 1000 * np.mean([d["wall_seconds"] for d in decisions]) if decisions else None,
        "cpu_ms_per_decision": 1000 * np.mean([d["cpu_seconds"] for d in decisions]) if decisions else None,
        "wall_ms_per_query": 1000 * np.mean([q["wall_seconds"] for q in queries]) if queries else None,
        "cpu_ms_per_query": 1000 * np.mean([q["cpu_seconds"] for q in queries]) if queries else None,
        "timing_protocol": "online telemetry, not isolated comparative timing",
        "final_diagnostics": policy.diagnostics(),
    }
    arrays = {
        "states": np.asarray(states), "observations": np.asarray(observations),
        "actions": np.asarray(actions).reshape(-1, 2),
        "proposed_actions": np.asarray(proposed_actions).reshape(-1, 2),
        "rewards": np.asarray(rewards), "coverage": np.array([s["coverage"] for s in snapshots]),
        "time": np.arange(len(states)) / 10, "goal_pose": np.asarray(snapshots[0]["goal_pose"]),
    }
    for key in ("agent_velocity", "block_velocity", "block_angular_velocity", "agent_angle",
                "agent_angular_velocity", "n_contacts"):
        arrays[key] = np.array([s[key] for s in snapshots])
    return result, arrays, decisions, queries, task.dynamics.geometry()


def run_fixed_trial(data, configuration, method, seed, cache=None):
    """Reset the recorded seed and execute one fixed-parameter controller."""
    from src.tasks.pusht import PushTTask
    cache = {} if cache is None else cache
    task = PushTTask(**configuration["task"])
    try:
        reference = np.array([0, 0, 256, 256, np.cos(np.pi / 4), np.sin(np.pi / 4)])
        if method["id"] not in cache:
            cache[method["id"]] = build_policy(data, configuration, method, reference)
        return run_episode(task, cache[method["id"]], seed,
                           seed + configuration["evaluation"]["policy_seed_offset"],
                           configuration["task"]["max_episode_steps"])
    finally:
        task.close()


def run_experiment(config_path=None, methods=None, overrides=None, output=None, *, seeds=None, steps=None):
    from src.benchmarks.runner import run_task
    return run_task("pusht", config_path=config_path, methods=methods, seeds=seeds,
                    steps=steps, output=output, overrides=overrides)


if __name__ == "__main__":
    from src.benchmarks.runner import main
    main("pusht")
