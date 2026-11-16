"""State-based Push-T with released human demonstrations."""
from copy import deepcopy
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from src.data import DeePCDims
from src.policies.bd_chi2 import BDChi2Policy
from src.tasks.pusht_coordinates import PushTCoordinates

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


class CoordinateBDChi2(RecordedBDChi2):
    """Use shared Push-T coordinates while exposing physical actions/observations."""

    def __init__(self, *args, coordinates=None, **kwargs):
        self.coordinates = (coordinates if isinstance(coordinates, PushTCoordinates)
                            else PushTCoordinates("world" if coordinates is None else coordinates))
        super().__init__(*args, **kwargs)

    @property
    def normalization(self):
        return {"history_mean": self._z_mean.copy(), "history_scale": self._z_scale.copy(),
                "action_mean": self._a_mean.copy(), "action_scale": self._a_scale.copy(),
                "convention": "Per dictionary coordinate, including separate history/action lags"}

    def act(self, observation, reference=None):
        # plan() is called only at a chunk boundary. Its anchor is then frozen
        # in the decoded world chunk, even while later observations change.
        self._planning_observation = np.asarray(observation, dtype=float).copy()
        return super().act(self.coordinates.observations(observation), reference)

    def plan(self, history):
        anchor = self._planning_observation
        if self.coordinates.name == "block_relative" and self._include_past_inputs:
            history = history.copy()
            count = self._T_past * self._m
            past_actions = history[:count].reshape(self._T_past, self._m)
            history[:count] = self.coordinates.encode_actions(past_actions, anchor).reshape(-1)
        coordinate_plan = super().plan(history)
        world_plan = self.coordinates.decode_actions(coordinate_plan, anchor)
        self.latest_query.update(
            representation=self.coordinates.name, coordinate_actions=coordinate_plan.copy(),
            planned_actions=world_plan.copy(), action_anchor_observation=anchor.copy())
        return world_plan


def effective_coordinates(configuration, method):
    """A method override selects the same shared representation as the task default."""
    # Read older BD-only configurations when neither shared setting is present.
    fallback = configuration.get("bd_coordinates", "world") if method["id"] == "bd_chi2" else "world"
    task_default = configuration["task"].get("representation", fallback)
    return PushTCoordinates(method.get("representation", task_default))


def build_policy(data, configuration, method, reference):
    options = deepcopy(method["parameters"])
    past, future = configuration["history"], method["prediction_horizon"]
    execute = configuration["execute_steps"]
    coordinates = effective_coordinates(configuration, method)
    if method["id"] == "bd_chi2":
        inputs, outputs = coordinates.dictionary(
            data, past, future, transform_past_inputs=options.get("include_past_inputs", True))
        return CoordinateBDChi2(inputs, outputs, DeePCDims(past, future, 6, 2), coordinates=coordinates,
                                action_horizon=future, execute_steps=execute, **options)
    if method["id"] == "select_dpc":
        from src.policies.select_dpc_chunks import SelectDPCChunks
        return SelectDPCChunks(data, reference=reference, past=past, horizon=future,
                               execute_steps=execute, coordinates=coordinates, **options)
    raise ValueError(f"Unknown method: {method['id']}")


def current_query(policy):
    if isinstance(policy, RecordedBDChi2):
        return policy.query_index
    return policy.diagnostics()["query_index"]


def run_episode(task, policy, environment_seed, policy_seed, maximum_steps):
    """Record one episode. Called only by the explicit run command."""
    raw = task.reset(environment_seed)
    policy.reset(policy_seed)
    coordinates = getattr(policy, "coordinates", PushTCoordinates())
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
                    "diagnostics": policy.diagnostics(), "proposed_action": proposed.copy(),
                    "representation": coordinates.name}
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
        latest_query = policy.latest_query
        if latest_query is not None and "action_anchor_observation" in latest_query:
            anchor = latest_query["action_anchor_observation"]
            decision["action_anchor_observation"] = np.asarray(anchor).copy()
            decision["coordinate_proposed_action"] = coordinates.encode_actions(proposed, anchor)
            decision["coordinate_applied_action"] = coordinates.encode_actions(action, anchor)
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
        "representation": coordinates.name,
        "normalization": getattr(policy, "normalization", None),
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
    # Physical arrays retain the original schema; policy features are separate.
    physical_observations = arrays["observations"]
    policy_observations = np.full_like(physical_observations, np.nan)
    finite = np.isfinite(physical_observations).all(axis=1)
    policy_observations[finite] = coordinates.observations(physical_observations[finite])
    arrays["policy_observations"] = policy_observations
    return result, arrays, decisions, queries, task.dynamics.geometry()


def run_fixed_trial(data, configuration, method, seed, cache=None):
    """Reset the recorded seed and execute one fixed-parameter controller."""
    from src.tasks.pusht import PushTTask
    cache = {} if cache is None else cache
    task = PushTTask(**configuration["task"])
    try:
        reference = np.array([0, 0, 256, 256, np.cos(np.pi / 4), np.sin(np.pi / 4)])
        # A shared cache must never reuse the other representation or parameters.
        import json
        key = (id(data), json.dumps({"method": method, "history": configuration["history"],
                          "execute_steps": configuration["execute_steps"],
                          "representation": effective_coordinates(configuration, method).name}, sort_keys=True))
        if key not in cache:
            cache[key] = build_policy(data, configuration, method, reference)
        return run_episode(task, cache[key], seed,
                           seed + configuration["evaluation"]["policy_seed_offset"],
                           configuration["task"]["max_episode_steps"])
    finally:
        task.close()


def run_experiment(config_path=None, methods=None, overrides=None, output=None, *, seeds=None, steps=None,
                   representation=None):
    from src.benchmarks.runner import run_task
    return run_task("pusht", config_path=config_path, methods=methods, seeds=seeds,
                    steps=steps, output=output, overrides=overrides, representation=representation)


if __name__ == "__main__":
    from src.benchmarks.runner import main
    main("pusht")
