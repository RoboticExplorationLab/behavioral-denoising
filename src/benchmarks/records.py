"""Export one benchmark episode without rerunning its controller or simulator.

``metadata`` is the experiment metadata (including its ``config`` mapping).
Rocket additionally needs ``metadata['reconstruction']['landing_position']``
with the exact six-vector used by RocketTask, or an explicit ``references``
array. Supply ``reconstruction.hover_input`` to validate the hover-relative cost.
The exporter never infers simulator constants from a trajectory.

Each file is replaced atomically. ``result.json`` is written last and serves as
the completed-record marker. A completed record may contain a controller failure.
"""

from collections.abc import Mapping
from pathlib import Path
import hashlib
import json
import os
import tempfile

import numpy as np


_DIMENSIONS = {"cartpole": (4, 5, 1), "reacher": (8, 8, 2), "rocket": (6, 6, 3)}
_CONTROLLER_FAILURES = {"history_infeasible", "nonfinite_action", "nonfinite_state"}


def _strict_json(value):
    """Return JSON-safe data plus a map preserving nonfinite values explicitly."""
    nonfinite = {}

    def convert(item, path):
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not np.isfinite(item):
            nonfinite[path] = "NaN" if np.isnan(item) else "+Infinity" if item > 0 else "-Infinity"
            return None
        if isinstance(item, Mapping):
            return {str(key): convert(part, f"{path}/{key}") for key, part in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(part, f"{path}/{index}") for index, part in enumerate(item)]
        if isinstance(item, Path):
            return str(item)
        return item

    return convert(value, ""), nonfinite


def _write_json(path, value):
    safe, nonfinite = _strict_json(value)
    if nonfinite:
        safe["nonfinite_values"] = nonfinite
    content = json.dumps(safe, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write(path, lambda stream: stream.write(content.encode("utf-8")))
    return safe


def _atomic_write(path, writer):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _trace(row, task):
    state_dimension, _, action_dimension = _DIMENSIONS[task]
    if "states" not in row or "actions" not in row:
        raise ValueError("Episode has no complete trace; states and actions are required")
    states = np.asarray(row["states"], dtype=float)
    actions = np.asarray(row["actions"], dtype=float)
    if actions.size == 0:
        actions = np.empty((0, action_dimension), dtype=float)
    if states.ndim != 2 or states.shape[1] != state_dimension:
        raise ValueError(f"Expected states shape (T+1, {state_dimension}), got {states.shape}")
    if actions.ndim != 2 or actions.shape[1] != action_dimension:
        raise ValueError(f"Expected actions shape (T, {action_dimension}), got {actions.shape}")
    if len(states) != len(actions) + 1:
        raise ValueError("Trace must include exactly one initial state and one following state per action")
    if "steps" in row and int(row["steps"]) != len(actions):
        raise ValueError("Recorded step count does not match the action trace")
    return states, actions


def _cartpole_observations(states):
    return np.column_stack((states[:, 0], np.sin(states[:, 1]), np.cos(states[:, 1]),
                            states[:, 2], states[:, 3]))


def _reacher_targets(metadata, row):
    config = metadata["config"]
    starts = metadata.get("test_starts", [])
    matching = [start for start in starts if int(start[0]) == int(row["seed"])]
    if len(matching) != 1:
        raise ValueError("Reacher metadata must identify exactly one test start for the episode seed")
    targets = matching[0][1]
    if targets is None:
        seed = config.get("task", {}).get("target_seed", 42)
        angles = np.linspace(3 * np.pi / 4, 2 * np.pi, 4)
        distances = np.random.RandomState(seed).uniform(0.04, 0.15, 4)
        targets = [[0.043, 0.092]]
        targets += [[distance * np.cos(angle), distance * np.sin(angle)]
                    for angle, distance in zip(angles, distances)]
        targets = np.asarray(targets, dtype=float)
        targets[:, 1] = np.minimum(targets[:, 1], 0.09)
    targets = np.asarray(targets, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 2 or not len(targets) or not np.isfinite(targets).all():
        raise ValueError("Reacher targets must be finite [x, y] positions")
    return targets


def _reacher_progress(observations, metadata, row):
    """Replay one target check per saved state, including cyclic wraparound."""
    targets = _reacher_targets(metadata, row)
    tolerance = float(metadata["config"].get("task", {}).get("target_tolerance", 0.005))
    count = len(observations) - 1
    references = np.zeros((count, 8), dtype=float)
    indices = np.empty(count, dtype=np.int64)
    completion_counts = np.zeros(count + 1, dtype=np.int64)
    completion_steps = []
    completed, target_index = 0, 0
    for step in range(count + 1):
        # A nonfinite post-action state exits before the final reference check.
        if step == count and row.get("terminal_reason") == "nonfinite_state":
            completion_counts[step] = completed
            break
        if np.linalg.norm(observations[step, 4:6] - targets[target_index]) < tolerance:
            following_index = (target_index + 1) % len(targets)
            if following_index != target_index:
                completed += 1
                completion_steps.append(step)
            target_index = following_index
        completion_counts[step] = completed
        if step < count:
            indices[step] = target_index
            references[step, 4:6] = targets[target_index]
    return dict(references=references, targets=targets, target_indices=indices,
                completed=completed, final_target_index=target_index,
                completion_counts=completion_counts,
                completion_steps=np.asarray(completion_steps, dtype=np.int64))



def _reacher_timestep(metadata, origin):
    """Read the control timestep from the recorded simulator settings."""
    explicit = metadata.get("reconstruction", {}).get("control_timestep_s")
    recorded, source = None, None
    if isinstance(origin, Mapping) and origin.get("raw_directory"):
        path = Path(origin["raw_directory"]) / "recording_rendering.json"
        if path.is_file():
            rendering = json.loads(path.read_text())
            if rendering.get("task") != "reacher":
                raise ValueError("Reacher timestep must come from a Reacher recording")
            recorded = rendering["timestep"]
            source = {"kind": "recorded_rendering_metadata", "path": str(path),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    timestep = recorded if recorded is not None else explicit
    if timestep is None or not np.isfinite(timestep) or float(timestep) <= 0:
        raise ValueError("Reacher export requires a positive recorded control timestep or reconstruction.control_timestep_s")
    if explicit is not None and not np.isclose(float(explicit), float(timestep), rtol=0, atol=1e-12):
        raise ValueError("Explicit and recorded Reacher timesteps disagree")
    return float(timestep), source or {"kind": "explicit_reconstruction_timestep"}


def _rocket_references(observations, metadata):
    reconstruction = metadata.get("reconstruction", {})
    if "landing_position" not in reconstruction:
        raise ValueError("Rocket requires explicit references or reconstruction.landing_position")
    landing = np.asarray(reconstruction["landing_position"], dtype=float)
    if landing.shape != (6,) or not np.isfinite(landing).all():
        raise ValueError("Rocket landing_position must be the exact finite six-vector used by RocketTask")
    mode = metadata["config"].get("task", {}).get("mode", "hover")
    if mode not in {"hover", "landing"}:
        raise ValueError(f"Unknown Rocket task mode: {mode}")
    references = np.empty((len(observations) - 1, 6), dtype=float)
    approach_hover = True
    for step, observation in enumerate(observations[:-1]):
        if mode == "landing" and abs(observation[0] - landing[0]) < 2 and observation[2] ** 2 + observation[3] ** 2 < 1:
            approach_hover = False
        references[step] = landing
        if mode == "hover" or approach_hover:
            references[step, 1] += 5
    return references


def _cost_terms(observations, actions, references, state_weights, action_weights):
    state_weights = np.asarray(state_weights, dtype=float)
    action_weights = np.asarray(action_weights, dtype=float)
    if state_weights.shape != (observations.shape[1],) or action_weights.shape != (actions.shape[1],):
        raise ValueError("Cost weight dimensions do not match observations/actions")
    errors = observations[:-1] - references
    state_costs = np.sum(errors * errors * state_weights, axis=1)
    action_costs = np.sum(actions * actions * action_weights, axis=1)
    return state_costs, action_costs


def _rocket_hover_actions(actions, metadata, origin):
    """Clip actions using the recorded simulator bounds."""
    bounds = metadata.get("reconstruction", {}).get("action_bounds")
    source = {"kind": "explicit_reconstruction_bounds"}
    if bounds is None:
        if not isinstance(origin, Mapping) or not origin.get("raw_directory"):
            raise ValueError("Rocket hover cost requires reconstruction.action_bounds or a recorded raw_directory")
        path = Path(origin["raw_directory"]) / "recording_rendering.json"
        simulator = json.loads(path.read_text())["simulator_config"]
        limits = [simulator[name] for name in
                  ("main_engine_thrust_limits", "side_engine_thrust_limits", "nozzle_angle_limits")]
        bounds = {"lower": [limit[0] for limit in limits], "upper": [limit[1] for limit in limits]}
        source = {"kind": "recorded_simulator_config", "path": str(path),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    lower, upper = np.asarray(bounds["lower"], dtype=float), np.asarray(bounds["upper"], dtype=float)
    if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or np.any(lower > upper)):
        raise ValueError("Rocket action bounds must be finite ordered three-vectors")
    return np.clip(actions, lower, upper), {"lower": lower.tolist(), "upper": upper.tolist()}, source


def _verify_cost(name, recorded, reconstructed):
    if recorded is not None and np.isfinite(recorded):
        if not np.isclose(float(recorded), reconstructed, rtol=1e-9, atol=1e-7):
            raise ValueError(f"{name} mismatch: recorded {recorded}, trace reconstruction {reconstructed}")


def export_episode(destination, task, row, metadata, expert_row=None, origin=None, *, references=None):
    """Validate and export a traced episode; return the strict-JSON result mapping.

    ``states`` means raw [x, theta, xdot, omega] for Cartpole and the stored
    task observation for Reacher/Rocket. NPZ ``observations`` means task
    observations; Reacher BD policies additionally condition on reference[4:6].
    References and cost terms align with actions and
    ``observations[:-1]``. The final observation remains in the trace.
    """
    if task not in _DIMENSIONS:
        raise ValueError(f"Unknown task: {task}")
    if "config" not in metadata:
        raise ValueError("metadata must include the effective config")
    config = metadata["config"]
    states, actions = _trace(row, task)
    observations = _cartpole_observations(states) if task == "cartpole" else states.copy()
    original = {key: value for key, value in row.items() if key not in {"states", "actions"}}
    result = dict(original)
    result.update(schema_version=1, task=task, record_status="completed", job_failure=False,
                  original_metrics=original, steps=len(actions))
    controller_failure = bool(row.get("controller_failure", False) or row.get("terminal_reason") in _CONTROLLER_FAILURES)
    result["controller_failure"] = controller_failure
    arrays = {"states": states, "observations": observations, "actions": actions}

    if task == "cartpole":
        default_reference = config.get("reference", [0, 0, 1, 0, 0])
        expected_references = np.tile(np.asarray(default_reference, dtype=float), (len(actions), 1))
        state_weights, action_weights = [5, 0, 10000, 0.1, 0.1], [1]
        angles = np.arctan2(np.sin(states[1:, 1]), np.cos(states[1:, 1]))
        stable = (np.abs(angles) <= np.deg2rad(15)) & (np.abs(states[1:, 3]) <= 1) & (np.abs(states[1:, 0]) <= 1)
        horizon = int(config.get("steps", 300))
        result["success"] = bool(len(actions) == horizon and len(stable) >= 50 and stable[-50:].all() and not controller_failure)
    elif task == "reacher":
        progress = _reacher_progress(observations, metadata, row)
        expected_references = progress["references"]
        completed = progress["completed"]
        arrays.update(targets=progress["targets"], target_indices=progress["target_indices"],
                      final_target_index=np.asarray(progress["final_target_index"]),
                      target_completion_counts=progress["completion_counts"],
                      target_completion_steps=progress["completion_steps"])
        if "reached_final" in row and int(row["reached_final"]) != completed:
            raise ValueError(f"Final target count mismatch: recorded {row['reached_final']}, reconstructed {completed}")
        strict_violations = int(np.sum(observations[1:, 5] > config.get("height_limit", 0.1)))
        if "violations_strict" in row and int(row["violations_strict"]) != strict_violations:
            raise ValueError("Strict violation count differs from the trace")
        result.update(reached_final=completed, targets_completed=completed,
                      violations_strict=strict_violations,
                      five_target_success=bool(completed >= 5 and not controller_failure))
        if expert_row is None or "reached_final" not in expert_row:
            raise ValueError("Reacher expert-matching success requires the paired expert's final target count")
        if "seed" in expert_row and int(expert_row["seed"]) != int(row["seed"]):
            raise ValueError("Expert seed does not match the episode seed")
        result["expert_reached_final"] = int(expert_row["reached_final"])
        result["expert_matched_success"] = bool(completed >= result["expert_reached_final"] and not controller_failure)
        result["expert_matched_clean_success_strict"] = result["expert_matched_success"] and strict_violations == 0
        result["success"] = result["five_target_success"]
        result["all_targets_completed"] = result["five_target_success"]
        result["clean_success"] = result["success"] and strict_violations == 0
        result["clean_success_strict"] = result["clean_success"]
        result["clean_success_scope"] = "Full executed rollout; first_five_clean_success_strict is measured only through first-five completion"
        timestep, timestep_source = _reacher_timestep(metadata, origin)
        evaluation_steps = config.get("evaluation", {}).get("steps")
        if type(evaluation_steps) is not int or evaluation_steps <= 0 or evaluation_steps < len(actions):
            raise ValueError("Reacher evaluation.steps must be a positive integer at least as long as the saved trace")
        completion_steps = progress["completion_steps"]
        first_five_step = int(completion_steps[4]) if len(completion_steps) >= 5 else None
        height_excess = np.maximum(observations[1:, 5] - config.get("height_limit", 0.1), 0)
        first_five_violations = None if first_five_step is None else int(np.sum(height_excess[:first_five_step] > 0))
        result.update(first_five_completion_step=first_five_step,
                      first_five_completion_time_s=None if first_five_step is None else first_five_step * timestep,
                      time_margin_s=None if first_five_step is None else (evaluation_steps - first_five_step) * timestep,
                      remaining_steps=None if first_five_step is None else evaluation_steps - first_five_step,
                      first_five_reached=first_five_step is not None,
                      strict_violations_until_first_five=first_five_violations,
                      max_height_excess_until_first_five_m=None if first_five_step is None else float(np.max(height_excess[:first_five_step], initial=0)),
                      first_five_clean_success_strict=bool(result["success"] and first_five_violations == 0),
                      max_height_excess_m=float(np.max(height_excess, initial=0)),
                      control_timestep_s=timestep, evaluation_steps=evaluation_steps,
                      evaluation_duration_s=evaluation_steps * timestep,
                      recorded_duration_s=len(actions) * timestep)
        state_weights, action_weights = config["state_weights"], config["action_weights"]
    else:
        expected_references = _rocket_references(observations, metadata) if references is None else None
        contact = bool(row.get("contact_success", row.get("success", False)))
        settled = bool(row.get("settled", False)) and contact and not controller_failure
        result.update(contact_success=contact, settled=settled, success=settled)
        if row.get("terminal_reason") == "settled" and not contact:
            result["terminal_reason"] = "airborne_sleep"
        state_weights, action_weights = config["state_weights"], config["action_weights"]

    if references is None:
        references = expected_references
    else:
        references = np.asarray(references, dtype=float)
        if task != "rocket" and not np.array_equal(references, expected_references):
            raise ValueError("Explicit references differ from the reconstructed task references")
    if references.shape != (len(actions), observations.shape[1]) or not np.isfinite(references).all():
        raise ValueError("References must be finite and have one observation-sized row per action")
    state_costs, action_costs = _cost_terms(observations, actions, references, state_weights, action_weights)
    trace_cost = float(np.sum(state_costs + action_costs))
    if task != "cartpole":
        _verify_cost("cost", row.get("cost"), trace_cost)
    cost = float("inf") if controller_failure else trace_cost
    if not controller_failure and row.get("cost") is not None and not np.isfinite(row["cost"]):
        # Preserve a recorded numerical failure even when its partial trace is finite.
        cost = float(row["cost"])
    failure_cost = float("inf") if controller_failure else float(row.get("failure_cost", 0))
    result.update(cost=cost, cost_finite=bool(np.isfinite(cost)), trace_cost=trace_cost,
                  state_cost=float(np.sum(state_costs)), action_cost=float(np.sum(action_costs)),
                  primary_success=bool(result["success"]),
                  failure_cost=failure_cost, failure_cost_finite=bool(np.isfinite(failure_cost)))
    if task == "reacher":
        result["tracking_cost"] = cost
        first_five_step = result["first_five_completion_step"]
        result["first_five_completion_cost"] = (None if first_five_step is None else
                                                float(np.sum((state_costs + action_costs)[:first_five_step])))
    arrays.update(references=references, state_costs=state_costs, action_costs=action_costs)
    hover_action_bounds, hover_bounds_source = None, None
    if task == "rocket" and "hover_input" in metadata.get("reconstruction", {}):
        hover = np.asarray(metadata["reconstruction"]["hover_input"], dtype=float)
        if hover.shape != (3,) or not np.isfinite(hover).all():
            raise ValueError("Rocket hover_input must be a finite three-vector")
        hover_actions, hover_action_bounds, hover_bounds_source = _rocket_hover_actions(actions, metadata, origin)
        hover_action_costs = np.sum((hover_actions - hover) ** 2 * np.asarray(action_weights), axis=1)
        hover_cost = float(np.sum(state_costs + hover_action_costs))
        _verify_cost("cost_hover", row.get("cost_hover"), hover_cost)
        result["cost_hover"] = float("inf") if controller_failure else hover_cost
        arrays["hover_action_costs"] = hover_action_costs
        arrays["hover_cost_actions"] = hover_actions
    if "cost_hover" in result:
        result["cost_hover_finite"] = bool(result["cost_hover"] is not None and np.isfinite(result["cost_hover"]))

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    trajectory_path = destination / "trajectory.npz"
    _atomic_write(trajectory_path, lambda stream: np.savez_compressed(stream, **arrays))
    export_metadata = dict(metadata)
    export_metadata.update(record_schema_version=1, task=task, origin=origin,
                           export_status="completed", controller_failure=controller_failure,
                           cost_alignment="pre-action observations; raw cost uses commands sent to dynamics.step; Rocket hover cost clips those commands to recorded bounds before subtracting hover; no dt multiplier or terminal cost",
                           cost_verification="computed_from_trace" if task == "cartpole" else "finite_recorded_cost_checked_against_trace",
                           rocket_hover_cost_verified=bool(task == "rocket" and "hover_input" in metadata.get("reconstruction", {})),
                           state_representation="raw_cartpole_state" if task == "cartpole" else "task_observation",
                           trajectory_sha256=hashlib.sha256(trajectory_path.read_bytes()).hexdigest())
    if task == "reacher":
        export_metadata.update(
            outcome_protocol="First five ordered target completions within the configured evaluation limit, with no controller failure",
            target_count_scope="Cumulative across cycles, including the final-state check; full cyclic rollout is retained",
            first_five_metric_scope="Initial state through the fifth completion; post-completion cycles are excluded from first-five cost and constraint metrics",
            full_rollout_metric_scope="Cost, violations_strict, max_height_excess_m and clean_success_strict cover all executed actions",
            timestep_source=timestep_source,
            control_timestep_s=timestep,
        )
    if hover_action_bounds is not None:
        export_metadata["reconstruction"] = {**metadata.get("reconstruction", {}), "action_bounds": hover_action_bounds}
        export_metadata["hover_cost_bounds_source"] = hover_bounds_source
    if expert_row is not None:
        export_metadata["paired_expert"] = {key: value for key, value in expert_row.items()
                                             if key not in {"states", "actions"}}
    _write_json(destination / "metadata.json", export_metadata)
    return _write_json(destination / "result.json", result)
