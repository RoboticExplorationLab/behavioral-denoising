"""State-based Push-T adapter and importer for the released human demonstrations.

The simulator under third_party/ is loaded
only when a task is constructed, so other tasks do not need Push-T dependencies.
"""

from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np

from src.data import TrajectoryData, TrajectoryDataSet, prepare_dataset
from src.dataset_io import save_dataset
from src.interfaces import Dynamics, Step, Task


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_REVISION = "5ba07ac6661db573af695b419a7947ecb704690f"
DATA_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
GOAL_POSE = np.array([256.0, 256.0, np.pi / 4])


def state_features(raw_state):
    """Map (..., 5) positions/angle to (..., 6) features without an angle seam."""
    raw_state = np.asarray(raw_state, dtype=float)
    if raw_state.ndim < 1 or raw_state.shape[-1] != 5 or not np.isfinite(raw_state).all():
        raise ValueError("Push-T state must be finite with five coordinates")
    return np.concatenate((raw_state[..., :4], np.cos(raw_state[..., 4:5]),
                           np.sin(raw_state[..., 4:5])), axis=-1)


def _simulator_module():
    source = ROOT / "third_party" / "diffusion_policy"
    if not (source / "diffusion_policy/env/pusht/pusht_env.py").is_file():
        raise ImportError("The vendored Push-T simulator is missing from third_party/diffusion_policy")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        module = importlib.import_module("diffusion_policy.env.pusht.pusht_env")
    except ImportError as error:
        raise ImportError("Push-T requires the independent environment in requirements/pusht.txt") from error
    if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
        raise ImportError("An external diffusion_policy module shadows the vendored simulator")
    return module


class PushTDynamics(Dynamics):
    """Push-T dynamics using the dataset-compatible simulator reset."""

    def __init__(self, max_episode_steps=300, legacy=True, render_size=512):
        if not isinstance(max_episode_steps, int) or max_episode_steps < 1:
            raise ValueError("max_episode_steps must be a positive integer")
        module = _simulator_module()
        self.env = module.PushTEnv(legacy=legacy, render_size=render_size, render_action=False)
        self._shape_geometry = module.pymunk_to_shapely
        self.max_episode_steps = max_episode_steps
        self.steps = 0
        self.control_hz = self.env.control_hz
        self.dt = 1.0 / self.control_hz
        self.legacy = bool(legacy)
        self._finished = False

    def reset(self, seed=None):
        self.env.seed(seed)
        self.env.reset()
        self.steps = 0
        self._finished = False
        return self.raw_state()

    def raw_state(self):
        """Preserve the simulator's unwrapped block angle in the saved raw state."""
        return np.array([*self.env.agent.position, *self.env.block.position, self.env.block.angle])

    def coverage(self):
        goal_body = self.env._get_goal_pose_body(self.env.goal_pose)
        goal = self._shape_geometry(goal_body, self.env.block.shapes)
        block = self._shape_geometry(self.env.block, self.env.block.shapes)
        return float(goal.intersection(block).area / goal.area)

    def snapshot(self):
        """State needed for offline geometry, motion and contact diagnostics."""
        info = self.env._get_info()
        return {
            "raw_state": self.raw_state(),
            "agent_velocity": np.array(self.env.agent.velocity),
            "block_velocity": np.array(self.env.block.velocity),
            "block_angular_velocity": float(self.env.block.angular_velocity),
            "agent_angle": float(self.env.agent.angle),
            "agent_angular_velocity": float(self.env.agent.angular_velocity),
            "coverage": self.coverage(),
            "n_contacts": int(info["n_contacts"]),
            "goal_pose": np.array(self.env.goal_pose),
            "step": self.steps,
            "time_seconds": self.steps * self.dt,
        }

    def geometry(self):
        return {
            "block_polygons": [np.array(shape.get_vertices()).tolist() for shape in self.env.block.shapes],
            "block_center_of_gravity": list(self.env.block.center_of_gravity),
            "agent_radius": float(next(iter(self.env.agent.shapes)).radius),
            "goal_pose": self.env.goal_pose.tolist(),
            "workspace": [0.0, 512.0, 0.0, 512.0],
            "success_threshold": float(self.env.success_threshold),
            "control_hz": self.control_hz,
            "physics_hz": self.env.sim_hz,
            "legacy": self.legacy,
        }

    def step(self, action):
        if self._finished:
            raise RuntimeError("Reset Push-T before stepping a completed episode")
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Push-T action must be a finite two-dimensional target position")
        if np.any(action < 0) or np.any(action > 512):
            raise ValueError("Clip the command to [0, 512] before stepping Push-T")
        _, reward, succeeded, _ = self.env.step(action)
        self.steps += 1
        timeout = self.steps >= self.max_episode_steps and not succeeded
        self._finished = bool(succeeded or timeout)
        info = self.snapshot()
        info["success"] = bool(succeeded)
        info["termination_reason"] = "success" if succeeded else "timeout" if timeout else None
        return Step(self.raw_state(), float(reward), bool(succeeded), timeout, info)

    def close(self):
        self.env.close()


class PushTTask(Task):
    observation_dim = 6
    action_dim = 2

    def __init__(self, max_episode_steps=300, legacy=True, render_size=512, dynamics=None):
        super().__init__(dynamics if dynamics is not None else PushTDynamics(
            max_episode_steps=max_episode_steps, legacy=legacy, render_size=render_size))

    def observation(self, raw):
        return state_features(raw)

    def reference(self, raw=None, step=0):
        return state_features(np.r_[np.zeros(2), GOAL_POSE])


def import_demonstrations(zarr_path, output_path, *, archive_path=None, manifest_path=None):
    """Import every released episode with its original ordering and timing.

    The source stores the state before each command and omits the terminal
    observation. The standard prepare_dataset function therefore drops the last
    action of each episode when creating post-action controller pairs.
    """
    import zarr

    zarr_path, output_path = Path(zarr_path), Path(output_path)
    saved = zarr.open(str(zarr_path), mode="r")
    raw_states = np.asarray(saved["data/state"])
    actions = np.asarray(saved["data/action"])
    episode_ends = np.asarray(saved["meta/episode_ends"])
    if (raw_states.ndim != 2 or raw_states.shape[1] != 5 or actions.shape != (len(raw_states), 2)
            or episode_ends.ndim != 1 or not len(episode_ends)
            or not np.issubdtype(episode_ends.dtype, np.integer)
            or episode_ends[-1] != len(actions) or np.any(np.diff(np.r_[0, episode_ends]) <= 1)):
        raise ValueError("Invalid released Push-T state/action arrays or episode boundaries")
    observations = state_features(raw_states)
    trajectories = [TrajectoryData(observations[start:end], actions[start:end])
                    for start, end in zip(np.r_[0, episode_ends[:-1]], episode_ends)]
    identities = [str(index) for index in range(len(trajectories))]
    data = TrajectoryDataSet(trajectories, "diffusion_policy_pusht_cchi_v7", "pre_action", identities)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(output_path, data, "pusht")
    prepared = prepare_dataset(data)
    lengths = np.diff(np.r_[0, episode_ends])
    source_files = {str(path.relative_to(zarr_path)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(zarr_path.rglob("*")) if path.is_file()}
    manifest = {
        "dataset": "Diffusion Policy released Push-T cchi_v7",
        "source_url": DATA_URL,
        "imported_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_repository": "https://github.com/real-stanford/diffusion_policy",
        "upstream_revision": UPSTREAM_REVISION,
        "source_schema": {name: {"shape": list(saved["data"][name].shape),
                                  "dtype": str(saved["data"][name].dtype)}
                          for name in sorted(saved["data"].array_keys())},
        "source_files_sha256": source_files,
        "raw_state_coordinates": ["agent_x", "agent_y", "block_x", "block_y", "block_angle"],
        "observation_coordinates": ["agent_x", "agent_y", "block_x", "block_y", "cos_block_angle", "sin_block_angle"],
        "action_coordinates": ["agent_target_x", "agent_target_y"],
        "position_units": "pixels",
        "angle_units": "radians",
        "control_hz": 10,
        "stored_timing": "pre_action",
        "alignment": "Pair action[k] with state[k+1] within each episode; drop unmatched final action",
        "terminal_observations_available": False,
        "episodes": len(trajectories),
        "recorded_pairs": len(actions),
        "aligned_pairs": sum(len(trajectory.input_trajectory) for trajectory in prepared.dataset),
        "dropped_unmatched_actions": len(trajectories),
        "episode_lengths": lengths.tolist(),
        "episode_ids": identities,
        "selection": "All released episodes, original order; no filtering or train cap",
        "upstream_training_note": "The upstream lowdim configuration caps training at 90 episodes; this study uses the full released bank for both methods",
        "mode_labels": None,
        "legacy_reset": True,
        "window_counts": {f"past_2_future_{future}": int(np.maximum(lengths - 2 - future, 0).sum())
                          for future in (8, 30)},
        "dataset_file": output_path.name,
        "dataset_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    if archive_path is not None:
        archive_path = Path(archive_path)
        manifest["source_archive"] = {"file": archive_path.name, "bytes": archive_path.stat().st_size,
                                      "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest()}
    manifest_path = Path(manifest_path) if manifest_path is not None else output_path.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
