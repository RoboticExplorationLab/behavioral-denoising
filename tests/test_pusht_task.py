"""Push-T setup checks: data timing, pinned source, and fixed-action fixtures."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.data import prepare_dataset
from src.dataset_io import load_dataset
from src.tasks.pusht import PushTTask, import_demonstrations, state_features


ROOT = Path(__file__).resolve().parents[1]


def require_simulator():
    pytest.importorskip("pymunk")
    pytest.importorskip("pygame")
    pytest.importorskip("gym")
    pytest.importorskip("shapely")


def test_angles_are_periodic_and_positions_keep_units():
    raw = np.array([[10, 20, 30, 40, 0], [10, 20, 30, 40, 2 * np.pi]])
    features = state_features(raw)
    np.testing.assert_allclose(features, [[10, 20, 30, 40, 1, 0]] * 2, atol=1e-14)
    with pytest.raises(ValueError, match="five coordinates"):
        state_features([1, 2, 3, 4])


def test_import_preserves_episodes_and_drops_only_unmatched_actions(tmp_path):
    zarr = pytest.importorskip("zarr")
    source = tmp_path / "source.zarr"
    raw = zarr.open(str(source), mode="w")
    states = np.zeros((7, 5))
    states[:, 0] = [1, 2, 3, 20, 21, 22, 23]
    actions = np.column_stack((np.arange(7), np.arange(7) + 10))
    raw.create_dataset("data/state", data=states)
    raw.create_dataset("data/action", data=actions)
    raw.create_dataset("meta/episode_ends", data=np.array([3, 7]))
    output = tmp_path / "demonstrations.npz"
    manifest = import_demonstrations(source, output)
    loaded = load_dataset(output, "pusht")
    assert loaded.timing == "pre_action"
    assert loaded.episode_ids == ["0", "1"]
    assert loaded.dataset[0].terminal_observation is None
    aligned = prepare_dataset(loaded)
    np.testing.assert_array_equal(aligned.dataset[0].state_trajectory[:, 0], [2, 3])
    np.testing.assert_array_equal(aligned.dataset[1].state_trajectory[:, 0], [21, 22, 23])
    np.testing.assert_array_equal(aligned.dataset[0].input_trajectory, actions[:2])
    np.testing.assert_array_equal(aligned.dataset[1].input_trajectory, actions[3:6])
    assert manifest["recorded_pairs"] == 7
    assert manifest["aligned_pairs"] == 5
    assert manifest["dropped_unmatched_actions"] == 2


def test_vendored_sources_match_recorded_upstream_bytes():
    source = ROOT / "third_party/diffusion_policy"
    manifest = json.loads((source / "UPSTREAM.json").read_text())
    for name, record in manifest["files"].items():
        path = source / name
        assert not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_legacy_reset_and_fixed_actions_match_released_recording():
    require_simulator()
    data = load_dataset(ROOT / "data/pusht/demonstrations.npz", "pusht")
    task = PushTTask()
    try:
        for seed in (0, 50, 205):
            state = task.reset(seed)
            expected = data.dataset[seed].state_trajectory[0]
            np.testing.assert_allclose(task.observation(state), expected, rtol=0, atol=3e-5)
        task.reset(0)
        for index in range(2):
            transition = task.dynamics.step(data.dataset[0].input_trajectory[index])
            np.testing.assert_allclose(task.observation(transition.observation),
                                       data.dataset[0].state_trajectory[index + 1], rtol=0, atol=3e-5)
            assert transition.reward == pytest.approx(min(transition.info["coverage"] / 0.95, 1.0))
            assert transition.terminated == (transition.info["coverage"] > 0.95)
            assert not transition.truncated
        assert task.dynamics.geometry()["legacy"] is True
        assert task.dynamics.geometry()["agent_radius"] == 15
        assert task.dynamics.snapshot()["time_seconds"] == 0.2
    finally:
        task.close()


def test_timeout_and_action_contract():
    require_simulator()
    task = PushTTask(max_episode_steps=1)
    try:
        state = task.reset(0)
        with pytest.raises(ValueError, match="Clip the command"):
            task.dynamics.step([-1, 1])
        with pytest.raises(ValueError, match="finite"):
            task.dynamics.step([np.nan, 1])
        transition = task.dynamics.step(state[:2])
        assert transition.truncated and not transition.terminated
        assert transition.info["termination_reason"] == "timeout"
        with pytest.raises(RuntimeError, match="Reset"):
            task.dynamics.step(state[:2])
    finally:
        task.close()


def test_released_manifest_matches_loaded_data():
    directory = ROOT / "data/pusht"
    manifest = json.loads((directory / "upstream_manifest.json").read_text())
    path = directory / manifest["dataset_file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["dataset_sha256"]
    dataset = load_dataset(path, "pusht")
    assert len(dataset.dataset) == manifest["episodes"] == 206
    assert sum(len(episode.input_trajectory) for episode in dataset.dataset) == 25650
    assert sum(len(episode.input_trajectory) for episode in prepare_dataset(dataset).dataset) == 25444
