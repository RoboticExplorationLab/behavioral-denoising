"""Release protocol integrity and saved-record resumption, without long DeePC runs."""

from importlib import import_module
import json
from pathlib import Path

import numpy as np
import pytest

from src.benchmarks import runner


@pytest.mark.parametrize("task", runner.TASKS)
def test_frozen_starts_are_the_paper_starts(task):
    settings = runner.load_experiment_config(task)
    starts = runner.experiment_starts(task, import_module(f"experiments.{task}"), settings)
    frozen = json.loads((runner.ROOT / f"data/{task}/test_starts.json").read_text())
    assert [start[0] for start in starts] == [record["seed"] for record in frozen["starts"]]
    assert all(start[2] == frozen["max_steps"] for start in starts)
    if task == "reacher":
        for start, saved in zip(starts, frozen["starts"]):
            np.testing.assert_array_equal(start[1], saved["targets"])
    if task == "rocket":
        assert starts[0][1] == [.2, .9, 0.]
        assert all(start[1] is None for start in starts[1:])


def test_resume_rejects_a_changed_protocol(tmp_path):
    original = {"sha256": "same", "parameters": {"bandwidth": .1}}
    runner.initialize_output(tmp_path, "cartpole", original)
    runner.initialize_output(tmp_path, "cartpole", original)
    with pytest.raises(ValueError, match="Changed protocol"):
        runner.initialize_output(tmp_path, "cartpole", {"sha256": "changed"})


def test_completion_requires_all_recorded_payloads(tmp_path):
    staging, destination = tmp_path / "staging", tmp_path / "seed_1"
    staging.mkdir()
    np.savez_compressed(staging / "trajectory.npz", actions=np.zeros((0, 1)))
    (staging / "result.json").write_text('{"success": false}')
    runner.publish_record(staging, destination, {"protocol_sha256": "protocol"})
    assert runner.completed_record(destination, "protocol")
    with pytest.raises(ValueError, match="another protocol"):
        runner.completed_record(destination, "other")
    (destination / "trajectory.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed completed"):
        runner.completed_record(destination, "protocol")


def test_partial_episode_is_not_a_completed_episode(tmp_path):
    (tmp_path / "metadata.json").write_text('{}')
    np.savez_compressed(tmp_path / "trajectory.npz", actions=np.zeros((0, 1)))
    assert not runner.completed_record(tmp_path, "protocol")


def test_duplicate_or_unknown_methods_fail_before_running(tmp_path):
    for methods in (["select_dpc", "select_dpc_upstream"], ["not_a_method"]):
        with pytest.raises(ValueError, match="distinct methods"):
            runner.run_task("cartpole", methods=methods, output=tmp_path)
    assert not (tmp_path / "protocols").exists()


def test_input_manifest_rejects_unregistered_data(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/manifest.json").write_text('{}')
    (tmp_path / "data/bank.npz").write_bytes(b"not a registered bank")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="differs from"):
        runner.checked_input("data/bank.npz")


def test_json_preserves_nonfinite_parameter_meaning():
    value = runner.json_value({"lambda": np.inf, "array": np.array([1., np.nan])})
    assert value == {"lambda": {"nonfinite": "+Infinity"}, "array": [1., {"nonfinite": "NaN"}]}
    json.dumps(value, allow_nan=False)


def test_cartpole_loop_feeds_back_the_clipped_action(monkeypatch):
    from types import SimpleNamespace
    from experiments import cartpole

    class Task:
        def __init__(self, **options):
            self.dynamics = self

        def reset(self, seed):
            return np.array([0., np.pi, 0., 0.])

        def observation(self, raw):
            return np.array([raw[0], np.sin(raw[1]), np.cos(raw[1]), raw[2], raw[3]])

        def step(self, action):
            np.testing.assert_array_equal(action, [3.])
            return SimpleNamespace(observation=self.reset(0), terminated=False, truncated=False)

        def close(self):
            pass

    class Policy:
        def reset(self, seed):
            self.feedback = []

        def act(self, observation, reference):
            if self.feedback:
                np.testing.assert_array_equal(self.feedback[-1], [3.])
            return np.array([7.])

        def record_applied_action(self, action):
            self.feedback.append(action.copy())

    monkeypatch.setattr(cartpole, "CartpoleTask", Task)
    policy = Policy()
    result = cartpole.run_episode(policy, 0, steps=2, record_trace=True, outcome_protocol="benchmark_v1")
    np.testing.assert_array_equal(result["actions"], [[3.], [3.]])
    assert len(policy.feedback) == 2


def test_output_resolution_canonicalizes_parent_segments_and_symlinks(tmp_path, monkeypatch):
    root = tmp_path / "release"
    (root / "results/paper").mkdir(parents=True)
    (root / "paper_alias").symlink_to(root / "results/paper", target_is_directory=True)
    monkeypatch.setattr(runner, "ROOT", root)
    assert runner.resolve("results/new/../paper") == root / "results/paper"
    assert runner.resolve("paper_alias") == root / "results/paper"
