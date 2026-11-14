"""Protocol and saved-record checks without running a Push-T benchmark."""

import json
import numpy as np
import pytest

from experiments import pusht
from src.data import TrajectoryData, TrajectoryDataSet, prepare_dataset
from src.dataset_io import load_dataset
from src.benchmarks import runner
from src.interfaces import Step


@pytest.mark.reference_results
def test_released_bank_and_final_configuration_match_frozen_winners(paper_results):
    settings = runner.load_experiment_config("pusht")
    methods = runner.load_method_configs("pusht", settings["methods"])
    path, _ = runner.checked_input(settings["demonstrations"]["data_file"])
    data = load_dataset(path, "pusht")
    assert len(data.dataset) == 206
    assert sum(len(item.input_trajectory) for item in data.dataset) == 25650
    aligned = prepare_dataset(data)
    assert sum(len(item.input_trajectory) for item in aligned.dataset) == 25444
    assert [sum(max(0, len(item.input_trajectory) - settings["history"] - method["prediction_horizon"] + 1)
                for item in aligned.dataset) for method in methods] == [23590, 19058]
    assert settings["task"]["legacy"] is True
    assert pusht.test_seeds(settings) == list(range(300005, 300055))
    frozen = json.loads((paper_results / "pusht/protocol.json").read_text())
    for method in methods:
        winner = frozen["selected_candidates"][method["id"]]
        assert method["parameters"] == winner["method"]["parameters"]
        assert method["prediction_horizon"] == winner["method"]["prediction_horizon"]
        assert settings["execute_steps"] == winner["execute_steps"]


def test_configured_builders_accept_parameters_without_simulation():
    settings = runner.load_experiment_config("pusht")
    methods = runner.load_method_configs("pusht", settings["methods"])
    rng = np.random.default_rng(7)
    episodes = [TrajectoryData(rng.normal(size=(60, 6)), rng.normal(size=(60, 2))) for _ in range(3)]
    data = TrajectoryDataSet(episodes, "fixture", "post_action")
    reference = np.array([0, 0, 256, 256, 2 ** -0.5, 2 ** -0.5])
    for method in methods:
        policy = pusht.build_policy(data, settings, method, reference)
        policy.reset(17)
        assert policy.diagnostics()["dictionary_columns"] > 0


class StubTask:
    """Controlled state stream for testing the recorder, not a simulator."""

    def __init__(self, coverages):
        self.coverages = coverages
        self.dynamics = self

    def reset(self, seed):
        self.index = 0
        self.raw = np.array([50, 50, 100, 100, 0.0])
        return self.raw.copy()

    def observation(self, raw):
        return np.r_[raw[:4], np.cos(raw[4]), np.sin(raw[4])]

    def reference(self, raw, step):
        return np.zeros(6)

    def snapshot(self):
        return {"coverage": self.coverages[self.index], "goal_pose": [256, 256, np.pi / 4],
                "agent_velocity": [0, 0], "block_velocity": [0, 0], "block_angular_velocity": 0,
                "agent_angle": 0, "agent_angular_velocity": 0, "n_contacts": 0}

    def geometry(self):
        return {"control_hz": 10, "agent_radius": 15}

    def step(self, action):
        self.index += 1
        self.raw[:2] = action
        coverage = self.coverages[self.index]
        return Step(self.raw.copy(), min(coverage / 0.95, 1), coverage > 0.95,
                    self.index == len(self.coverages) - 1 and coverage <= 0.95)


class StubPolicy:
    def reset(self, seed):
        self.index = -1
        self.feedback = []
        self.latest_query = None

    def act(self, observation, reference):
        self.index += 1
        self.latest_query = {"query_index": self.index, "planned_actions": np.array([[-1, 513]])}
        return np.array([-1, 513])

    def record_applied_action(self, action):
        self.feedback.append(action.copy())

    def diagnostics(self):
        return {"query_index": self.index}


def test_recording_uses_strict_success_and_retains_raw_clipped_actions(tmp_path):
    task, policy = StubTask([0, 0.95, 0.96]), StubPolicy()
    result, arrays, decisions, queries, geometry = pusht.run_episode(task, policy, 1, 2, 10)
    assert result["success"] and result["steps"] == 2
    assert result["score"] == 1
    assert result["max_coverage"] == 0.96
    assert result["time_to_success_seconds"] == 0.2
    assert result["queries"] == 2
    assert arrays["states"].shape == (3, 5)
    assert arrays["observations"].shape == (3, 6)
    np.testing.assert_array_equal(arrays["actions"], [[0, 512], [0, 512]])
    np.testing.assert_array_equal(arrays["proposed_actions"], [[-1, 513], [-1, 513]])
    np.testing.assert_array_equal(policy.feedback, arrays["actions"])
    assert decisions[0]["new_query"]
    assert len(queries) == 2


def test_exact_threshold_at_timeout_is_not_success():
    result, *_ = pusht.run_episode(StubTask([0, 0.95]), StubPolicy(), 1, 2, 1)
    assert not result["success"]
    assert result["score"] == 1
    assert result["termination_reason"] == "timeout"


def test_invalid_action_does_not_advance_environment_or_discard_error_record():
    class InvalidPolicy(StubPolicy):
        def act(self, observation, reference):
            return np.array([np.nan, 0])
    task = StubTask([0, 0.1])
    result, arrays, decisions, *_ = pusht.run_episode(task, InvalidPolicy(), 1, 2, 1)
    assert result["termination_reason"] == "nonfinite_action"
    assert not result["success"] and result["steps"] == 0 and len(decisions) == 1
    assert arrays["actions"].shape == (0, 2)
    assert task.index == 0





def test_describe_never_builds_or_runs_a_policy(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Inspection must not execute controllers")
    monkeypatch.setattr(pusht, "build_policy", forbidden)
    monkeypatch.setattr(pusht, "run_episode", forbidden)
    monkeypatch.setattr("sys.argv", ["pusht", "describe"])
    runner.main("pusht")
    assert json.loads(capsys.readouterr().out)["task"] == "pusht"
