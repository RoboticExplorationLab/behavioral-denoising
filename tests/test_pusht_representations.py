"""Push-T representation choices preserve physical contracts and legacy behavior."""

from copy import deepcopy
import json
import sys

import cvxpy as cp
import numpy as np
import pytest

from experiments import pusht
from src.benchmarks import reporting, runner
from src.data import DeePCDims, TrajectoryData, TrajectoryDataSet, prepare_dataset
from src.interfaces import Step
from src.optimization.deepc_qp import DeePCQP
from src.optimization.hankel import HankelMatrixGenerator
from src.policies.bd_chi2 import BDChi2Policy
from src.policies.select_dpc_chunks import SelectDPCChunks
from src.tasks.pusht import PushTTask, state_features
from src.tasks.pusht_coordinates import PushTCoordinates


REFERENCE = np.r_[0., 0., 256., 256., np.cos(np.pi / 4), np.sin(np.pi / 4)]
SELECT_OPTIONS = dict(g_l1=.1, columns=32, iterations=2, output_weight=10.,
                      input_weight=.01, slack=100., g_projection=.1)
BD_OPTIONS = dict(include_past_inputs=False, neighbors=32, bandwidth=.4,
                  mode_width=.3, iterations=5, slope_ridge=1e-6,
                  extrapolation_limit=3., diffusion_steps=30,
                  sampling_scheme="sigma", smoothing=0.)


def example_data():
    rng = np.random.default_rng(17)
    episodes = []
    for _ in range(2):
        actions = rng.uniform(220., 300., (24, 2))
        positions = rng.uniform(220., 300., (24, 4))
        angles = rng.uniform(.3, 1.2, 24)
        episodes.append(TrajectoryData(np.column_stack((positions, np.cos(angles), np.sin(angles))), actions))
    return TrajectoryDataSet(episodes, "synthetic_push_t", "post_action")


def settings(representation=None):
    task = {} if representation is None else {"representation": representation}
    return {"history": 2, "execute_steps": 2, "task": task}


def method(name, representation=None):
    result = {"id": name, "prediction_horizon": 3,
              "parameters": deepcopy(BD_OPTIONS if name == "bd_chi2" else SELECT_OPTIONS)}
    if representation is not None:
        result["representation"] = representation
    return result


@pytest.mark.parametrize("name,expected", [("bd_chi2", "block_relative"), ("select_dpc", "world")])
def test_legacy_bd_coordinates_only_affects_bd_when_shared_settings_are_absent(name, expected):
    configuration = settings()
    configuration["bd_coordinates"] = "block_relative"
    chosen = method(name)
    assert pusht.effective_coordinates(configuration, chosen).name == expected
    configuration["task"]["representation"] = "world"
    assert pusht.effective_coordinates(configuration, chosen).name == "world"
    chosen["representation"] = "block_relative"
    assert pusht.effective_coordinates(configuration, chosen).name == "block_relative"


def test_legacy_relative_bd_configuration_matches_new_shared_setting():
    data = example_data()
    legacy = settings()
    legacy["bd_coordinates"] = "block_relative"
    chosen = method("bd_chi2")
    old_setting = pusht.build_policy(data, legacy, chosen, REFERENCE)
    shared_setting = pusht.build_policy(data, settings("block_relative"), chosen, REFERENCE)
    observations = data.dataset[0].state_trajectory[:5]
    np.testing.assert_array_equal(propose(old_setting, observations), propose(shared_setting, observations))


def propose(policy, observations, seed=1300005):
    policy.reset(seed)
    actions = []
    for observation in observations:
        actions.append(policy.act(observation, REFERENCE))
        policy.record_applied_action(np.clip(actions[-1], 235., 270.))
    return np.asarray(actions)


@pytest.mark.parametrize("name", ["bd_chi2", "select_dpc"])
def test_world_factory_matches_direct_legacy_controller_and_method_override(name):
    data = example_data()
    if name == "bd_chi2":
        inputs, outputs = HankelMatrixGenerator(2, 3).generate_hankel_matrices(prepare_dataset(data))
        direct = BDChi2Policy(inputs, outputs, DeePCDims(2, 3, 6, 2),
                              action_horizon=3, execute_steps=2, **BD_OPTIONS)
    else:
        direct = SelectDPCChunks(data, reference=REFERENCE, past=2, horizon=3,
                                 execute_steps=2, **SELECT_OPTIONS)
    observations = data.dataset[0].state_trajectory[:5]
    expected = propose(direct, observations)
    for configuration, chosen in ((settings(), method(name)),
                                   (settings("world"), method(name)),
                                   (settings("block_relative"), method(name, "world"))):
        controller = pusht.build_policy(data, configuration, chosen, REFERENCE)
        assert controller.coordinates.name == "world"
        np.testing.assert_array_equal(propose(controller, observations), expected)


def test_relative_bd_reproduces_prechange_fixed_frame_fixture():
    # Captured from the previous dictionary transformation and fixed-frame
    # adapter before the shared representation refactor.
    # The fixture contains four plans, moving block poses and clipped feedback.
    expected = np.array([
        [249.23312459807204, 322.1595970787603],
        [223.5907359625005, 291.44755346974614],
        [254.25315992748867, 268.90744867991845],
        [278.9108696410629, 221.22317252773692],
        [240.32327310341114, 268.3316683651144],
        [226.69893557497144, 299.8210944738381],
        [286.5876899600563, 222.94218861338598],
    ])
    data = example_data()
    for configuration, chosen in ((settings("block_relative"), method("bd_chi2")),
                                   (settings("world"), method("bd_chi2", "block_relative"))):
        controller = pusht.build_policy(data, configuration, chosen, REFERENCE)
        assert controller.coordinates.name == "block_relative"
        actual = propose(controller, data.dataset[0].state_trajectory[:7])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


def test_relative_bd_with_past_inputs_uses_applied_world_actions_in_new_frame():
    data = example_data()
    chosen = method("bd_chi2")
    chosen["parameters"]["include_past_inputs"] = True
    controller = pusht.build_policy(data, settings("block_relative"), chosen, REFERENCE)
    observations = data.dataset[0].state_trajectory[:3]
    controller.act(observations[0])
    controller.record_applied_action([240., 250.])
    controller.act(observations[1])
    controller.record_applied_action([260., 270.])
    controller.act(observations[2])
    history = controller.latest_query["history"]
    applied = np.array([[240., 250.], [260., 270.]])
    np.testing.assert_array_equal(history[:4].reshape(2, 2),
                                  controller.coordinates.encode_actions(applied, observations[2]))
    np.testing.assert_array_equal(history[4:].reshape(2, 6),
                                  controller.coordinates.observations(observations[1:]))
    # The adapter transforms only the planning history, not its physical buffer.
    np.testing.assert_array_equal(controller._u_past[:2], applied[1])


@pytest.mark.parametrize("name", ["bd_chi2", "select_dpc"])
def test_chunk_world_targets_do_not_follow_later_block_motion(name):
    data = example_data()
    controller = pusht.build_policy(data, settings("block_relative"), method(name), REFERENCE)
    observation = data.dataset[0].state_trajectory[0].copy()
    controller.reset(1300005)
    first = controller.act(observation, REFERENCE)
    query = deepcopy(controller.latest_query)
    assert query["planned_actions"] is not None
    expected_second = query["planned_actions"][1].copy()
    controller.record_applied_action(np.clip(first, 235., 270.))
    moved = observation.copy()
    moved[:4] += [37., -11., 90., 60.]
    moved[4:6] = [0., 1.]
    second = controller.act(moved, REFERENCE)
    np.testing.assert_array_equal(second, expected_second)
    assert controller.latest_query["query_index"] == query["query_index"]
    np.testing.assert_array_equal(controller.latest_query["action_anchor_observation"], observation)


def relative_select(data=None):
    return SelectDPCChunks(example_data() if data is None else data, reference=REFERENCE,
                           past=2, horizon=3, execute_steps=2,
                           coordinates=PushTCoordinates("block_relative"), **SELECT_OPTIONS)


def test_relative_select_reencodes_actual_history_in_each_new_planning_frame():
    data = example_data()
    controller = relative_select(data)
    observations = data.dataset[0].state_trajectory[:3]
    controller.act(observations[0])
    first_query = controller.latest_query
    np.testing.assert_array_equal(first_query["history_actions"], np.tile(observations[0][:2], (2, 1)))
    assert controller.record_applied_action([240., 250.])
    assert not controller.record_applied_action([10., 20.])
    controller.act(observations[1])
    controller.record_applied_action([260., 270.])
    controller.act(observations[2])
    query = controller.latest_query
    applied = np.array([[240., 250.], [260., 270.]])
    np.testing.assert_array_equal(query["history_actions"], applied)
    np.testing.assert_array_equal(query["history_observations"], observations[1:])
    np.testing.assert_allclose(query["history_policy_actions"],
                               controller.coordinates.encode_actions(applied, observations[2]),
                               rtol=0, atol=2e-13)
    np.testing.assert_allclose(query["history_policy_observations"],
                               controller.coordinates.observations(observations[1:]), rtol=0, atol=1e-13)
    assert not np.allclose(query["history_policy_actions"],
                           controller.coordinates.encode_actions(applied, observations[0]))
    assert query["predicted_observations"] is None
    assert query["predicted_policy_observations"].shape == (3, 6)
    # A caller inspecting recorded plans cannot mutate the next action.
    expected_next = controller.latest_query["commanded_chunk"][1]
    query["commanded_chunk"][:] = -1000
    controller.record_applied_action([240., 240.])
    np.testing.assert_array_equal(controller.act(observations[0]), expected_next)


def test_relative_select_qp_retains_physical_cost_and_workspace():
    controller = relative_select()
    observation = example_data().dataset[0].state_trajectory[0]
    controller.act(observation)
    query = controller.latest_query
    normalization = controller.normalization
    world_targets = np.array([[0., 0.], [512., 512.], [256., 200.], [-5., 250.], [300., 520.]])
    native = controller.coordinates.encode_actions(world_targets, observation)
    standardized = (native - normalization["action_mean"]) / normalization["action_scale"]
    difference = standardized - query["input_reference"]
    actual = np.einsum("ni,ij,nj->n", difference, query["input_cost_matrix"], difference)
    expected = SELECT_OPTIONS["input_weight"] * np.sum(
        ((world_targets - normalization["world_action_mean"]) / normalization["world_action_scale"]) ** 2, axis=1)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    feasible = np.all(standardized @ query["input_constraint_matrix"].T <= query["input_constraint_bound"] + 1e-10, axis=1)
    np.testing.assert_array_equal(feasible, [True, True, True, False, False])


def test_relative_select_failed_plan_holds_world_position_across_block_motion(monkeypatch):
    controller = relative_select()
    observations = example_data().dataset[0].state_trajectory[:2]
    monkeypatch.setattr(DeePCQP, "solve", lambda solver: cp.INFEASIBLE)
    with pytest.warns(RuntimeWarning):
        first = controller.act(observations[0])
    np.testing.assert_array_equal(first, observations[0][:2])
    controller.record_applied_action([240., 250.])
    moved = observations[1].copy()
    moved[2:4] += [100., -70.]
    moved[4:6] = [0., 1.]
    np.testing.assert_array_equal(controller.act(moved), observations[0][:2])
    assert controller.diagnostics()["action_source"] == "hold_position"
    assert not controller.latest_query["plan_valid"]
    assert controller.latest_query["planned_actions"] is None


def test_task_representation_does_not_change_physical_observation_or_reference():
    raw = np.array([110., 220., 100., 200., np.pi / 2])
    world = PushTTask(dynamics=object(), representation="world")
    relative = PushTTask(dynamics=object(), representation="block_relative")
    np.testing.assert_array_equal(relative.observation(raw), world.observation(raw))
    np.testing.assert_array_equal(relative.reference(raw), world.reference(raw))
    np.testing.assert_allclose(relative.policy_observation(raw), [20., -10., 100., 200., 0., 1.], atol=1e-13)
    np.testing.assert_array_equal(world.policy_observation(raw), world.observation(raw))


class RecordingTask:
    """Physical pose stream with deterministic block motion; no simulator."""

    def __init__(self, representation):
        self.coordinates = PushTCoordinates(representation)
        self.dynamics = self

    def reset(self, seed):
        self.index = 0
        self.raw = np.array([250., 270., 252., 240., .3])
        return self.raw.copy()

    def observation(self, raw):
        return state_features(raw)

    def reference(self, raw, step):
        return REFERENCE.copy()

    def policy_observation(self, raw):
        return self.coordinates.observations(self.observation(raw))

    def snapshot(self):
        return {"coverage": .2, "goal_pose": [256, 256, np.pi / 4], "agent_velocity": [0, 0],
                "block_velocity": [0, 0], "block_angular_velocity": 0, "agent_angle": 0,
                "agent_angular_velocity": 0, "n_contacts": 0}

    def geometry(self):
        return {"control_hz": 10, "agent_radius": 15}

    def step(self, action):
        self.index += 1
        self.raw[:2] = action
        self.raw[2] += 2
        self.raw[4] += .02
        return Step(self.raw.copy(), .2 / .95, False, self.index == 3)


@pytest.mark.parametrize("task_representation,method_representation", [
    ("world", "block_relative"), ("block_relative", "world")])
def test_recording_keeps_physical_arrays_and_effective_method_coordinates(task_representation, method_representation):
    task = RecordingTask(task_representation)
    controller = pusht.build_policy(example_data(), settings(task_representation),
                                    method("bd_chi2", method_representation), REFERENCE)
    result, arrays, decisions, queries, _ = pusht.run_episode(task, controller, 17, 1300005, 3)
    assert result["representation"] == method_representation
    np.testing.assert_array_equal(arrays["observations"], state_features(arrays["states"]))
    expected = PushTCoordinates(method_representation).observations(arrays["observations"])
    np.testing.assert_array_equal(arrays["policy_observations"], expected)
    np.testing.assert_array_equal(arrays["actions"], np.clip(arrays["proposed_actions"], 0., 512.))
    assert len(queries) == 2 and len(decisions) == 3


def test_trial_cache_separates_representation_parameters_and_data(monkeypatch):
    class Task:
        def __init__(self, **options):
            pass

        def close(self):
            pass

    monkeypatch.setattr("src.tasks.pusht.PushTTask", Task)
    monkeypatch.setattr(pusht, "run_episode", lambda task, policy, *args: policy)
    data, cache = example_data(), {}
    configuration = settings("world")
    configuration["evaluation"] = {"policy_seed_offset": 1000000}
    configuration["task"]["max_episode_steps"] = 300
    chosen = method("bd_chi2")
    first = pusht.run_fixed_trial(data, configuration, chosen, 1, cache)
    assert pusht.run_fixed_trial(data, configuration, chosen, 2, cache) is first
    relative = deepcopy(configuration)
    relative["task"]["representation"] = "block_relative"
    second = pusht.run_fixed_trial(data, relative, chosen, 1, cache)
    assert second is not first and second.coordinates.name == "block_relative"
    changed = deepcopy(chosen)
    changed["parameters"]["bandwidth"] = .7
    assert pusht.run_fixed_trial(data, configuration, changed, 1, cache) is not first
    assert pusht.run_fixed_trial(example_data(), configuration, chosen, 1, cache) is not first
    assert len(cache) == 4


@pytest.mark.parametrize("representation", ["world", "block_relative"])
def test_describe_cli_override_wins_over_each_method_override(monkeypatch, capsys, representation):
    other = "world" if representation == "block_relative" else "block_relative"
    monkeypatch.setattr(runner, "load_method_configs", lambda *args: [
        method("bd_chi2", other), method("select_dpc", other)])
    monkeypatch.setattr(sys, "argv", ["pusht", "describe", "--representation", representation])
    runner.main("pusht")
    description = json.loads(capsys.readouterr().out)
    assert description["configuration"]["task"]["representation"] == representation
    assert description["representations"] == {"bd_chi2": representation, "select_dpc": representation}


@pytest.mark.parametrize("representation", ["world", "block_relative"])
def test_run_override_updates_protocol_and_both_methods_before_execution(monkeypatch, representation):
    other = "world" if representation == "block_relative" else "block_relative"
    monkeypatch.setattr(runner, "load_method_configs", lambda *args: [
        method("bd_chi2", other), method("select_dpc", other)])
    monkeypatch.setattr(runner, "experiment_starts", lambda *args: [(300005, None, 300)])
    monkeypatch.setattr(runner, "checked_input", lambda path: (runner.resolve(path), "fixture"))

    class ProtocolCaptured(Exception):
        pass

    captured = {}

    def capture(task, configuration, methods, starts, inputs):
        captured.update(configuration=configuration, methods=methods)
        raise ProtocolCaptured

    monkeypatch.setattr(runner, "protocol_manifest", capture)
    with pytest.raises(ProtocolCaptured):
        runner.run_task("pusht", representation=representation)
    assert captured["configuration"]["task"]["representation"] == representation
    assert all(item["representation"] == representation for item in captured["methods"])


def save_report_fixture(root, seed, result_representation=None, metadata_representation=None,
                        *, method_id="bd_chi2", configuration=None, method_representation=None):
    directory = root / "pusht" / method_id / f"seed_{seed}"
    directory.mkdir(parents=True)
    result = {"seed": seed, "steps": 1, "score": .5 / .95, "success": False,
              "termination_reason": "timeout"}
    metadata = {"config": configuration or {}, "method": {"id": method_id}}
    if method_representation is not None:
        metadata["method"]["representation"] = method_representation
    if result_representation is not None:
        result["representation"] = result_representation
    if metadata_representation is not None:
        metadata["representation"] = metadata_representation
    (directory / "result.json").write_text(json.dumps(result))
    (directory / "metadata.json").write_text(json.dumps(metadata))
    np.savez(directory / "trajectory.npz", observations=np.zeros((2, 6)),
             actions=np.zeros((1, 2)), coverage=np.array([.2, .5]))


def test_reports_label_relative_results_and_reject_mixed_method_rows(tmp_path):
    save_report_fixture(tmp_path, 300005, "block_relative", "block_relative")
    report = reporting.aggregate_results(tmp_path)
    assert report["rows"][0]["representation"] == "block_relative"
    assert not report["complete"]
    assert "| BD-χ² | block_relative |" in (tmp_path / "README.md").read_text()
    assert ",block_relative," in (tmp_path / "summary.csv").read_text()
    save_report_fixture(tmp_path, 300006, "world", "world")
    with pytest.raises(ValueError, match="different Push-T representations"):
        reporting.aggregate_results(tmp_path)


def test_report_rejects_inconsistent_representation_metadata(tmp_path):
    save_report_fixture(tmp_path, 300005, "block_relative", "world")
    with pytest.raises(ValueError, match="differs between result and metadata"):
        reporting.aggregate_results(tmp_path)


def test_old_results_without_representation_still_mean_world(tmp_path):
    save_report_fixture(tmp_path, 300005)
    report = reporting.aggregate_results(tmp_path)
    assert report["rows"][0]["representation"] == "world"


@pytest.mark.parametrize("method_id,expected", [("bd_chi2", "block_relative"), ("select_dpc", "world")])
def test_legacy_result_labels_use_saved_bd_only_setting(tmp_path, method_id, expected):
    save_report_fixture(tmp_path, 17, method_id=method_id,
                        configuration={"bd_coordinates": "block_relative"})
    report = reporting.aggregate_results(tmp_path)
    assert report["rows"][0]["representation"] == expected


@pytest.mark.parametrize("method_representation,expected", [(None, "world"), ("block_relative", "block_relative")])
def test_unlabelled_result_uses_saved_method_then_task_setting(tmp_path, method_representation, expected):
    save_report_fixture(tmp_path, 17,
                        configuration={"bd_coordinates": "block_relative", "task": {"representation": "world"}},
                        method_representation=method_representation)
    report = reporting.aggregate_results(tmp_path)
    assert report["rows"][0]["representation"] == expected
