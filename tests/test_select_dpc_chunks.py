"""Fixed-input migration and history/failure checks for the optional Push-T adapter."""

import cvxpy as cp
import numpy as np
import pytest

from src.data import TrajectoryData, TrajectoryDataSet, prepare_dataset
from src.optimization.deepc_qp import DeePCQP
from src.policies.select_dpc_chunks import SelectDPCChunks


REFERENCE = np.r_[0., 0., 256., 256., np.cos(np.pi / 4), np.sin(np.pi / 4)]
PARAMETERS = dict(horizon=3, g_l1=.1, columns=32, iterations=2,
                  output_weight=10., input_weight=.01, slack=100., g_projection=.1)


def example_data(timing="post_action"):
    """A small, fixed synthetic dictionary; these are not simulator rollouts."""
    rng = np.random.default_rng(17)
    episodes = []
    for _ in range(2):
        actions = rng.uniform(220.0, 300.0, (24, 2))
        positions = rng.uniform(220.0, 300.0, (24, 4))
        angles = rng.uniform(0.3, 1.2, 24)
        states = np.column_stack((positions, np.cos(angles), np.sin(angles)))
        episodes.append(TrajectoryData(states, actions))
    return TrajectoryDataSet(episodes, "synthetic_push_t", timing)


def policy(data=None, **overrides):
    return SelectDPCChunks(example_data() if data is None else data, reference=REFERENCE,
                           past=2, execute_steps=2, **(PARAMETERS | overrides))


def test_fixed_input_actions_reproduce_preliminary_adapter():
    # Frozen with NumPy 1.26.4 and 2.1.2, OSQP 0.6.7.post3, from the preliminary
    # evaluate_pusht.py SelectDPCChunks class (no simulator or benchmark run).
    # Source SHA256: efbb1bcba728068d43ece71e1d1a5bd658a32a8d10af38b1956cce397970338c.
    # Five observations, three queries, and deliberately clipped action feedback.
    expected = np.array([
        [250.83251733977716, 238.55735204329397],
        [227.45038896686117, 314.15326496613943],
        [208.75562982790024, 341.98969458839247],
        [286.90187106288204, 178.7738117991017],
        [246.07065647076058, 217.60798030719218],
    ])
    data = example_data()
    controller = policy(data)
    actual = []
    for step, observation in enumerate(data.dataset[0].state_trajectory[:5]):
        actual.append(controller.act(observation))
        controller.record_applied_action(np.clip(actual[-1], 235., 270.))
        assert controller.diagnostics()["new_query"] == (step % 2 == 0)
        assert controller.diagnostics()["query_index"] == step // 2
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
    assert controller.diagnostics()["queries"] == 3
    assert controller.diagnostics()["solver_failures"] == 0
    assert controller.latest_query["planned_actions"].shape == (3, 2)
    assert controller.latest_query["predicted_observations"].shape == (3, 6)


def test_query_history_contains_each_measured_output_and_applied_input():
    data = example_data()
    controller = policy(data)
    observations = data.dataset[0].state_trajectory[:3]
    controller.act(observations[0])
    first_query = controller.latest_query
    np.testing.assert_allclose(first_query["history_observations"], [observations[0]] * 2)
    np.testing.assert_allclose(first_query["history_actions"], [observations[0][:2]] * 2)
    controller.record_applied_action([240., 250.])
    controller.act(observations[1])
    controller.record_applied_action([260., 270.])
    controller.act(observations[2])
    latest = controller.latest_query
    np.testing.assert_allclose(latest["history_observations"], observations[1:])
    np.testing.assert_allclose(latest["history_actions"], [[240., 250.], [260., 270.]])
    np.testing.assert_array_equal(first_query["reference"], REFERENCE)
    assert latest["step"] == 2
    assert controller.diagnostics()["new_query"]
    # Saved diagnostics cannot mutate the live controller.
    latest["planned_actions"][:] = np.nan
    assert np.isfinite(controller.latest_query["planned_actions"]).all()


def test_failed_query_holds_current_position_and_does_not_reuse_old_query(monkeypatch):
    controller = policy()
    observations = example_data().dataset[0].state_trajectory[:4]
    for observation in observations[:2]:
        controller.record_applied_action(controller.act(observation))
    assert controller.latest_query["plan_valid"]
    monkeypatch.setattr(DeePCQP, "solve", lambda self: cp.INFEASIBLE)
    with pytest.warns(RuntimeWarning):
        failed_action = controller.act(observations[2])
    np.testing.assert_allclose(failed_action, observations[2][:2])
    controller.record_applied_action([241., 251.])
    # A cached hold chunk retains the query-time position for both actions.
    np.testing.assert_allclose(controller.act(observations[3]), observations[2][:2])
    failed_query = controller.latest_query
    assert not failed_query["plan_valid"]
    assert failed_query["planned_actions"] is None
    assert failed_query["predicted_observations"] is None
    assert failed_query["selected_indices"] is None
    assert [item["status"] for item in failed_query["rounds"]] == [cp.INFEASIBLE] * 2
    assert controller.diagnostics()["action_source"] == "hold_position"
    assert controller.diagnostics()["solver_failures"] == 1
    assert controller.diagnostics()["failed_solves"] == 2


def test_late_failed_refinement_retains_accepted_plan_within_same_query(monkeypatch):
    original = DeePCQP.solve
    calls = []

    def fail_second(solver):
        calls.append(solver)
        return original(solver) if len(calls) == 1 else cp.INFEASIBLE

    monkeypatch.setattr(DeePCQP, "solve", fail_second)
    controller = policy()
    with pytest.warns(RuntimeWarning):
        action = controller.act(example_data().dataset[0].state_trajectory[0])
    query = controller.latest_query
    assert [item["accepted"] for item in query["rounds"]] == [True, False]
    assert query["plan_valid"]
    np.testing.assert_allclose(action, query["planned_actions"][0])
    np.testing.assert_array_equal(query["selected_indices"], query["rounds"][0]["selected_indices"])
    assert controller.diagnostics()["solver_failures"] == 0
    assert controller.diagnostics()["failed_solves"] == 1


def test_pre_action_alignment_matches_explicit_conversion():
    data = example_data("pre_action")
    implicit, explicit = policy(data), policy(prepare_dataset(data))
    observation = data.dataset[0].state_trajectory[0]
    np.testing.assert_allclose(implicit.act(observation), explicit.act(observation), rtol=0, atol=1e-10)
    # Each 24-sample episode loses its unmatched final action; no cross-boundary windows.
    assert implicit.diagnostics()["dictionary_columns"] == 2 * (23 - 2 - 3 + 1)


def test_feedback_is_consumed_once_and_reset_discards_histories():
    controller = policy()
    assert controller.record_applied_action([240., 250.]) is False
    observation = example_data().dataset[0].state_trajectory[0]
    controller.act(observation)
    assert controller.record_applied_action([240., 250.]) is True
    assert controller.record_applied_action([260., 270.]) is False
    np.testing.assert_array_equal(controller.diagnostics()["applied_action"], [240., 250.])
    controller.reset(300000)
    assert controller.latest_query is None
    assert controller.diagnostics()["queries"] == 0
    assert controller.record_applied_action([260., 270.]) is False


@pytest.mark.parametrize("invalid", [[1.], [1., 2., 3.], [np.inf, 0.], [0., np.nan]])
def test_invalid_feedback_does_not_consume_pending_action(invalid):
    controller = policy()
    controller.act(example_data().dataset[0].state_trajectory[0])
    with pytest.raises(ValueError, match="finite vector"):
        controller.record_applied_action(invalid)
    assert controller.record_applied_action([240., 250.]) is True


@pytest.mark.parametrize("overrides", [dict(horizon=1), dict(columns=0), dict(iterations=0),
                                      dict(g_l1=-1), dict(output_weight=np.nan)])
def test_reject_invalid_protocol(overrides):
    with pytest.raises(ValueError):
        policy(**overrides)
