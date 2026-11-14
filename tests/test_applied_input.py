"""Executed-input feedback must reach the next solve and online dictionary."""

from copy import deepcopy

import cvxpy as cp
import numpy as np
import pytest

from src.data import (DeePCConstraints, DeePCControllerArgs, DeePCCost, DeePCDims,
                      TrajectoryData, TrajectoryDataSet)
from src.evaluation import record_applied_action
from src.optimization.deepc_qp import DeePCQP
from src.optimization.hankel import HankelMatrixGenerator
from src.policies.deepc import DeePC
from src.policies.select_dpc import SelectDPC
from src.policies.windowed_deepc import WindowedDeePC


KINDS = ["deepc", "select_dpc", "windowed_deepc"]


def make_policy(kind):
    """A small, persistently excited two-input linear system with real QP solves."""
    inputs = np.random.default_rng(17).normal(size=(36, 2))
    outputs, state = [], 0.0
    for action in inputs:
        state = 0.8 * state + action[0] - 0.5 * action[1]
        outputs.append([state])
    data = TrajectoryDataSet([TrajectoryData(outputs, inputs)], "feedback", "post_action")
    dims = DeePCDims(T_past=2, T_fut=2, p=1, m=2)
    costs = DeePCCost(np.eye(1), 0.1 * np.eye(2), 20.0, 0.01, 0.0)
    constraints = DeePCConstraints(None, None, None, None)
    args = DeePCControllerArgs(data, dims, len(inputs), costs, constraints, np.zeros(2), False)
    H_u, H_y = HankelMatrixGenerator(dims.T_past, dims.T_fut).generate_hankel_matrices(data)
    options = {"eps_abs": 1e-8, "eps_rel": 1e-8, "max_iter": 20000}
    if kind == "deepc":
        return DeePC(H_u, H_y, dims, costs, constraints, np.zeros(2), solver_options=options)
    if kind == "select_dpc":
        return SelectDPC(args, num_hankel_cols=-1, n_iter=2,
                         hankel_matrices=(H_u, H_y), solver_options=options)
    return WindowedDeePC(H_u, H_y, args, solver_options=options)


def solver(policy):
    return policy if isinstance(policy, DeePC) else policy._deepc


def outer_history(policy):
    if isinstance(policy, DeePC):
        return policy._u_past.value
    if isinstance(policy, WindowedDeePC):
        return policy._u_init
    return policy._u_past


@pytest.mark.parametrize("kind", KINDS)
def test_applied_input_reaches_next_solve_without_changing_accepted_plan(kind):
    policy = make_policy(kind)
    proposed = policy.act(np.array([0.0]), np.array([3.0]))
    accepted = solver(policy)._plan
    assert accepted is not None
    frozen_plan = deepcopy(accepted)
    frozen_predictions = policy.get_planned_input_trajectory().copy()
    applied = np.array([0.07, -0.13])
    assert not np.allclose(proposed, applied)

    # Use the same public callback dispatch as the benchmark rollouts.
    record_applied_action(policy, applied)
    np.testing.assert_array_equal(outer_history(policy), [0.0, 0.0, 0.07, -0.13])
    np.testing.assert_array_equal(solver(policy)._u_past.value, outer_history(policy))
    np.testing.assert_array_equal(policy.diagnostics()["executed_action"], applied)
    assert policy.diagnostics()["action_source"] == "optimized"
    np.testing.assert_array_equal(policy.get_planned_input_trajectory(), frozen_predictions)
    assert solver(policy)._plan is accepted
    for key, value in frozen_plan.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(accepted[key], value)
            assert not accepted[key].flags.writeable
        else:
            assert accepted[key] == value

    # The next observed output was produced by the executed input, not the proposal.
    next_output = applied[0] - 0.5 * applied[1]
    policy.act(np.array([next_output]), np.array([3.0]))
    assert solver(policy)._plan is not accepted
    np.testing.assert_array_equal(solver(policy)._plan["u_past"], [0.0, 0.0, 0.07, -0.13])
    np.testing.assert_array_equal(solver(policy)._plan["y_past"], [0.0, next_output])


def test_windowed_dictionary_pairs_each_executed_input_with_its_resulting_output():
    policy = make_policy("windowed_deepc")
    applied_inputs = np.array([[0.1, -0.2], [-0.05, 0.3], [0.2, 0.1], [-0.4, -0.2]])
    outputs, state = [], 0.0
    initial_dictionary = policy._H_u.copy()
    for action in applied_inputs:
        policy.act(np.array([state]), np.array([3.0]))
        record_applied_action(policy, action)
        np.testing.assert_array_equal(policy._pending_action, action)
        state = 0.8 * state + action[0] - 0.5 * action[1]
        outputs.append(state)
    # A complete length-four column exists only after the fourth resulting output.
    np.testing.assert_array_equal(policy._H_u, initial_dictionary)
    policy.act(np.array([state]), np.array([3.0]))
    np.testing.assert_array_equal(policy._H_u[:, -1], applied_inputs.reshape(-1))
    np.testing.assert_array_equal(policy._H_y[:, -1], outputs)
    np.testing.assert_array_equal(policy._H_u[:, :-1], initial_dictionary[:, 1:])


@pytest.mark.parametrize("kind", KINDS)
def test_applied_feedback_also_corrects_fallback_and_zero_actions(kind, monkeypatch):
    policy = make_policy(kind)
    first_action = policy.act(np.array([0.0]), np.array([3.0]))
    record_applied_action(policy, first_action)
    accepted = solver(policy)._plan
    original_plan_inputs = accepted["u"].copy()
    monkeypatch.setattr(DeePCQP, "solve", lambda self: cp.INFEASIBLE)

    for source, applied in [("fallback", [0.03, -0.04]), ("zero", [0.01, 0.02])]:
        with pytest.warns(RuntimeWarning):
            policy.act(np.array([0.1]), np.array([3.0]))
        assert policy.diagnostics()["action_source"] == source
        record_applied_action(policy, applied)
        np.testing.assert_array_equal(outer_history(policy)[-2:], applied)
        np.testing.assert_array_equal(solver(policy)._u_past.value, outer_history(policy))
        np.testing.assert_array_equal(policy.diagnostics()["executed_action"], applied)
        assert solver(policy)._plan is accepted
        np.testing.assert_array_equal(accepted["u"], original_plan_inputs)


@pytest.mark.parametrize("kind", KINDS)
def test_feedback_without_a_new_policy_action_does_not_invent_a_history_pair(kind):
    policy = make_policy(kind)
    # Rocket may begin in contact and use its touchdown override without calling act.
    assert policy.record_applied_action([0.2, -0.1]) is False
    assert outer_history(policy) is None
    if isinstance(policy, WindowedDeePC):
        assert policy._pending_action is None

    policy.act(np.array([0.0]), np.array([3.0]))
    assert policy.record_applied_action([0.07, -0.13]) is True
    before = outer_history(policy).copy()
    # Sticky touchdown control skips all subsequent policy calls. Its callbacks
    # must not overwrite the last action already paired with a policy observation.
    assert policy.record_applied_action([0.4, 0.5]) is False
    np.testing.assert_array_equal(outer_history(policy), before)
    np.testing.assert_array_equal(policy.diagnostics()["executed_action"], [0.07, -0.13])
    if isinstance(policy, WindowedDeePC):
        np.testing.assert_array_equal(policy._pending_action, [0.07, -0.13])

    policy.reset()
    assert policy.record_applied_action([0.2, -0.1]) is False
    assert outer_history(policy) is None


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("invalid", [[1.0], [1.0, 2.0, 3.0], [np.nan, 0.0], [0.0, np.inf]])
def test_invalid_applied_input_does_not_consume_pending_feedback(kind, invalid):
    policy = make_policy(kind)
    with pytest.raises(ValueError, match="finite vector"):
        policy.record_applied_action(invalid)
    policy.act(np.array([0.0]), np.array([3.0]))
    before = outer_history(policy).copy()
    with pytest.raises(ValueError, match="finite vector"):
        policy.record_applied_action(invalid)
    np.testing.assert_array_equal(outer_history(policy), before)
    assert policy.record_applied_action([0.07, -0.13]) is True


@pytest.mark.parametrize("kind", KINDS)
def test_unchanged_applied_input_preserves_following_actions(kind):
    original, with_feedback = make_policy(kind), make_policy(kind)
    for observation in [0.0, 0.2, -0.1, 0.3, 0.1]:
        expected = original.act(np.array([observation]), np.array([3.0]))
        actual = with_feedback.act(np.array([observation]), np.array([3.0]))
        np.testing.assert_array_equal(actual, expected)
        record_applied_action(with_feedback, actual)
        np.testing.assert_array_equal(outer_history(with_feedback), outer_history(original))
