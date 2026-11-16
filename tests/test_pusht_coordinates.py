"""Physical coordinate, window-alignment and optimization-equivalence checks."""

from copy import deepcopy

import numpy as np
import pytest

from src.data import TrajectoryData, TrajectoryDataSet, prepare_dataset
from src.optimization.hankel import HankelMatrixGenerator
from src.tasks.pusht_coordinates import PushTCoordinates


def moving_block_data(timing="post_action"):
    count = 8
    index = np.arange(count)
    observations = np.column_stack((130 + index, 230 - index,
                                    100 + 10 * index, 200 + 5 * index,
                                    np.ones(count), np.zeros(count)))
    actions = np.column_stack((150 + index, 260 - 2 * index))
    return TrajectoryDataSet([TrajectoryData(observations, actions)], "moving_frame", timing)


def test_relative_axes_have_explicit_physical_meaning():
    coordinates = PushTCoordinates("block_relative")
    anchor = np.array([110., 220., 100., 200., 0., 1.])
    np.testing.assert_array_equal(coordinates.observations(anchor), [20., -10., 100., 200., 0., 1.])
    world = np.array([[100., 210.], [80., 200.]])
    expected = np.array([[10., 0.], [0., 20.]])
    np.testing.assert_array_equal(coordinates.encode_actions(world, anchor), expected)
    np.testing.assert_array_equal(coordinates.decode_actions(expected, anchor), world)


@pytest.mark.parametrize("representation", ["world", "block_relative"])
@pytest.mark.parametrize("shape", [(2,), (4, 2), (3, 4, 2)])
def test_actions_round_trip_without_changing_inputs(representation, shape):
    rng = np.random.default_rng(48)
    actions = rng.uniform(-50, 550, shape)
    angles = np.array([.4, -.9, 2.2]) if len(shape) == 3 else np.array(.4)
    if len(shape) == 3:
        anchors = np.column_stack((np.ones((3, 2)), rng.uniform(50, 450, (3, 2)),
                                   np.cos(angles), np.sin(angles)))
    else:
        anchors = np.r_[1., 2., 120., 230., np.cos(angles), np.sin(angles)]
    old_actions, old_anchors = actions.copy(), anchors.copy()
    coordinates = PushTCoordinates(representation)
    restored = coordinates.decode_actions(coordinates.encode_actions(actions, anchors), anchors)
    np.testing.assert_allclose(restored, actions, rtol=0, atol=2e-13)
    np.testing.assert_array_equal(actions, old_actions)
    np.testing.assert_array_equal(anchors, old_anchors)
    if representation == "world":
        np.testing.assert_array_equal(restored, actions)


@pytest.mark.parametrize("timing", ["pre_action", "post_action"])
def test_world_windows_preserve_original_alignment_exactly(timing):
    data = moving_block_data(timing)
    expected = HankelMatrixGenerator(2, 3).generate_hankel_matrices(prepare_dataset(data))
    actual = PushTCoordinates("world").dictionary(data, 2, 3)
    for before, after in zip(expected, actual):
        np.testing.assert_array_equal(after, before)


@pytest.mark.parametrize("timing", ["pre_action", "post_action"])
def test_relative_window_uses_last_history_pose_for_past_and_future_inputs(timing):
    data = moving_block_data(timing)
    raw = deepcopy(data)
    coordinates = PushTCoordinates("block_relative")
    inputs, outputs = coordinates.dictionary(data, 2, 3)
    aligned = prepare_dataset(data).dataset[0]
    for start in range(inputs.shape[1]):
        anchor = aligned.state_trajectory[start + 1]
        # All fixture orientations are zero, so encoding is only translation.
        expected = aligned.input_trajectory[start:start + 5] - anchor[2:4]
        np.testing.assert_array_equal(inputs[:, start].reshape(5, 2), expected)
        np.testing.assert_array_equal(outputs[6:12, start][2:6], anchor[2:6])
    np.testing.assert_array_equal(data.dataset[0].state_trajectory, raw.dataset[0].state_trajectory)
    np.testing.assert_array_equal(data.dataset[0].input_trajectory, raw.dataset[0].input_trajectory)


def test_overlapping_windows_have_distinct_frames_and_no_future_pose_leakage():
    coordinates = PushTCoordinates("block_relative")
    data = moving_block_data()
    inputs, outputs = coordinates.dictionary(data, 2, 3)
    # The same world action u[2] occurs at two row/column positions, with
    # different last-history block positions. A global trajectory conversion
    # would incorrectly assign the same relative target to both occurrences.
    first, second = inputs[4:6, 0], inputs[2:4, 1]
    np.testing.assert_array_equal(first - second, [10., 5.])
    changed = deepcopy(data)
    changed.dataset[0].state_trajectory[2:, 2:4] += [73., -19.]
    changed.dataset[0].state_trajectory[2:, 4:6] = [0., 1.]
    new_inputs, new_outputs = coordinates.dictionary(changed, 2, 3)
    np.testing.assert_array_equal(new_inputs[:, 0], inputs[:, 0])
    np.testing.assert_array_equal(new_outputs[:12, 0], outputs[:12, 0])
    assert not np.array_equal(new_outputs[12:, 0], outputs[12:, 0])


def test_bd_compatibility_keeps_unused_past_inputs_in_world_coordinates():
    data = moving_block_data()
    world, _ = PushTCoordinates("world").dictionary(data, 2, 3)
    full, _ = PushTCoordinates("block_relative").dictionary(data, 2, 3)
    bd, _ = PushTCoordinates("block_relative").dictionary(data, 2, 3, transform_past_inputs=False)
    np.testing.assert_array_equal(bd[:4], world[:4])
    np.testing.assert_array_equal(bd[4:], full[4:])


@pytest.mark.parametrize("representation", ["world", "block_relative"])
def test_standardized_map_preserves_workspace_and_physical_input_cost(representation):
    coordinates = PushTCoordinates(representation)
    angle = .73
    anchor = np.array([130., 180., 190., 270., np.cos(angle), np.sin(angle)])
    mean, scale = np.array([12., -31.]), np.array([28., 53.])
    world_mean, world_scale = np.array([201., 309.]), np.array([90., 60.])
    weight = .07
    offset, matrix = coordinates.standardized_action_map(anchor, mean, scale)
    bounds, bound_values = coordinates.input_constraints(anchor, mean, scale)
    cost_matrix, reference = coordinates.input_cost(anchor, mean, scale, world_mean, world_scale, weight)
    world_targets = np.array([[0., 0.], [512., 512.], [60., 100.],
                              [-5., 100.], [100., 518.], [260., 259.]])
    native = coordinates.encode_actions(world_targets, anchor)
    standardized = (native - mean) / scale
    np.testing.assert_allclose(standardized @ matrix.T + offset, world_targets, rtol=0, atol=2e-13)
    expected_feasible = np.all((world_targets >= 0) & (world_targets <= 512), axis=1)
    actual_feasible = np.all(standardized @ bounds.T <= bound_values + 1e-10, axis=1)
    np.testing.assert_array_equal(actual_feasible, expected_feasible)
    expected_cost = weight * np.sum(((world_targets - world_mean) / world_scale) ** 2, axis=1)
    error = standardized - reference
    actual_cost = np.einsum("ni,ij,nj->n", error, cost_matrix, error)
    np.testing.assert_allclose(actual_cost, expected_cost, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(cost_matrix, cost_matrix.T, atol=1e-15)
    assert np.linalg.eigvalsh(cost_matrix).min() > 0


def test_invalid_representation_is_not_silently_treated_as_world():
    with pytest.raises(ValueError):
        PushTCoordinates("relative_to_current_step")
