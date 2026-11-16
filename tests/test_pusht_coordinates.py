"""Physical coordinate checks independent of archived experiment results."""

from copy import deepcopy

import numpy as np
import pytest

from experiments.pusht import RelativeBDChi2, build_policy
from src.data import DeePCDims, TrajectoryData, TrajectoryDataSet
from src.policies.bd_chi2 import BDChi2Policy
from src.policies.pusht_coordinates import (
    decode_actions, encode_actions, relative_dictionary, relative_observations,
)
from src.tasks.pusht import state_features


def test_relative_observation_rotates_displacement_and_retains_absolute_block_pose():
    observations = np.array([[13, 24, 10, 20, 1, 0], [13, 24, 10, 20, 0, 1]], dtype=float)
    transformed = relative_observations(observations)
    np.testing.assert_array_equal(transformed[:, :2], [[3, 4], [4, -3]])
    np.testing.assert_array_equal(transformed[:, 2:], observations[:, 2:])
    np.testing.assert_array_equal(observations[:, :2], [[13, 24], [13, 24]])


def test_action_encoding_uses_one_anchor_for_entire_chunk():
    anchor = np.array([3, 4, 10, 20, 0, 1], dtype=float)
    actions = np.array([[13, 24], [15, 26], [17, 28]], dtype=float)
    encoded = encode_actions(actions, anchor)
    np.testing.assert_array_equal(encoded, [[4, -3], [6, -5], [8, -7]])
    np.testing.assert_array_equal(decode_actions(encoded, anchor), actions)
    anchors = state_features(np.array([[3, 4, 10, 20, .3], [5, 6, -20, 50, 1.2]]))
    batch = np.stack((actions, actions + 20))
    np.testing.assert_allclose(decode_actions(encode_actions(batch, anchors), anchors), batch,
                               rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="cosine and sine"):
        encode_actions(actions, [0, 0, 0, 0, .5, .5])


def test_window_alignment_anchors_targets_at_last_history_not_future_poses():
    raw = np.array([[20 + i, 30 + 2 * i, 50 + 5 * i, 70 + 3 * i, .2 * i] for i in range(7)])
    observations = state_features(raw)
    actions = np.array([[100 + 4 * i, 150 + 7 * i] for i in range(7)])
    data = TrajectoryDataSet([TrajectoryData(observations, actions)], "fixture", "pre_action")
    original = deepcopy(data)
    inputs, outputs = relative_dictionary(data, past=2, future=2)
    for window in range(3):
        anchor = observations[window + 2]
        np.testing.assert_array_equal(outputs[:12, window],
                                      relative_observations(observations[window + 1:window + 3]).ravel())
        np.testing.assert_array_equal(inputs[:4, window], actions[window:window + 2].ravel())
        np.testing.assert_array_equal(inputs[4:, window],
                                      encode_actions(actions[window + 2:window + 4], anchor).ravel())
    np.testing.assert_array_equal(data.dataset[0].state_trajectory, original.dataset[0].state_trajectory)
    np.testing.assert_array_equal(data.dataset[0].input_trajectory, original.dataset[0].input_trajectory)


def test_chunk_targets_remain_fixed_while_block_moves(monkeypatch):
    native_plan = np.array([[1., 2.], [3., 4.], [5., 6.]])
    monkeypatch.setattr(BDChi2Policy, "plan", lambda self, history: native_plan.copy())
    dims = DeePCDims(2, 3, 6, 2)
    policy = RelativeBDChi2(np.zeros((10, 5)), np.zeros((30, 5)), dims,
                           action_horizon=3, execute_steps=3, include_past_inputs=False)
    policy.reset(7)
    first = np.array([3, 4, 10, 20, 0, 1], dtype=float)
    second = np.array([8, 21, 100, 120, 1, 0], dtype=float)
    third = np.array([6, 23, 200, 220, -1, 0], dtype=float)
    actual = [policy.act(observation) for observation in (first, second, third)]
    np.testing.assert_array_equal(actual, [[8, 21], [6, 23], [4, 25]])
    assert policy.query_index == 0
    np.testing.assert_array_equal(policy.latest_query["action_anchor_observation"], first)
    np.testing.assert_array_equal(policy.latest_query["coordinate_actions"], native_plan)
    np.testing.assert_array_equal(policy.latest_query["planned_actions"], actual)
    fourth = np.array([4, 25, 40, 50, 1, 0], dtype=float)
    np.testing.assert_array_equal(policy.act(fourth), [41, 52])
    assert policy.query_index == 1


def test_relative_builder_rejects_unsupported_past_input_conditioning():
    method = {"id": "bd_chi2", "prediction_horizon": 8, "parameters": {"include_past_inputs": True}}
    configuration = {"history": 2, "execute_steps": 8, "bd_coordinates": "block_relative"}
    with pytest.raises(ValueError, match="include_past_inputs=false"):
        build_policy(None, configuration, method, None)
