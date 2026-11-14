"""Timing replays preserve feedback and separate invalid results from measurements."""
import numpy as np
import pytest

from scripts.timing import replay_sequence, restore_nonfinite_values


class StatefulPolicy:
    def __init__(self):
        self.previous = np.zeros(2)
        self.observations = []
        self.feedback = []

    def act(self, observation, reference):
        self.observations.append(observation.copy())
        assert len(observation) == 10
        np.testing.assert_array_equal(observation[-2:], reference[4:6])
        return np.array([observation[0] + self.previous[0], observation[-2]])

    def record(self, value):
        self.previous = value.copy()
        self.feedback.append(value.copy())


def saved_queries():
    observations = np.zeros((5, 8))
    observations[:, 0] = np.arange(5)
    references = np.zeros((4, 8))
    references[:, 4] = np.arange(10, 14)
    references[:, 5] = np.arange(20, 24)
    feedback = np.array([[1., 2.], [-2., 3.], [4., 5.], [6., 7.]])
    return {"observations": observations, "references": references,
            "actions": feedback.copy(), "feedback_actions": feedback,
            "policy_decision_mask": np.array([True, False, True, True]),
            "proposed_actions": np.array([[0., 10.], [np.nan, np.nan], [0., 12.], [7., 13.]])}


def replay(policy, arrays, **options):
    return replay_sequence(policy, arrays, (np.full(2, -1), np.ones(2)), True,
                           lambda controller, action: controller.record(action), **options)


def test_override_feedback_and_reference_augmentation():
    policy = StatefulPolicy()
    arrays = saved_queries()
    result, steps = replay(policy, arrays)
    assert steps == 4
    np.testing.assert_array_equal(result["step_indices"], [0, 2, 3])
    np.testing.assert_array_equal(result["absolute_action_gaps"], 0)
    np.testing.assert_array_equal(policy.feedback, arrays["feedback_actions"])
    assert len(policy.observations) == 3
    assert np.all(result["cpu_ns"] > 0)


def test_partial_replay_keeps_feedback_without_extra_decisions():
    policy = StatefulPolicy()
    arrays = saved_queries()
    result, steps = replay(policy, arrays, max_decisions=1)
    assert len(result["cpu_ns"]) == 1
    assert steps == 2
    np.testing.assert_array_equal(policy.feedback, arrays["feedback_actions"][:2])


def test_action_mismatch_is_not_hidden():
    arrays = saved_queries()
    arrays["proposed_actions"][2, 0] = 9
    result, _ = replay(StatefulPolicy(), arrays)
    assert result["absolute_action_gaps"][1] == 9


def test_nonfinite_tags_preserve_plain_null_and_input():
    encoded = {"parameters": {"bandwidth": None, "plain_null": None},
               "nonfinite_values": {"/parameters/bandwidth": "+Infinity"}}
    decoded = restore_nonfinite_values(encoded)
    assert np.isposinf(decoded["parameters"]["bandwidth"])
    assert decoded["parameters"]["plain_null"] is None
    assert encoded["parameters"]["bandwidth"] is None


def test_invalid_array_alignment_is_rejected():
    arrays = saved_queries()
    arrays["references"] = arrays["references"][:-1]
    with pytest.raises(ValueError):
        replay(StatefulPolicy(), arrays)
