"""Block-relative coordinates for Push-T observations and action chunks."""

import numpy as np

from src.data import TrajectoryData, TrajectoryDataSet, prepare_dataset
from src.optimization.hankel import HankelMatrixGenerator


def relative_observations(observations):
    """Rotate pusher-minus-block into the block frame; retain absolute block pose."""
    values = np.asarray(observations, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 6 or not np.isfinite(values).all():
        raise ValueError("Expected finite six-coordinate Push-T observations")
    result = values.copy()
    delta_x = values[..., 0] - values[..., 2]
    delta_y = values[..., 1] - values[..., 3]
    cosine, sine = values[..., 4], values[..., 5]
    result[..., 0] = cosine * delta_x + sine * delta_y
    result[..., 1] = -sine * delta_x + cosine * delta_y
    return result


def _check_actions_and_anchor(actions, anchor):
    actions = np.asarray(actions, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    if (actions.ndim < 2 or anchor.ndim < 1 or actions.shape[-1] != 2
            or anchor.shape[-1] != 6 or actions.ndim != anchor.ndim + 1
            or actions.shape[:-2] != anchor.shape[:-1]):
        raise ValueError("Expected (..., horizon, 2) actions and (..., 6) planning anchors")
    if not np.isfinite(actions).all() or not np.isfinite(anchor).all():
        raise ValueError("Actions and planning anchors must be finite")
    if not np.allclose(anchor[..., 4] ** 2 + anchor[..., 5] ** 2, 1, rtol=0, atol=1e-12):
        raise ValueError("Planning anchor must contain the cosine and sine of the block angle")
    return actions, anchor


def encode_actions(actions, anchor):
    """Express every target in one fixed planning-time block frame."""
    actions, anchor = _check_actions_and_anchor(actions, anchor)
    displacement = actions - anchor[..., None, 2:4]
    cosine, sine = anchor[..., None, 4], anchor[..., None, 5]
    return np.stack((cosine * displacement[..., 0] + sine * displacement[..., 1],
                     -sine * displacement[..., 0] + cosine * displacement[..., 1]), axis=-1)


def decode_actions(actions, anchor):
    """Decode a complete chunk once; its world targets stay fixed during execution."""
    actions, anchor = _check_actions_and_anchor(actions, anchor)
    cosine, sine = anchor[..., None, 4], anchor[..., None, 5]
    displacement = np.stack((cosine * actions[..., 0] - sine * actions[..., 1],
                             sine * actions[..., 0] + cosine * actions[..., 1]), axis=-1)
    return displacement + anchor[..., None, 2:4]


def relative_dictionary(data, past, future):
    """Build observation histories and future targets without future pose leakage.

    Each history observation uses its own block pose. All future action targets
    in a window use the last history observation's block pose. Past inputs stay
    in world coordinates and are excluded by the configured Push-T policy.
    """
    episodes = [TrajectoryData(
        relative_observations(episode.state_trajectory), episode.input_trajectory.copy(),
        None if episode.terminal_observation is None else relative_observations(episode.terminal_observation),
    ) for episode in data.dataset]
    transformed = TrajectoryDataSet(episodes, data.dataset_name + "_block_relative", data.timing,
                                    None if data.episode_ids is None else list(data.episode_ids))
    inputs, outputs = HankelMatrixGenerator(past, future).generate_hankel_matrices(prepare_dataset(transformed))
    anchors = outputs[(past - 1) * 6:past * 6].T
    actions = inputs[past * 2:].T.reshape(-1, future, 2)
    inputs = inputs.copy()
    inputs[past * 2:] = encode_actions(actions, anchors).reshape(-1, future * 2).T
    return inputs, outputs
