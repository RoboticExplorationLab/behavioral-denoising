"""Shared Push-T observations, planning frames and demonstration windows.

The simulator always uses world coordinates. A relative action chunk is tied to
the block pose when planning starts, so decoding it never follows later block
motion. Normalization remains the controller's responsibility.
"""

import numpy as np

from ..data import TrajectoryData, TrajectoryDataSet, prepare_dataset
from ..optimization.hankel import HankelMatrixGenerator


REPRESENTATIONS = ("world", "block_relative")


def _observations(values):
    values = np.asarray(values, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 6 or not np.isfinite(values).all():
        raise ValueError("Expected finite six-coordinate Push-T observations")
    return values


def _pair(value, name, *, positive=False, scalar=False):
    values = np.asarray(value, dtype=float)
    if scalar and values.ndim == 0:
        values = np.repeat(values, 2)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain two finite coordinates")
    if positive and np.any(values <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return values


def _action_chunk(actions, anchor):
    actions = np.asarray(actions, dtype=float)
    if actions.ndim < 1 or actions.shape[-1] != 2 or not np.isfinite(actions).all():
        raise ValueError("Expected finite two-coordinate Push-T action targets")
    anchor = _observations(anchor)
    single_action = actions.ndim == anchor.ndim
    if single_action:
        actions = actions[..., None, :]
    if actions.ndim != anchor.ndim + 1 or actions.shape[:-2] != anchor.shape[:-1]:
        raise ValueError("Action chunks and planning anchors must have matching batch dimensions")
    if not np.allclose(anchor[..., 4] ** 2 + anchor[..., 5] ** 2, 1, rtol=0, atol=1e-12):
        raise ValueError("Planning anchor must contain the cosine and sine of the block angle")
    return actions, anchor, single_action


class PushTCoordinates:
    """Choose one coordinate convention for any Push-T controller adapter.

    ``world`` preserves absolute pusher positions and action targets.
    ``block_relative`` expresses pusher positions in their observed block frame,
    retaining absolute block position/orientation. Every action in a planning
    window uses the last history observation's block frame, including past
    inputs when the controller conditions on them.
    """

    def __init__(self, representation="world"):
        if representation not in REPRESENTATIONS:
            raise ValueError(f"Push-T representation must be one of {REPRESENTATIONS}")
        self.name = representation

    def observations(self, observations):
        values = _observations(observations)
        result = values.copy()
        if self.name == "block_relative":
            displacement_x = values[..., 0] - values[..., 2]
            displacement_y = values[..., 1] - values[..., 3]
            cosine, sine = values[..., 4], values[..., 5]
            result[..., 0] = cosine * displacement_x + sine * displacement_y
            result[..., 1] = -sine * displacement_x + cosine * displacement_y
        return result

    def encode_actions(self, actions, anchor=None):
        """Express world targets in a fixed planning frame, without scaling."""
        if self.name == "world":
            values = np.asarray(actions, dtype=float)
            if values.ndim < 1 or values.shape[-1] != 2 or not np.isfinite(values).all():
                raise ValueError("Expected finite two-coordinate Push-T action targets")
            return values.copy()
        actions, anchor, single_action = _action_chunk(actions, anchor)
        displacement = actions - anchor[..., None, 2:4]
        cosine, sine = anchor[..., None, 4], anchor[..., None, 5]
        encoded = np.stack((cosine * displacement[..., 0] + sine * displacement[..., 1],
                            -sine * displacement[..., 0] + cosine * displacement[..., 1]), axis=-1)
        return encoded[..., 0, :] if single_action else encoded

    def decode_actions(self, actions, anchor=None):
        """Decode a whole chunk once; keep its world targets fixed in execution."""
        if self.name == "world":
            return self.encode_actions(actions)
        actions, anchor, single_action = _action_chunk(actions, anchor)
        cosine, sine = anchor[..., None, 4], anchor[..., None, 5]
        displacement = np.stack((cosine * actions[..., 0] - sine * actions[..., 1],
                                 sine * actions[..., 0] + cosine * actions[..., 1]), axis=-1)
        decoded = displacement + anchor[..., None, 2:4]
        return decoded[..., 0, :] if single_action else decoded

    def dictionary(self, data, past, future, transform_past_inputs=True):
        """Build raw windows without crossing episodes or using future anchors.

        Relative histories use each observation's own block frame. Inputs use
        the planning frame shared by that window. ``transform_past_inputs=False``
        preserves legacy BD dictionaries, whose unused past inputs stayed in
        world coordinates; controllers using past inputs must leave it True.
        """
        if (isinstance(past, bool) or isinstance(future, bool)
                or not isinstance(past, (int, np.integer))
                or not isinstance(future, (int, np.integer)) or past < 1 or future < 1):
            raise ValueError("History and prediction lengths must be positive integers")
        if not isinstance(transform_past_inputs, (bool, np.bool_)):
            raise ValueError("transform_past_inputs must be a boolean")
        if not isinstance(data, TrajectoryDataSet):
            raise TypeError("Push-T dictionaries require a TrajectoryDataSet")
        if any(episode.state_trajectory.shape[1] != 6 or episode.input_trajectory.shape[1] != 2
               for episode in data.dataset):
            raise ValueError("Push-T demonstrations require six observations and two actions")
        if self.name == "world":
            return HankelMatrixGenerator(past, future).generate_hankel_matrices(prepare_dataset(data))
        episodes = [TrajectoryData(
            self.observations(episode.state_trajectory), episode.input_trajectory.copy(),
            None if episode.terminal_observation is None else self.observations(episode.terminal_observation),
        ) for episode in data.dataset]
        transformed = TrajectoryDataSet(episodes, data.dataset_name + "_block_relative", data.timing,
                                        None if data.episode_ids is None else list(data.episode_ids))
        inputs, outputs = HankelMatrixGenerator(past, future).generate_hankel_matrices(prepare_dataset(transformed))
        anchors = outputs[(past - 1) * 6:past * 6].T
        first_input = 0 if transform_past_inputs else past
        input_count = past + future - first_input
        targets = inputs[first_input * 2:].T.reshape(-1, input_count, 2)
        inputs = inputs.copy()
        inputs[first_input * 2:] = self.encode_actions(targets, anchors).reshape(-1, input_count * 2).T
        return inputs, outputs

    def standardized_action_map(self, anchor, mean, scale):
        """Return offset/matrix such that world_action = offset + matrix @ z.

        ``z`` is a standardized controller action; mean/scale are the controller's
        two-coordinate action normalization, before conversion to world units.
        """
        mean = _pair(mean, "Action mean")
        scale = _pair(scale, "Action scale", positive=True)
        if self.name == "world":
            return mean.copy(), np.diag(scale)
        anchor = _observations(anchor)
        if anchor.shape != (6,):
            raise ValueError("An input-cost/bounds map requires one planning anchor")
        offset = self.decode_actions(mean, anchor)
        cosine, sine = anchor[4:6]
        matrix = np.array([[cosine * scale[0], -sine * scale[1]],
                           [sine * scale[0], cosine * scale[1]]])
        return offset, matrix

    def input_constraints(self, anchor, mean, scale, lower=0.0, upper=512.0):
        """Return linear bounds on standardized actions for the world workspace."""
        mean = _pair(mean, "Action mean")
        scale = _pair(scale, "Action scale", positive=True)
        lower = _pair(lower, "Lower workspace bound", scalar=True)
        upper = _pair(upper, "Upper workspace bound", scalar=True)
        if np.any(lower > upper):
            raise ValueError("Lower workspace bounds must not exceed upper bounds")
        normals = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        if self.name == "world":
            normalized_lower = (lower - mean) / scale
            normalized_upper = (upper - mean) / scale
            return normals, np.array([normalized_upper[0], -normalized_lower[0],
                                      normalized_upper[1], -normalized_lower[1]])
        offset, matrix = self.standardized_action_map(anchor, mean, scale)
        bounds = np.array([upper[0], -lower[0], upper[1], -lower[1]])
        return normals @ matrix, bounds - normals @ offset

    def input_cost(self, anchor, mean, scale, world_mean, world_scale, input_weight):
        """Preserve the original physical input penalty in controller coordinates.

        The returned quadratic matrix/reference represent
        input_weight * ||(world_action - world_mean) / world_scale||^2.
        This prevents a representation switch from silently changing the target
        of Select-DPC's input regularization.
        """
        mean = _pair(mean, "Action mean")
        scale = _pair(scale, "Action scale", positive=True)
        world_mean = _pair(world_mean, "World action mean")
        world_scale = _pair(world_scale, "World action scale", positive=True)
        if not np.isfinite(input_weight) or input_weight < 0:
            raise ValueError("Input weight must be finite and nonnegative")
        reference = (self.encode_actions(world_mean, anchor) - mean) / scale
        if self.name == "world":
            return np.diag(input_weight * (scale / world_scale) ** 2), reference
        _, matrix = self.standardized_action_map(anchor, mean, scale)
        scaled_matrix = matrix / world_scale[:, None]
        return input_weight * (scaled_matrix.T @ scaled_matrix), reference
