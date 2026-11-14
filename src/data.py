"""Trajectory and DeePC parameter containers; arrays use time-major ordering."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class TrajectoryData:
    state_trajectory: NDArray
    input_trajectory: NDArray
    terminal_observation: NDArray | None = None

    def __post_init__(self):
        self.state_trajectory = np.asarray(self.state_trajectory, dtype=float)
        self.input_trajectory = np.asarray(self.input_trajectory, dtype=float)
        y, u = self.state_trajectory, self.input_trajectory
        if (y.ndim != 2 or u.ndim != 2 or y.shape[0] != u.shape[0]
                or y.shape[1] == 0 or u.shape[1] == 0):
            raise ValueError("Expected observations (N, p) and inputs (N, m), with p,m > 0")
        if not np.isfinite(y).all() or not np.isfinite(u).all():
            raise ValueError("Trajectory observations and inputs must be finite")
        if self.terminal_observation is not None:
            self.terminal_observation = np.asarray(self.terminal_observation, dtype=float)
            if (self.terminal_observation.shape != (y.shape[1],)
                    or not np.isfinite(self.terminal_observation).all()):
                raise ValueError("terminal_observation must be a finite p-vector")


@dataclass
class TrajectoryDataSet:
    dataset: list[TrajectoryData]
    dataset_name: str
    timing: str = "pre_action"
    episode_ids: list[str] | None = None

    def __post_init__(self):
        if self.timing not in ("pre_action", "post_action"):
            raise ValueError("timing must be pre_action or post_action")
        if self.episode_ids is not None:
            self.episode_ids = [str(identity) for identity in self.episode_ids]
        if self.episode_ids is not None and len(self.episode_ids) != len(self.dataset):
            raise ValueError("Each episode must have one episode_id")
        shapes = {(item.state_trajectory.shape[1], item.input_trajectory.shape[1])
                  for item in self.dataset}
        if len(shapes) > 1:
            raise ValueError("All episodes must have the same observation/input dimensions")

    def __add__(self, other):
        if self.timing != other.timing:
            raise ValueError("Cannot combine datasets with different observation timing")
        new_dataset = deepcopy(self)
        new_dataset.dataset.extend(deepcopy(other.dataset))
        new_dataset.episode_ids = (
            self.episode_ids + other.episode_ids
            if self.episode_ids is not None and other.episode_ids is not None else None
        )
        new_dataset.dataset_name += f"_plus_{other.dataset_name}"
        return new_dataset


def prepare_dataset(data, episode_order="preserve"):
    """Copy episodes into controller pairs (u[k], y[k+1]), preserving raw data.

    Pre-action episodes store (u[k], y[k]). Their final action is discarded only
    when its terminal observation is unavailable. Never shift across episodes.
    ``upstream_filename`` sorts by the complete CSV filename,
    e.g. 0,10,...,19,1,20,..., rather than integer or bare-string ID ordering.
    """
    if not isinstance(data, TrajectoryDataSet):
        raise TypeError("prepare_dataset requires a TrajectoryDataSet with explicit timing")
    if not data.dataset:
        raise ValueError("Cannot prepare an empty dataset")
    if data.timing not in ("pre_action", "post_action"):
        raise ValueError("timing must be pre_action or post_action")
    if episode_order not in ("preserve", "upstream_filename"):
        raise ValueError("episode_order must be preserve or upstream_filename")
    indices = list(range(len(data.dataset)))
    if episode_order == "upstream_filename":
        if data.episode_ids is None:
            raise ValueError("upstream_filename ordering requires original episode_ids")
        indices.sort(key=lambda i: f"{data.episode_ids[i]}_random_data_state.csv")
    episodes, kept_indices = [], []
    for i in indices:
        episode = data.dataset[i]
        y, u = episode.state_trajectory, episode.input_trajectory
        if data.timing == "post_action" or len(u) == 0:
            aligned_y, aligned_u = y.copy(), u.copy()
        elif episode.terminal_observation is None:
            aligned_y, aligned_u = y[1:].copy(), u[:-1].copy()
        else:
            aligned_y = np.vstack((y[1:], episode.terminal_observation))
            aligned_u = u.copy()
        if len(aligned_u):
            episodes.append(TrajectoryData(aligned_y, aligned_u))
            kept_indices.append(i)
    if not episodes:
        raise ValueError("No complete action/observation pairs remain after alignment")
    episode_ids = None if data.episode_ids is None else [data.episode_ids[i] for i in kept_indices]
    return TrajectoryDataSet(episodes, data.dataset_name, "post_action", episode_ids)


@dataclass
class DeePCConstraints:
    A_u: NDArray
    b_u: NDArray
    A_y: NDArray
    b_y: NDArray


@dataclass
class DeePCCost:
    Q: NDArray
    R: NDArray
    slack_cost: float
    regularizer_cost_g_1: float
    regularizer_cost_g_pi: float


@dataclass
class DeePCDims:
    """Dimension struct for deepc

    p: measurement size
    m: input size
    """

    T_past: int
    T_fut: int
    p: int
    m: int


@dataclass
class DeePCControllerArgs:
    trajectory_data: TrajectoryDataSet
    deepc_dims: DeePCDims
    T_hankel: int
    controller_costs: DeePCCost
    controller_constraints: DeePCConstraints
    input_reference: np.array
    verbose: bool
