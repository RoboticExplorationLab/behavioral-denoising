"""Small, pickle-free trajectory archives with explicit timing and episode identity."""

from pathlib import Path

import numpy as np

from .data import TrajectoryData, TrajectoryDataSet


def save_dataset(path, data, task_name):
    if not data.dataset:
        raise ValueError("Cannot save an empty dataset")
    if any(len(trajectory.input_trajectory) == 0 for trajectory in data.dataset):
        raise ValueError("Cannot save empty episodes")
    p = data.dataset[0].state_trajectory.shape[1]
    terminals = [trajectory.terminal_observation for trajectory in data.dataset]
    arrays = dict(
        states=np.concatenate([trajectory.state_trajectory for trajectory in data.dataset]),
        actions=np.concatenate([trajectory.input_trajectory for trajectory in data.dataset]),
        episode_ends=np.cumsum([len(trajectory.input_trajectory) for trajectory in data.dataset]),
        timing=data.timing, task=task_name, name=data.dataset_name,
        terminal_observations=np.array([np.zeros(p) if value is None else value for value in terminals]),
        has_terminal_observation=np.array([value is not None for value in terminals]),
    )
    if data.episode_ids is not None:
        arrays["episode_ids"] = np.asarray(data.episode_ids, dtype=str)
    np.savez_compressed(Path(path), **arrays)


def load_dataset(path, task_name):
    with np.load(path, allow_pickle=False) as saved:
        if "task" not in saved or str(saved["task"]) != task_name:
            raise ValueError("Dataset must record the same task as the experiment")
        states, actions, ends = saved["states"], saved["actions"], saved["episode_ends"]
        if (states.ndim != 2 or actions.ndim != 2 or len(states) != len(actions)
                or ends.ndim != 1 or not len(ends) or ends[-1] != len(actions)
                or not np.issubdtype(ends.dtype, np.integer)
                or np.any(np.diff(np.r_[0, ends]) <= 0)):
            raise ValueError("Invalid trajectory arrays or episode boundaries")
        terminals = saved.get("terminal_observations", np.zeros((len(ends), states.shape[1])))
        has_terminal = saved.get("has_terminal_observation", np.zeros(len(ends), dtype=bool))
        if terminals.shape != (len(ends), states.shape[1]) or has_terminal.shape != (len(ends),):
            raise ValueError("Invalid terminal observation arrays")
        episodes = []
        for i, (start, end) in enumerate(zip(np.r_[0, ends[:-1]], ends)):
            episodes.append(TrajectoryData(states[start:end], actions[start:end],
                            terminals[i] if has_terminal[i] else None))
        identities = saved["episode_ids"].tolist() if "episode_ids" in saved else None
        return TrajectoryDataSet(episodes, str(saved.get("name", task_name)),
                                 str(saved["timing"]), episode_ids=identities)
