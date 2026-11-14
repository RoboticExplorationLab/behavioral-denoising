"""Reacher-v4 with fingertip features and an ordered sequence of references."""

import gymnasium as gym
import numpy as np

from src.interfaces import Task
from ._gym import GymDynamics


class ReacherDynamics(GymDynamics):
    def __init__(self, max_episode_steps=100, render_mode=None):
        super().__init__(gym.make("Reacher-v4", max_episode_steps=max_episode_steps, render_mode=render_mode))


class ReacherTask(Task):
    observation_dim = 8
    action_dim = 2

    def __init__(self, max_episode_steps=100, targets=None, target_seed=42,
                 target_tolerance=0.005, render_mode=None, dynamics=None):
        super().__init__(dynamics if dynamics is not None else ReacherDynamics(max_episode_steps, render_mode))
        if targets is None:
            angles = np.linspace(3 * np.pi / 4, 2 * np.pi, 4)
            distances = np.random.RandomState(target_seed).uniform(0.04, 0.15, 4)
            targets = [[0.043, 0.092]]
            targets += [[d * np.cos(a), d * np.sin(a)] for a, d in zip(angles, distances)]
            targets = np.asarray(targets)
            targets[:, 1] = np.minimum(targets[:, 1], 0.09)
        self.targets = np.asarray(targets, dtype=float).copy()
        if self.targets.ndim != 2 or self.targets.shape[1] != 2 or not len(self.targets):
            raise ValueError("targets must be a nonempty list of [x, y] positions")
        self.target_tolerance = target_tolerance
        self.target_index = 0

    def reset(self, seed=None):
        self.target_index = 0
        return super().reset(seed)

    def observation(self, raw):
        observation = np.asarray(raw[:8]).copy()
        observation[4:6] += raw[8:10]
        return observation

    def reference(self, raw, step):
        fingertip = np.asarray(raw[4:6]) + raw[8:10]
        if np.linalg.norm(fingertip - self.targets[self.target_index]) < self.target_tolerance:
            self.target_index = (self.target_index + 1) % len(self.targets)
        target = self.targets[self.target_index]
        return np.array([0, 0, 0, 0, target[0], target[1], 0, 0], dtype=float)
