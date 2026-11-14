"""Small adapter for Gymnasium's reset and step return values."""

import numpy as np

from src.interfaces import Dynamics, Step


class GymDynamics(Dynamics):
    def __init__(self, env):
        self.env = env

    def reset(self, seed=None):
        observation, _ = self.env.reset(seed=seed)
        return np.asarray(observation).copy()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return Step(np.asarray(observation).copy(), reward, terminated, truncated, info)

    def close(self):
        self.env.close()
