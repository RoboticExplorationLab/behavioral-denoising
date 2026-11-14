"""Small contracts shared by tasks, demonstration controllers, and policies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Step:
    observation: np.ndarray
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict = field(default_factory=dict)


class Dynamics(ABC):
    @abstractmethod
    def reset(self, seed=None):
        """Return the initial raw observation."""

    @abstractmethod
    def step(self, action) -> Step:
        """Advance one control step."""

    def close(self):
        pass


class Task(ABC):
    """Task-specific observations and references around a dynamics instance."""

    def __init__(self, dynamics):
        self.dynamics = dynamics

    def reset(self, seed=None):
        return self.dynamics.reset(seed)

    @abstractmethod
    def observation(self, raw):
        """Map a raw observation to the policy's feature vector."""

    @abstractmethod
    def reference(self, raw, step):
        """Return the policy reference at this step."""

    def close(self):
        self.dynamics.close()


class Controller(ABC):
    """A demonstration generator, optionally given a model at construction."""

    @abstractmethod
    def reset(self, seed=None):
        pass

    @abstractmethod
    def act(self, state, reference):
        pass


class Policy(ABC):
    """Online inference only; offline fitting belongs outside this contract."""

    @abstractmethod
    def reset(self, seed=None):
        pass

    @abstractmethod
    def act(self, observation, reference):
        """Return one action, updating policy-owned history."""

    def diagnostics(self):
        return {}
