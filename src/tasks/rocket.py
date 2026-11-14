"""RocketLander adapter for hover and landing tasks."""

import numpy as np

from src.interfaces import Task
from ._gym import GymDynamics


def _fix_observation_bounds():
    """Correct the pinned simulator's angular-speed upper bound without changing dynamics."""
    import types
    from gymnasium import spaces
    from coco_rocket_lander.env import rocketlander

    def box(low, high, *args, **kwargs):
        high = np.asarray(high, dtype=float)
        corrected_high = np.where(np.isneginf(high), np.inf, high)
        return spaces.Box(low, corrected_high, *args, **kwargs)

    simulator_spaces = types.ModuleType("spaces")
    simulator_spaces.__dict__.update(spaces.__dict__)
    simulator_spaces.Box = box
    rocketlander.spaces = simulator_spaces


class RocketDynamics(GymDynamics):
    def __init__(self, initial_condition=(0.2, 0.9, 0.1), max_episode_steps=1000,
                 render_mode=None, **env_args):
        try:
            import coco_rocket_lander  # noqa: F401; registers the environment
        except ImportError as exc:
            raise ImportError("Rocket tasks require coco_rocket_lander; see the rocket dependency instructions in README.md") from exc
        import gymnasium as gym

        _fix_observation_bounds()
        args = dict(env_args)
        if "random_initial_position" not in args:
            initial_condition = np.asarray(initial_condition)
            if initial_condition.size not in (3, 6):
                raise ValueError("initial_condition must contain 3 positions or 6 state values")
            args["initial_position" if initial_condition.size == 3 else "initial_state"] = initial_condition
        super().__init__(gym.make("coco_rocket_lander/RocketLander-v0", args=args,
                                 render_mode=render_mode, max_episode_steps=max_episode_steps))


class RocketTask(Task):
    observation_dim = 6
    action_dim = 3

    def __init__(self, mode="hover", initial_condition=(0.2, 0.9, 0.1),
                 max_episode_steps=1000, render_mode=None, dynamics=None, **env_args):
        if mode not in ("hover", "landing"):
            raise ValueError("mode must be 'hover' or 'landing'")
        super().__init__(dynamics if dynamics is not None else RocketDynamics(
            initial_condition, max_episode_steps, render_mode, **env_args))
        self.mode = mode
        self._approach_hover_point = True
        self._is_landed = False
        self._landing_position = np.append(self.dynamics.env.unwrapped.get_landing_position(), np.zeros(3))

    def reset(self, seed=None):
        self._approach_hover_point = True
        self._is_landed = False
        observation = super().reset(seed)
        self._landing_position = np.append(self.dynamics.env.unwrapped.get_landing_position(), np.zeros(3))
        return observation

    def observation(self, raw):
        return np.asarray(raw[:6]).copy()

    def include_demonstration(self, raw):
        return not np.any(raw[6:])

    def reference(self, raw, step):
        landing = self._landing_position.copy()
        if self.mode == "landing":
            if abs(raw[0] - landing[0]) < 2 and raw[2] ** 2 + raw[3] ** 2 < 1:
                self._approach_hover_point = False
        if self.mode == "hover" or self._approach_hover_point:
            landing[1] += 5
        return landing

    def override_action(self, raw):
        if np.any(raw[6:]):
            self._is_landed = True
        if self._is_landed:
            if np.sign(raw[4]) == np.sign(raw[5]) and abs(raw[4]) > 0.1:
                return np.array([0, 5 * raw[4], 0])
            return np.zeros(3)
        return None

    def is_successful(self, raw):
        if self.mode == "landing":
            return bool(np.any(raw[6:]))
        return not (abs(raw[4]) > 0.5 or raw[0] < 5 or raw[0] > 25 or raw[1] > 25 or raw[1] < 5)
