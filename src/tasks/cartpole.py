"""MuJoCo cartpole with swing-up initial conditions."""

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from src.interfaces import Task
from ._gym import GymDynamics


class CartpoleEnv(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 25}

    def __init__(self, render_mode=None, angle_termination=7.0, reset_state=None, reset_noise=None):
        if angle_termination is not None and (not np.isfinite(angle_termination) or angle_termination <= 0):
            raise ValueError("angle_termination must be positive or None")
        self.angle_termination = angle_termination
        self.reset_state = None if reset_state is None else np.asarray(reset_state, dtype=float)
        self.reset_noise = np.zeros(4) if reset_noise is None else np.asarray(reset_noise, dtype=float)
        if reset_noise is not None and reset_state is None:
            raise ValueError("reset_noise requires reset_state")
        if self.reset_state is not None:
            if (self.reset_state.shape != (4,) or self.reset_noise.shape != (4,)
                    or not np.isfinite(self.reset_state).all() or not np.isfinite(self.reset_noise).all()
                    or (self.reset_noise < 0).any()):
                raise ValueError("reset_state/reset_noise must be finite four-vectors with nonnegative noise")
        model = Path(__file__).parent / "assets" / "inverted_pendulum_swingup.xml"
        super().__init__(
            str(model), 2,
            observation_space=Box(-np.inf, np.inf, shape=(4,), dtype=np.float64),
            default_camera_config={"trackbodyid": 0, "distance": 2.04},
            render_mode=render_mode,
        )

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def reset_model(self):
        if self.reset_state is not None:
            state = self.reset_state + self.np_random.uniform(-self.reset_noise, self.reset_noise)
            self.set_state(state[:2], state[2:])
            return self._get_obs()
        qpos = np.array([0, np.pi]) + self.np_random.uniform(-0.1, 0.1, self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(0.001, 0.02, self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        terminated = bool(
            not np.isfinite(observation).all()
            or (self.angle_termination is not None and abs(observation[1]) > self.angle_termination)
        )
        if self.render_mode == "human":
            self.render()
        return observation, 1.0, terminated, False, {}


class CartpoleDynamics(GymDynamics):
    def __init__(self, max_episode_steps=1000, render_mode=None, angle_termination=7.0,
                 reset_state=None, reset_noise=None):
        env = CartpoleEnv(render_mode=render_mode, angle_termination=angle_termination,
                          reset_state=reset_state, reset_noise=reset_noise)
        super().__init__(gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps))


class CartpoleTask(Task):
    observation_dim = 5
    action_dim = 1

    def __init__(self, max_episode_steps=1000, render_mode=None, dynamics=None, angle_termination=7.0,
                 reset_state=None, reset_noise=None):
        super().__init__(dynamics if dynamics is not None else CartpoleDynamics(
            max_episode_steps, render_mode, angle_termination, reset_state, reset_noise,
        ))

    def apply_evaluation_force(self, force):
        """Apply an external cart force in newtons, separately from its actuator."""
        self.dynamics.env.unwrapped.data.qfrc_applied[0] = force

    def observation(self, raw):
        x, angle, velocity, angular_velocity = raw
        return np.array([x, np.sin(angle), np.cos(angle), velocity, angular_velocity])

    def reference(self, raw, step):
        return np.array([0.0, 0.0, 1.0, 0.0, 0.0])
