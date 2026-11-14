"""Nonlinear MPC demonstrator for the Select-DPC reacher (Reacher-v4), built like the cartpole MPC.

The two-link model is read from the MuJoCo model: link masses, centres of mass and inertias
(the fingertip body is lumped into the second link), joint armature, damping and motor gear.
The arm moves in the horizontal plane, so gravity exerts no joint torque. Two RK4 substeps per
control step match MuJoCo's integrator and frame skip. The MPC minimises the reacher config's
stage cost (Q on the fingertip and the joint speeds, R on the inputs) under its input bound
and fingertip-height constraint, so the expert optimises what the DeePC family optimises.
"""

from time import perf_counter

import casadi as ca
import mujoco
import numpy as np


def _zz_inertia(model, body):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.body_iquat[body])
    rotation = rotation.reshape(3, 3)
    return float((rotation @ np.diag(model.body_inertia[body]) @ rotation.T)[2, 2])


class ReacherModel:
    def __init__(self, env):
        model = env.unwrapped.model
        if model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4:
            raise ValueError("The reacher model assumes MuJoCo's RK4 integrator")
        link1, link2, tip = (model.body(name).id for name in ("body0", "body1", "fingertip"))
        m1, c1, i1 = float(model.body_mass[link1]), float(model.body_ipos[link1, 0]), _zz_inertia(model, link1)
        mb, cb, ib = float(model.body_mass[link2]), float(model.body_ipos[link2, 0]), _zz_inertia(model, link2)
        tip_offset = float(model.body_pos[tip, 0])
        mf, cf, i_f = float(model.body_mass[tip]), tip_offset + float(model.body_ipos[tip, 0]), _zz_inertia(model, tip)
        m2 = mb + mf
        c2 = (mb * cb + mf * cf) / m2
        i2 = ib + mb * (cb - c2) ** 2 + i_f + mf * (cf - c2) ** 2
        self.parameters = p = dict(
            m1=m1, c1=c1, i1=i1, m2=m2, c2=c2, i2=i2, l1=float(model.body_pos[link2, 0]), tip=tip_offset,
            armature=model.dof_armature[:2].astype(float).copy(), damping=model.dof_damping[:2].astype(float).copy(),
            gear=model.actuator_gear[:2, 0].astype(float).copy(), timestep=float(model.opt.timestep),
            frame_skip=int(env.unwrapped.frame_skip), joint1_range=model.jnt_range[1].astype(float).copy())
        self.dt = p["timestep"] * p["frame_skip"]
        state, action = ca.SX.sym("state", 4), ca.SX.sym("action", 2)
        q0, q1, w0, w1 = ca.vertsplit(state)
        coupling = p["m2"] * p["l1"] * p["c2"]
        mass = ca.vertcat(
            ca.horzcat(p["i1"] + p["m1"] * p["c1"] ** 2 + p["i2"] + p["m2"] * (p["l1"] ** 2 + p["c2"] ** 2)
                       + 2 * coupling * ca.cos(q1) + p["armature"][0],
                       p["i2"] + p["m2"] * p["c2"] ** 2 + coupling * ca.cos(q1)),
            ca.horzcat(p["i2"] + p["m2"] * p["c2"] ** 2 + coupling * ca.cos(q1),
                       p["i2"] + p["m2"] * p["c2"] ** 2 + p["armature"][1]))
        h = coupling * ca.sin(q1)
        bias = ca.vertcat(-h * (2 * w0 * w1 + w1 ** 2), h * w0 ** 2)
        torque = p["gear"] * action - p["damping"] * ca.vertcat(w0, w1) - bias
        derivative = ca.vertcat(w0, w1, ca.solve(mass, torque))
        f = ca.Function("reacher_derivative", [state, action], [derivative])
        x, dt = state, p["timestep"]
        for _ in range(p["frame_skip"]):
            k1 = f(x, action)
            k2 = f(x + dt * k1 / 2, action)
            k3 = f(x + dt * k2 / 2, action)
            k4 = f(x + dt * k3, action)
            x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self.transition = ca.Function("reacher_transition", [state, action], [x])
        angles = ca.SX.sym("angles", 2)
        self.fingertip = ca.Function("fingertip", [angles], [ca.vertcat(
            p["l1"] * ca.cos(angles[0]) + p["tip"] * ca.cos(angles[0] + angles[1]),
            p["l1"] * ca.sin(angles[0]) + p["tip"] * ca.sin(angles[0] + angles[1]))])

    @staticmethod
    def state_from_observation(observation):
        """Joint angles and speeds from the task observation (cos q, sin q, fingertip, speeds)."""
        observation = np.asarray(observation, dtype=float)
        return np.array([np.arctan2(observation[2], observation[0]), np.arctan2(observation[3], observation[1]),
                         observation[6], observation[7]])


class ReacherMPC:
    """act(observation, reference) with the task's 8-dimensional observation and reference."""

    def __init__(self, env, horizon=30, position_weight=40000.0, speed_weight=10.0, action_weight=10.0,
                 action_limit=0.5, height_limit=0.1, max_iterations=200, tolerance=1e-6):
        self.model = ReacherModel(env)
        self.horizon = int(horizon)
        self.action_limit = float(action_limit)
        opti = self.optimizer = ca.Opti()
        self.states = opti.variable(4, self.horizon + 1)
        self.actions = opti.variable(2, self.horizon)
        self.initial = opti.parameter(4)
        self.target = opti.parameter(2)
        opti.subject_to(self.states[:, 0] == self.initial)
        objective = 0
        joint1 = self.model.parameters["joint1_range"]
        for k in range(self.horizon):
            opti.subject_to(self.states[:, k + 1] == self.model.transition(self.states[:, k], self.actions[:, k]))
            tip = self.model.fingertip(self.states[:2, k + 1])
            objective += (position_weight * ca.sumsqr(tip - self.target) + speed_weight * ca.sumsqr(self.states[2:, k + 1])
                          + action_weight * ca.sumsqr(self.actions[:, k]))
            opti.subject_to(tip[1] <= height_limit)
        opti.subject_to(opti.bounded(-self.action_limit, self.actions, self.action_limit))
        opti.subject_to(opti.bounded(joint1[0] + 0.1, self.states[1, 1:], joint1[1] - 0.1))
        self.objective = objective
        opti.minimize(objective)
        opti.solver("ipopt", {"expand": True, "print_time": False},
                    {"print_level": 0, "sb": "yes", "max_iter": int(max_iterations), "tol": float(tolerance)})
        self.reset()

    def reset(self, seed=None):
        self.plan_states = None
        self.plan_actions = None
        self.last = {}
        self.status_counts = {}  # IPOPT return status -> steps, this episode (logged by the sweeps).

    def _usable(self, plan_states, plan_actions, tolerance=1e-4):
        """A stopped IPOPT iterate is used only if it is finite and nearly satisfies the model's dynamics
        (review issue E20); the action and joint bounds are enforced afterwards by clipping."""
        if not (np.isfinite(plan_states).all() and np.isfinite(plan_actions).all()):
            return False
        predicted = np.column_stack([np.asarray(self.model.transition(plan_states[:, k], plan_actions[:, k])).ravel()
                                     for k in range(self.horizon)])
        return float(np.max(np.abs(predicted - plan_states[:, 1:]))) <= tolerance

    def act(self, observation, reference):
        state = self.model.state_from_observation(observation)
        if self.plan_states is not None:  # Keep q0 continuous with the previous plan.
            state[0] += 2 * np.pi * np.round((self.plan_states[0, 1] - state[0]) / (2 * np.pi))
            states = np.column_stack([self.plan_states[:, 1:], self.plan_states[:, -1]])
            actions = np.column_stack([self.plan_actions[:, 1:], self.plan_actions[:, -1]])
        else:
            states = np.tile(state[:, None], (1, self.horizon + 1))
            actions = np.zeros((2, self.horizon))
        opti = self.optimizer
        opti.set_value(self.initial, state)
        opti.set_value(self.target, np.asarray(reference, dtype=float)[4:6])
        opti.set_initial(self.states, states)
        opti.set_initial(self.actions, actions)
        start = perf_counter()
        try:
            solution = opti.solve()
            self.plan_states = np.asarray(solution.value(self.states))
            self.plan_actions = np.asarray(solution.value(self.actions)).reshape(2, self.horizon)
            status = solution.stats()["return_status"]
        except RuntimeError:
            # Keep IPOPT's last iterate if it stops early and the iterate is usable; otherwise the shifted plan.
            status = opti.stats().get("return_status", "solver_error")
            try:
                plan_states = np.asarray(opti.debug.value(self.states))
                plan_actions = np.asarray(opti.debug.value(self.actions)).reshape(2, self.horizon)
            except RuntimeError:
                plan_states = None
            if plan_states is not None and self._usable(plan_states, plan_actions):
                self.plan_states = plan_states
                self.plan_actions = plan_actions
            else:
                self.plan_states = states
                self.plan_actions = actions
                status += " (shifted plan used)"
        self.last = dict(status=status, seconds=perf_counter() - start)
        self.status_counts[status] = self.status_counts.get(status, 0) + 1
        return np.clip(self.plan_actions[:, 0], -self.action_limit, self.action_limit)
