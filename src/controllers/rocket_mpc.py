"""Linear MPC demonstrator for the Select-DPC rocket landing task.

The model is the simulator's own linearisation about hover (coco_rocket_lander SystemModel, exact
discretisation at the simulator's 60 Hz), in the normalised inputs the simulator takes. The MPC
uses the rocket config's cost (Q on the state, R on the input's deviation from hover thrust), input
bounds and state bounds, and follows the task's reference (a hover point above the pad, then the
pad), so it is the model-based counterpart of the DeePC family on this task.
"""

import cvxpy as cp
import numpy as np


class RocketMPC:
    def __init__(self, task, Q, R, horizon=60, state_bounds=None):
        from coco_rocket_lander.env.system_model import SystemModel

        env = task.dynamics.env.unwrapped
        model = SystemModel(env)
        model.calculate_linear_system_matrices()
        model.discretize_system_matrices(env.cfg.update_time)
        state_transition, input_transition = model.get_discrete_linear_system_matrices()
        mass, _ = env.get_mass_properties()
        limits = [env.cfg.main_engine_thrust_limits, env.cfg.side_engine_thrust_limits, env.cfg.nozzle_angle_limits]
        self.low = np.array([l[0] for l in limits], float)
        self.high = np.array([l[1] for l in limits], float)
        self.hover = np.array([-env.cfg.gravity * mass / env.cfg.main_engine_thrust, 0.0, 0.0])
        self.horizon = int(horizon)
        # The reference enters as Q r, so the problem stays parameter-affine (DPP) and compiles once.
        self.Q = np.asarray(Q, float)
        self.x0 = cp.Parameter(6)
        self.Qr = cp.Parameter(6)
        predicted_states = cp.Variable((6, self.horizon + 1))
        predicted_actions = cp.Variable((3, self.horizon))
        cost = 0
        constraints = [predicted_states[:, 0] == self.x0]
        for step in range(self.horizon):
            constraints += [predicted_states[:, step + 1] == state_transition @ predicted_states[:, step]
                            + input_transition @ (predicted_actions[:, step] - self.hover),
                            self.low <= predicted_actions[:, step], predicted_actions[:, step] <= self.high]
            cost += (cp.quad_form(predicted_states[:, step + 1], self.Q) - 2 * self.Qr @ predicted_states[:, step + 1]
                     + cp.quad_form(predicted_actions[:, step] - self.hover, np.asarray(R, float)))
        if state_bounds is not None:  # (A_y, b_y) from the config, on the predicted states.
            A_y, b_y = (np.asarray(v, float) for v in state_bounds)
            constraints += [A_y @ predicted_states[:, step] <= b_y.ravel() for step in range(1, self.horizon + 1)]
        self.U = predicted_actions
        self.problem = cp.Problem(cp.Minimize(cost), constraints)
        self.reset()

    def reset(self, seed=None):
        self.plan = None
        self.status = None

    def act(self, observation, reference):
        self.x0.value = np.asarray(observation, float)[:6]
        self.Qr.value = self.Q @ np.asarray(reference, float)[:6]
        try:
            # Use an iteration-based rho update schedule independent of machine load.
            self.problem.solve(solver=cp.OSQP, warm_start=True, max_iter=20000, eps_abs=1e-5, eps_rel=1e-5,
                               adaptive_rho_interval=25)
            self.status = self.problem.status
        except cp.error.SolverError:
            self.status = "solver_error"
        if self.status in ("optimal", "optimal_inaccurate") and self.U.value is not None:
            self.plan = np.asarray(self.U.value)
        elif self.plan is not None:  # Shift the last accepted plan.
            self.plan = np.column_stack([self.plan[:, 1:], self.plan[:, -1]])
        else:
            return self.hover.copy()
        return np.clip(self.plan[:, 0], self.low, self.high)


class RocketPID:
    """Simulator PID with saturated position errors for distant landing starts."""

    def __init__(self, task, clip_dx=1.0, clip_dy=1.0, follow_reference=False):
        environment = task.dynamics.env.unwrapped
        limits = [environment.cfg.main_engine_thrust_limits, environment.cfg.side_engine_thrust_limits,
                  environment.cfg.nozzle_angle_limits]
        self.low = np.array([limit[0] for limit in limits], dtype=float)
        self.high = np.array([limit[1] for limit in limits], dtype=float)
        self.pad = np.asarray(environment.get_landing_position())[:2]
        self.clip = np.array([clip_dx, clip_dy], dtype=float)
        self.follow_reference = follow_reference
        self.reset()

    def reset(self, seed=None):
        from coco_rocket_lander.algs import PID_Benchmark
        self.pid = PID_Benchmark((10, 0, 10), (0.085, 0.001, 10.55), (5, 0, 6))

    def act(self, raw, reference):
        state = np.asarray(raw, dtype=float).copy()
        target = np.asarray(reference, dtype=float)[:2] if self.follow_reference else self.pad
        state[:2] = np.clip(state[:2] - target, -self.clip, self.clip)
        action = np.asarray(self.pid.pid_algorithm(state), dtype=float)
        return np.clip(action, self.low, self.high)
