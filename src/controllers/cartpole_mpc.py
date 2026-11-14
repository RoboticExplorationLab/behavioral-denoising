"""A single nonlinear MPC demonstration controller for the MuJoCo cartpole."""

from time import perf_counter

import casadi as ca
import mujoco
import numpy as np

from control_bench.interfaces import Controller
from .cartpole_model import CartpoleModel


class CartpoleMPCController(CartpoleModel, Controller):
    def __init__(self, task, horizon_steps=100, state_weights=(2, 30, 0.1, 0.1),
                 terminal_weights=(20, 300, 2, 10), action_weight=0.05,
                 cart_limit=1.8, action_limit=3.0, max_iterations=300,
                 solver_tolerance=1e-6, initialization_seconds=2.0):
        self.horizon_steps = int(horizon_steps)
        if self.horizon_steps != horizon_steps or self.horizon_steps < 2:
            raise ValueError("horizon_steps must be an integer of at least two")
        self.state_weights = self._weights(state_weights)
        self.terminal_weights = self._weights(terminal_weights)
        for name, value in {
            "action_weight": action_weight, "cart_limit": cart_limit,
            "action_limit": action_limit, "solver_tolerance": solver_tolerance,
            "initialization_seconds": initialization_seconds,
        }.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if int(max_iterations) != max_iterations or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.action_limit = float(action_limit)
        self.cart_limit = float(cart_limit)
        self.initialization_seconds = float(initialization_seconds)
        self.solver_tolerance = float(solver_tolerance)
        env = task.dynamics.env.unwrapped
        model = env.model
        if (model.nq, model.nv, model.nu) != (2, 2, 1):
            raise ValueError("Cartpole MPC requires the two-joint MuJoCo cartpole")
        if model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4:
            raise ValueError("Cartpole MPC requires MuJoCo's RK4 integrator")
        if np.any(model.dof_armature != 0):
            raise ValueError("Cartpole MPC currently requires zero joint armature")
        if self.action_limit > np.min(np.abs(model.actuator_ctrlrange[0])):
            raise ValueError("action_limit exceeds the MuJoCo actuator bounds")
        if self.cart_limit >= np.min(np.abs(model.jnt_range[0])):
            raise ValueError("cart_limit must lie inside the MuJoCo joint limits")
        self.dt = float(model.opt.timestep * env.frame_skip)
        self._make_dynamics(model, env.frame_skip)
        self._make_optimizer(float(action_weight), int(max_iterations), model.jnt_range[1])
        self.reset()

    @staticmethod
    def _weights(values):
        weights = np.asarray(values, dtype=float)
        if weights.shape != (4,) or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("State weights must be four finite, nonnegative numbers")
        return weights

    def _state_cost(self, state, weights):
        return (weights[0] * state[0] ** 2
                + weights[1] * (1 - ca.cos(state[1] - self.upright_angle))
                + weights[2] * state[2] ** 2 + weights[3] * state[3] ** 2)

    def _make_optimizer(self, action_weight, max_iterations, angle_limits):
        opti = self.optimizer = ca.Opti()
        self.states = opti.variable(4, self.horizon_steps + 1)
        self.actions = opti.variable(1, self.horizon_steps)
        self.initial_state = opti.parameter(4)
        opti.subject_to(self.states[:, 0] == self.initial_state)
        objective = 0
        for step in range(self.horizon_steps):
            state, action = self.states[:, step], self.actions[step]
            opti.subject_to(self.states[:, step + 1] == self.discrete_model(state, action))
            objective += self.dt * (self._state_cost(state, self.state_weights) + action_weight * action ** 2)
        objective += self._state_cost(self.states[:, -1], self.terminal_weights)
        opti.subject_to(opti.bounded(-self.action_limit, self.actions, self.action_limit))
        opti.subject_to(opti.bounded(-self.cart_limit, self.states[0, 1:], self.cart_limit))
        opti.subject_to(opti.bounded(float(angle_limits[0] + 0.5), self.states[1, 1:], float(angle_limits[1] - 0.5)))
        self.objective = objective
        opti.minimize(objective)
        opti.solver("ipopt", {"expand": True, "print_time": False}, {
            "print_level": 0, "sb": "yes", "max_iter": max_iterations,
            "tol": self.solver_tolerance,
        })

    def reset(self, seed=None):
        # Optimization is deterministic; the seed does not add action noise.
        self.reset_episode()

    def reset_episode(self):
        self.plan_states = None
        self.plan_actions = None
        self._diagnostics = {"status": "not_run", "success": None}

    def _initial_guesses(self, state):
        times = np.arange(self.horizon_steps + 1) * self.dt
        progress = np.minimum(times / self.initialization_seconds, 1.0)
        lower = self.upright_angle + 2 * np.pi * np.floor((state[1] - self.upright_angle) / (2 * np.pi))
        guesses = []
        # Both swing directions avoid committing to the symmetric hanging state.
        for label, target in [("lower_upright", lower), ("upper_upright", lower + 2 * np.pi)]:
            states = np.tile(state[:, None], (1, self.horizon_steps + 1))
            states[1] = state[1] + (target - state[1]) * progress
            states[3] = np.gradient(states[1], self.dt)
            states[0] = state[0] * (1 - progress) + 0.5 * np.sign(target - state[1]) * np.sin(np.pi * progress)
            states[:, 0] = state
            guesses.append((label, states, np.zeros(self.horizon_steps)))
        return guesses

    def _solve(self, label, states, actions):
        self.optimizer.set_initial(self.states, states)
        self.optimizer.set_initial(self.actions, actions)
        start = perf_counter()
        try:
            solution = self.optimizer.solve()
        except RuntimeError as error:
            try:
                stats = self.optimizer.stats()
            except RuntimeError:
                stats = {}
            return None, {
                "initialization": label, "success": False,
                "status": stats.get("return_status", "solver_error"),
                "iterations": stats.get("iter_count"), "seconds": perf_counter() - start,
                "error": str(error),
            }
        stats = solution.stats()
        objective = float(solution.value(self.objective))
        record = {
            "initialization": label, "success": True, "status": stats["return_status"],
            "iterations": int(stats["iter_count"]), "seconds": perf_counter() - start,
            "objective": objective if np.isfinite(objective) else None,
        }
        plan = np.asarray(solution.value(self.states)), np.asarray(solution.value(self.actions)).reshape(-1)
        constraints = np.asarray(solution.value(self.optimizer.g)).reshape(-1)
        lower = np.asarray(solution.value(self.optimizer.lbg)).reshape(-1)
        upper = np.asarray(solution.value(self.optimizer.ubg)).reshape(-1)
        violation = float(np.max(np.maximum(lower - constraints, constraints - upper), initial=0))
        record["max_constraint_violation"] = violation if np.isfinite(violation) else None
        if (not all(np.isfinite(values).all() for values in plan)
                or not np.isfinite(objective) or not np.isfinite(violation)
                or violation > max(1e-5, 10 * self.solver_tolerance)):
            record.update(success=False, solver_status=record["status"], status="solution_failed_validation")
            return None, record
        return plan, record

    def act(self, state, reference):
        state = np.asarray(state, dtype=float)
        if state.shape != (4,) or not np.isfinite(state).all():
            raise ValueError("Cartpole MPC requires a finite raw state with shape (4,)")
        self.optimizer.set_value(self.initial_state, state)
        if self.plan_states is None:
            guesses = self._initial_guesses(state)
        else:
            states = np.column_stack([self.plan_states[:, 1:], self.plan_states[:, -1]])
            actions = np.r_[self.plan_actions[1:], self.plan_actions[-1]]
            guesses = [("shifted_plan", states, actions)]
        attempts, candidates = [], []
        for label, states, actions in guesses:
            plan, record = self._solve(label, states, actions)
            attempts.append(record)
            if plan is not None:
                candidates.append((plan, record))
        if not candidates and self.plan_states is not None:
            for label, states, actions in self._initial_guesses(state):
                plan, record = self._solve(label, states, actions)
                attempts.append(record)
                if plan is not None:
                    candidates.append((plan, record))
        self._diagnostics = {
            "success": bool(candidates), "attempts": attempts,
            "solve_seconds": sum(attempt["seconds"] for attempt in attempts),
        }
        if not candidates:
            self._diagnostics["status"] = "all_initializations_failed"
            raise RuntimeError("Cartpole MPC failed: " + ", ".join(attempt["status"] for attempt in attempts))
        plan, best = min(candidates, key=lambda candidate: candidate[1]["objective"])
        self.plan_states, self.plan_actions = plan
        self._diagnostics.update({key: best[key] for key in (
            "status", "objective", "iterations", "initialization", "max_constraint_violation",
        )})
        # Remove only solver tolerance overshoot at the actuator bounds.
        return np.clip(self.plan_actions[:1], -self.action_limit, self.action_limit)

    def diagnostics(self):
        return self._diagnostics
