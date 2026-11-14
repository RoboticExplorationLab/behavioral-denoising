"""Select-DPC repository's DeePC objective with post-action trajectory timing.

A dictionary row pair is (u_k, y_{k+1}). Solving never advances control time;
``take_action`` commits exactly one command. Accepted plans are copied before
history changes, so retrieval and diagnostics cannot read a shifted/failed plan.
"""

import warnings

import cvxpy as cp
import numpy as np

from ..data import prepare_dataset
from ..interfaces import Policy
from ..optimization.deepc_qp import DeePCQP
from ..optimization.hankel import HankelMatrixGenerator


def _frozen_array(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _reference_vector(reference, p, horizon):
    reference = np.asarray(reference)
    if reference.shape == (p,):
        result = np.tile(reference, horizon)
    elif reference.shape == (horizon, p):
        result = reference.reshape(-1)
    else:
        raise ValueError("Reference must be a vector or a horizon of vectors with the policy dimensions")
    if not np.isfinite(result).all():
        raise ValueError("Reference must be finite")
    return result


def _applied_action_vector(action, dimension):
    action = np.asarray(action, dtype=float).reshape(-1)
    if action.shape != (dimension,) or not np.isfinite(action).all():
        raise ValueError("Executed action must be a finite vector of input dimension")
    return action


class DeePC(Policy):
    def __init__(
        self, H_u, H_y, dims, costs, constraints, input_reference,
        verbose=False, decompose_hankel_matrix=False, solver_options=None,
    ):
        self._dims = dims
        self._T_past, self._T_fut = dims.T_past, dims.T_fut
        self._m, self._p = dims.m, dims.p
        self._dictionary_columns = H_u.shape[1]
        self._coefficient_representation = "legacy_cholesky" if decompose_hankel_matrix else "dictionary"
        self._qp = DeePCQP(
            H_u, H_y, dims, costs, constraints, input_reference,
            decompose_hankel_matrix, verbose, solver_options,
        )
        self._g, self._u, self._y = self._qp.g, self._qp.u, self._qp.y
        self._u_past, self._y_past = self._qp.u_past, self._qp.y_past
        self._reference_traj = self._qp.reference
        self._problem = self._qp.problem
        self.reset()

    @classmethod
    def create_from_data(cls, args, decompose_hankel_matrices=False, solver_options=None):
        data = prepare_dataset(args.trajectory_data)
        H_u, H_y = HankelMatrixGenerator(
            args.deepc_dims.T_past, args.deepc_dims.T_fut, args.T_hankel,
        ).generate_hankel_matrices(data)
        return cls(
            H_u, H_y, args.deepc_dims, args.controller_costs,
            args.controller_constraints, args.input_reference, args.verbose,
            decompose_hankel_matrices, solver_options,
        )

    def reset(self, seed=None):
        self._initial_run = True
        self._initial_state_overridden = False
        self._plan = self._prev_u = None
        self._open_loop_index = 0
        self._status = self._objective = None
        self.begin_step()
        for value in [self._g, self._u, self._y, self._u_past, self._y_past,
                      self._reference_traj, self._qp.y_slack]:
            value.value = None

    def begin_step(self):
        self._plan_is_current = False
        self._executed_action = self._action_source = None
        self._awaiting_applied_action = False

    def carry_plan(self, previous):
        """Transfer the accepted, immutable plan across local solver rebuilds."""
        self._plan, self._prev_u = previous._plan, previous._prev_u
        self._open_loop_index = previous._open_loop_index
        self._plan_is_current = previous._plan_is_current

    def solve(self, reference, selected_indices=None):
        """Attempt one optimization without consuming an action or changing history."""
        self._reference_traj.value = _reference_vector(reference, self._p, self._T_fut)
        try:
            self._status = self._qp.solve()
        except cp.SolverError:
            self._status = "failed"
        accepted = self._status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        if accepted:
            values = [self._u.value, self._y.value, self._g.value, self._qp.y_slack.value, self._problem.value]
            if any(value is None or not np.isfinite(value).all() for value in values):
                self._status, accepted = "invalid_solution", False
        self._objective = self._problem.value if accepted else None
        if accepted:
            self._plan = {
                "u_past": _frozen_array(self._u_past.value),
                "y_past": _frozen_array(self._y_past.value),
                "u": _frozen_array(self._u.value), "y": _frozen_array(self._y.value),
                "g": _frozen_array(self._g.value), "slack": _frozen_array(self._qp.y_slack.value),
                "objective": self._objective, "status": self._status,
                "coefficient_representation": self._coefficient_representation,
                "dictionary_columns": self._dictionary_columns,
                "selected_indices": None if selected_indices is None else _frozen_array(selected_indices),
            }
            self._prev_u = self._plan["u"]
            self._open_loop_index = 0
            self._plan_is_current = True
        else:
            warnings.warn(f"MPC solver status: {self._status}; keeping the last accepted plan.", RuntimeWarning)
        return accepted

    def take_action(self):
        """Consume one accepted-plan action, or zero when no action remains."""
        start = self._open_loop_index * self._m
        if self._prev_u is not None and start + self._m <= len(self._prev_u):
            action = self._prev_u[start:start + self._m].copy()
            self._open_loop_index += 1
            self._action_source = "optimized" if self._plan_is_current else "fallback"
        else:
            action = np.zeros(self._m)
            self._action_source = "zero"
            warnings.warn("No remaining previous plan; sending zero input", RuntimeWarning)
        self._executed_action = action.copy()
        self._u_past.value = np.append(self._u_past.value[self._m:], action)
        self._awaiting_applied_action = True
        self._initial_state_overridden = False
        return action

    def record_applied_action(self, action):
        """Replace this step's command by its executed input, leaving its plan intact.

        Call once after ``act`` and before the next observation. Without a new
        command, return False: an external controller may have taken over, and
        its input cannot be paired with an old policy observation. The diagnostic
        action source still identifies the optimized, fallback or zero proposal.
        """
        action = _applied_action_vector(action, self._m)
        if not self._awaiting_applied_action:
            return False
        history = self._u_past.value.copy()
        history[-self._m:] = action
        self._u_past.value = history
        self._executed_action = action.copy()
        self._awaiting_applied_action = False
        return True

    def act(self, observation, reference):
        _reference_vector(reference, self._p, self._T_fut)  # Validate before changing history.
        self.begin_step()
        if not self._initial_state_overridden:
            if self._initial_run:
                self._initial_run = False
                self._y_past.value = np.tile(observation, self._T_past)
                self._u_past.value = np.zeros(self._m * self._T_past)
            else:
                self._y_past.value = np.append(self._y_past.value[self._p:], observation)
        self.solve(reference)
        return self.take_action()

    def set_initial_state(self, y_past, u_past):
        self._initial_state_overridden = True
        self._initial_run = False
        self._y_past.value = np.asarray(y_past).copy()
        self._u_past.value = np.asarray(u_past).copy()

    def get_planned_state_trajectory(self):
        if self._plan is not None:
            # Measured past outputs; the fitted past additionally contains slack.
            return np.concatenate((self._plan["y_past"], self._plan["y"])).reshape(-1, self._p)
        past = self._y_past.value
        if past is None:
            return np.zeros((self._T_past + self._T_fut, self._p))
        return np.concatenate((past, np.tile(past[-self._p:], self._T_fut))).reshape(-1, self._p)

    def get_planned_input_trajectory(self):
        if self._plan is not None:
            return np.concatenate((self._plan["u_past"], self._plan["u"])).reshape(-1, self._m)
        past = self._u_past.value
        past = np.zeros(self._T_past * self._m) if past is None else past
        return np.concatenate((past, np.zeros(self._T_fut * self._m))).reshape(-1, self._m)

    def diagnostics(self):
        plan = self._plan
        return {
            "status": self._status, "objective": self._objective,
            "dictionary_columns": self._dictionary_columns,
            "accepted_dictionary_columns": None if plan is None else plan["dictionary_columns"],
            "plan_valid": plan is not None,
            "remaining_plan_actions": 0 if plan is None else max(0, self._T_fut - self._open_loop_index),
            "plan_status": None if plan is None else plan["status"],
            "accepted_objective": None if plan is None else plan["objective"],
            "action_source": self._action_source,
            "executed_action": None if self._executed_action is None else self._executed_action.copy(),
            "predicted_observations": self.get_planned_state_trajectory(),
            "predicted_actions": self.get_planned_input_trajectory(),
            "coefficients": None if plan is None else plan["g"].copy(),
            "past_output_slack": None if plan is None else plan["slack"].copy(),
            "coefficient_representation": self._coefficient_representation if plan is None else plan["coefficient_representation"],
            "accepted_selected_indices": None if plan is None or plan["selected_indices"] is None else plan["selected_indices"].copy(),
        }
