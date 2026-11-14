"""Select-DPC repository's FIFO DeePC, with consistently post-action columns."""

import numpy as np

from ..interfaces import Policy
from .deepc import DeePC, _applied_action_vector, _reference_vector


class WindowedDeePC(Policy):
    def __init__(self, H_u, H_y, args, solver_options=None, factorization_threshold=None):
        if args.trajectory_data.timing != "post_action":
            raise ValueError("Windowed DeePC requires post_action data; prepare the dataset before building Hankel matrices")
        self._initial_H_u, self._initial_H_y = H_u.copy(), H_y.copy()
        self._args = args
        self._solver_options = solver_options
        self._factorization_threshold = factorization_threshold
        self._dims = args.deepc_dims
        self._m, self._p = self._dims.m, self._dims.p
        self._T_past, self._T_fut = self._dims.T_past, self._dims.T_fut
        self._L = self._T_past + self._T_fut
        self.reset()

    def reset(self, seed=None):
        self._H_u, self._H_y = self._initial_H_u.copy(), self._initial_H_y.copy()
        self._recent_inputs, self._recent_outputs = [], []
        self._pending_action = None
        self._deepc = None
        self._u_init = self._y_init = None

    def act(self, observation, reference):
        _reference_vector(reference, self._p, self._T_fut)  # Validate before changing history.
        if self._deepc is None:
            self._y_init = np.tile(observation, self._T_past)
            self._u_init = np.zeros(self._m * self._T_past)
        else:
            self._y_init = np.append(self._y_init[self._p:], observation)
            self._deepc.begin_step()
        if self._pending_action is not None:
            # u[k-1] is paired with its resulting measurement y[k].
            self._recent_inputs.append(self._pending_action.copy())
            self._recent_outputs.append(np.asarray(observation).copy())
            self._recent_inputs = self._recent_inputs[-self._L:]
            self._recent_outputs = self._recent_outputs[-self._L:]
            if len(self._recent_inputs) == self._L:
                self._H_u = np.column_stack((self._H_u[:, 1:], np.array(self._recent_inputs).reshape(-1)))
                self._H_y = np.column_stack((self._H_y[:, 1:], np.array(self._recent_outputs).reshape(-1)))
        args = self._args
        decompose = (
            self._factorization_threshold is not None
            and self._H_u.shape[1] > self._factorization_threshold
        )
        previous = self._deepc
        self._deepc = DeePC(
            self._H_u, self._H_y, self._dims, args.controller_costs,
            args.controller_constraints, args.input_reference, args.verbose,
            decompose, self._solver_options,
        )
        if previous is not None:
            self._deepc.carry_plan(previous)
        self._deepc.set_initial_state(self._y_init, self._u_init)
        self._deepc.solve(reference)
        action = self._deepc.take_action()
        self._u_init = np.append(self._u_init[self._m:], action)
        self._pending_action = action.copy()
        return action

    def record_applied_action(self, action):
        """Correct the solve history and the input awaiting its next measurement."""
        action = _applied_action_vector(action, self._m)
        if self._deepc is None or not self._deepc.record_applied_action(action):
            return False
        self._u_init[-self._m:] = action
        self._pending_action = action.copy()
        return True

    def get_planned_state_trajectory(self):
        if self._deepc is None:
            return np.zeros((self._L, self._p))
        return self._deepc.get_planned_state_trajectory()

    def get_planned_input_trajectory(self):
        if self._deepc is None:
            return np.zeros((self._L, self._m))
        return self._deepc.get_planned_input_trajectory()

    def diagnostics(self):
        result = self._deepc.diagnostics() if self._deepc is not None else {}
        return {**result, "dictionary_columns": self._H_u.shape[1]}
