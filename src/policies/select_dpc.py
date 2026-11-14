"""Select-DPC: choose nearby trajectories and solve a local DeePC problem."""

from time import perf_counter

import numpy as np

from ..data import prepare_dataset
from ..interfaces import Policy
from ..optimization.hankel import HankelMatrixGenerator
from ..selectors import LkSelector
from .deepc import DeePC, _applied_action_vector, _reference_vector


class SelectDPC(Policy):
    def __init__(
        self, args, selector=None, num_hankel_cols=180, n_iter=1,
        subsample_factor=1.0, solver_options=None,
        factorization_threshold=None, hankel_matrices=None,
    ):
        if n_iter < 1:
            raise ValueError("n_iter must be at least one")
        self._args = args
        self._dims = args.deepc_dims
        self._T_past, self._T_fut = self._dims.T_past, self._dims.T_fut
        self._m, self._p = self._dims.m, self._dims.p
        if hankel_matrices is None:
            self._H_u, self._H_y = HankelMatrixGenerator(
                self._T_past, self._T_fut,
            ).generate_hankel_matrices(prepare_dataset(args.trajectory_data), subsample_factor)
        else:
            if args.trajectory_data.timing != "post_action":
                raise ValueError("Supplied Hankel matrices require post_action data; prepare the dataset before building them")
            if subsample_factor != 1.0:
                raise ValueError("Subsample the supplied Hankel matrices before constructing SelectDPC")
            if len(hankel_matrices) != 2:
                raise ValueError("hankel_matrices must contain (H_u, H_y)")
            # Use the processed dictionary directly, preserving its column order.
            self._H_u = np.asarray(hankel_matrices[0])
            self._H_y = np.asarray(hankel_matrices[1])
            length = self._T_past + self._T_fut
            if (self._H_u.ndim != 2 or self._H_y.ndim != 2
                    or self._H_u.shape[0] != length * self._m
                    or self._H_y.shape[0] != length * self._p
                    or self._H_u.shape[1] != self._H_y.shape[1]
                    or self._H_u.shape[1] == 0):
                raise ValueError("Hankel shapes must be (L*m, columns) and (L*p, columns), with columns > 0")
            if not np.isfinite(self._H_u).all() or not np.isfinite(self._H_y).all():
                raise ValueError("Supplied Hankel matrices must be finite")
        self._num_hankel_cols = (
            self._H_u.shape[1] if num_hankel_cols == -1 else num_hankel_cols
        )
        if self._num_hankel_cols < 1:
            raise ValueError("num_hankel_cols must be positive or -1 for all columns")
        # Rank nearby trajectories by L1 distance unless a selector is supplied.
        self._selector_callback = selector if selector is not None else LkSelector(
            order=1, deepc_dims=self._dims
        )
        self._n_iter = n_iter
        self._solver_options = solver_options
        self._factorization_threshold = factorization_threshold
        self.reset()

    def reset(self, seed=None):
        self._deepc = None
        self._y_past = self._u_past = None
        self._selected_indices = None
        self._selection_seconds = self._solve_seconds = 0.0
        self._solve_count = 0
        if hasattr(self._selector_callback, "reset"):
            self._selector_callback.reset(seed)

    def act(self, observation, reference):
        _reference_vector(reference, self._p, self._T_fut)  # Validate before changing history.
        if self._deepc is None:
            self._y_past = np.tile(observation, self._T_past)
            self._u_past = np.zeros(self._m * self._T_past)
        else:
            self._y_past = np.append(self._y_past[self._p:], observation)
            self._deepc.begin_step()
        for iteration in range(self._n_iter):
            if self._deepc is None or self._deepc._plan is None:
                # Before any accepted solve, retain known measurements in the query.
                state_traj = np.append(self._y_past, np.tile(observation, self._T_fut)).reshape(-1, 1)
                input_traj = np.append(self._u_past, np.zeros(self._m * self._T_fut)).reshape(-1, 1)
            else:
                state_traj = self._deepc.get_planned_state_trajectory().reshape(-1, 1)
                input_traj = self._deepc.get_planned_input_trajectory().reshape(-1, 1)

            start = perf_counter()
            indices, _ = self._selector_callback(
                input_traj, state_traj, self._H_u, self._H_y, reference,
            )
            self._selection_seconds += perf_counter() - start
            indices = indices[:self._num_hankel_cols]
            self._selected_indices = indices.copy()
            previous = self._deepc
            decompose = (
                self._factorization_threshold is not None
                and indices.size > self._factorization_threshold
            )
            args = self._args
            self._deepc = DeePC(
                self._H_u[:, indices], self._H_y[:, indices], self._dims,
                args.controller_costs, args.controller_constraints,
                args.input_reference, args.verbose, decompose, self._solver_options,
            )
            if previous is not None:
                self._deepc.carry_plan(previous)
            self._deepc.set_initial_state(self._y_past, self._u_past)
            start = perf_counter()
            self._deepc.solve(reference, selected_indices=indices)
            self._solve_seconds += perf_counter() - start
            self._solve_count += 1

        action = self._deepc.take_action()
        self._u_past = np.append(self._u_past[self._m:], action)
        return action

    def record_applied_action(self, action):
        """Keep wrapper and local-solver histories aligned with the executed input."""
        action = _applied_action_vector(action, self._m)
        if self._deepc is None or not self._deepc.record_applied_action(action):
            return False
        self._u_past[-self._m:] = action
        return True

    def get_planned_state_trajectory(self):
        if self._deepc is None:
            return np.zeros((self._T_past + self._T_fut, self._p))
        return self._deepc.get_planned_state_trajectory()

    def get_planned_input_trajectory(self):
        if self._deepc is None:
            return np.zeros((self._T_past + self._T_fut, self._m))
        return self._deepc.get_planned_input_trajectory()

    def diagnostics(self):
        result = self._deepc.diagnostics() if self._deepc is not None else {}
        return {
            **result,
            "available_dictionary_columns": self._H_u.shape[1],
            "selector_cache": getattr(self._selector_callback, "cache_info", None),
            "selected_indices": (
                None if self._selected_indices is None else self._selected_indices.copy()
            ),
            "selection_seconds": self._selection_seconds / max(1, self._solve_count),
            "solve_seconds": self._solve_seconds / max(1, self._solve_count),
        }
