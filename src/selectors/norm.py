"""Rank trajectory columns by the sum of their input and output L-k distances."""

import numpy as np

from ..data import DeePCDims


class LkSelector:
    def __init__(
        self,
        order=1,
        deepc_dims: DeePCDims = None,
        forgetting_factor=1,
        custom_callback=None,
    ):
        self._order = order
        self._custom_callback = custom_callback
        if deepc_dims is not None:
            horizon = deepc_dims.T_past + deepc_dims.T_fut
            self._forgetter_u = np.array(
                [np.ones(deepc_dims.m) * forgetting_factor**i for i in range(horizon)]
            ).reshape(-1, 1)
            self._forgetter_y = np.array(
                [np.ones(deepc_dims.p) * forgetting_factor**i for i in range(horizon)]
            ).reshape(-1, 1)

    def __call__(self, input_traj, state_traj, H_u, H_y, reference):
        adjusted_H_u = H_u - input_traj
        adjusted_H_y = H_y - state_traj
        if self._custom_callback is not None:
            adjusted_H_u, adjusted_H_y = self._custom_callback(adjusted_H_u, adjusted_H_y)
        if hasattr(self, "_forgetter_u"):
            adjusted_H_u *= self._forgetter_u
            adjusted_H_y *= self._forgetter_y
        norms = np.linalg.norm(adjusted_H_u, ord=self._order, axis=0) + np.linalg.norm(
            adjusted_H_y, ord=self._order, axis=0
        )
        return np.argsort(norms), norms

    def get_selector_name(self):
        return f"l{self._order}"

    @staticmethod
    def position_equivariancer(H_u, H_y):
        """Ignore x/y position differences in the six-coordinate rocket state."""
        H_y[::6, :] = 0
        H_y[1::6, :] = 0
        return H_u, H_y
