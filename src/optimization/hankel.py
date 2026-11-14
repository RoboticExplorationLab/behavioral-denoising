"""Time-major trajectory dictionaries, without windows across episode boundaries."""

import warnings

import numpy as np

from ..data import TrajectoryData, TrajectoryDataSet


class HankelMatrixGenerator:
    def __init__(self, T_past=2, T_fut=10, T_hankel=None):
        self.L = T_past + T_fut
        self.T_hankel = T_hankel

    @staticmethod
    def block_hankel(w, L):
        T, dimension = w.shape
        if L > T:
            raise ValueError(f"L {L} must be smaller than T {T}")
        matrix = np.zeros((L * dimension, T - L + 1))
        flattened = w.reshape(-1)
        for column in range(T - L + 1):
            matrix[:, column] = flattened[
                dimension * column : dimension * (L + column)
            ]
        return matrix

    def generate_hankel_matrices(self, data, subsample_factor=1.0):
        """Build raw array windows; use prepare_dataset first for controller timing."""
        if isinstance(data, TrajectoryDataSet):
            inputs, outputs = [], []
            for trajectory in data.dataset:
                H_u, H_y = self.generate_hankel_matrices(trajectory)
                if H_u is not None:
                    inputs.append(H_u)
                    outputs.append(H_y)
            if not inputs:
                raise ValueError("No trajectory is long enough for the requested horizon")
            H_u, H_y = np.hstack(inputs), np.hstack(outputs)
            indices = np.linspace(
                0, H_u.shape[1], int(H_u.shape[1] * subsample_factor),
                dtype=int, endpoint=False,
            )
            return H_u[:, indices], H_y[:, indices]

        if not isinstance(data, TrajectoryData):
            raise TypeError("Expected TrajectoryData or TrajectoryDataSet")
        if len(data.input_trajectory) < self.L:
            warnings.warn("Trajectory is too short for the requested horizon")
            return None, None
        if self.T_hankel is not None and len(data.input_trajectory) <= self.T_hankel:
            warnings.warn("Trajectory is shorter than the requested Hankel data length")
        # As in the reference, subsampling applies only to a collection of episodes.
        inputs = data.input_trajectory[:self.T_hankel]
        outputs = data.state_trajectory[:self.T_hankel]
        return self.block_hankel(inputs, self.L), self.block_hankel(outputs, self.L)


def generate_multistep_predictor_alt(H_u, H_y, dims):
    """Causal least-squares predictor: [past u, past y, future u] → future y."""
    tp, tf, m, p = dims.T_past, dims.T_fut, dims.m, dims.p
    U_p, U_f = H_u[:tp * m], H_u[tp * m:]
    Y_p, Y_f = H_y[:tp * p], H_y[tp * p:]
    predictor = np.zeros((tf * p, tp * (m + p) + tf * m))
    for step in range(tf):
        # Each predicted output uses only inputs through its own future step.
        regressors = np.vstack((U_p, Y_p, U_f[:(step + 1) * m]))
        predictor[step * p:(step + 1) * p, :regressors.shape[0]] = (
            Y_f[step * p:(step + 1) * p] @ np.linalg.pinv(regressors)
        )
    return predictor
