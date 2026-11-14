"""BD-mean-χ²: signed local affine imitation on behavior windows."""

import numpy as np

from .behavioral_denoising import BehavioralDenoisingPolicy


class BDMeanChi2Policy(BehavioralDenoisingPolicy):
    """Signed χ² projection with a stabilized covariance solve (ridge=1e-6).

    With zero ridge and full rank, the weights reproduce the query history and
    minimize the squared coefficient norm weighted by the inverse kernel mass.
    Infinite bandwidth over all columns gives global affine least squares.
    The default adds ridge times the identity to stabilize singular or thin
    directions. Set ridge=0 for the exact full-rank projection; positive ridge
    need not reproduce the query history exactly.
    """

    def __init__(self, *args, ridge=1e-6, **kwargs):
        if ridge < 0 or not np.isfinite(ridge):
            raise ValueError("ridge must be finite and nonnegative")
        self._ridge = float(ridge)
        super().__init__(*args, sampling="mean", **kwargs)

    def history_weights(self, differences):
        kernel_weights, _ = super().history_weights(differences)
        mean_difference = kernel_weights @ differences
        centered = differences - mean_difference
        covariance = (centered * kernel_weights[:, None]).T @ centered + self._ridge * np.eye(len(mean_difference))
        slope_target = self._solve_projection(covariance, mean_difference)
        signed_weights = kernel_weights * (1 - centered @ slope_target)
        return signed_weights, {"negative_mass": float(-signed_weights[signed_weights < 0].sum())}

    def _solve_projection(self, covariance, mean_difference):
        try:
            if self._ridge == 0:
                np.linalg.cholesky(covariance)
            return np.linalg.solve(covariance, mean_difference)
        except np.linalg.LinAlgError as error:
            raise ValueError("BD-mean-chi2 requires positive-definite history covariance; "
                             "set ridge>0 explicitly for the regularized approximation") from error

    def _sample(self, kernel_weights, indices):
        return kernel_weights @ self._A[indices]
