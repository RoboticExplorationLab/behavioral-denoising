"""BD-χ²: a tracked robust affine policy initialized by behavioral denoising.

A diffusion sample initializes a local affine fit. Iteratively reweighted least
squares gives more weight to windows whose actions follow the fitted line.
The next query carries that line forward, refits it, and selects a new mode only
when the retained kernel mass falls below ``reselect_mass``. A leverage limit
shrinks the returned action toward the weighted mean; the stored line is unlimited.

Covariance solves add ``slope_ridge * I`` (default 1e-6) to stabilize thin or
singular directions. Set ``slope_ridge=0`` for exact weighted least-squares
identities on full-rank weighted histories. Positive ridge changes the history
target and does not guarantee monotonicity of the unpenalized mode mass.
"""

from time import perf_counter

import numpy as np
from scipy.special import logsumexp, softmax

from .behavioral_denoising import BehavioralDenoisingPolicy


class BDChi2Policy(BehavioralDenoisingPolicy):
    def __init__(self, H_u, H_y, dims, *args, mode_width=0.3, iterations=5,
                 slope_ridge=1e-6, reselect_mass=0.05, extrapolation_limit=3.0, **kwargs):
        if not mode_width > 0 or iterations < 1:
            raise ValueError("Require mode_width > 0 and iterations >= 1")
        if not np.isfinite(slope_ridge) or slope_ridge < 0:
            raise ValueError("slope_ridge must be finite and nonnegative")
        if not 0 <= reselect_mass <= 1 or not extrapolation_limit >= 0:
            raise ValueError("Require 0 <= reselect_mass <= 1 and extrapolation_limit >= 0")
        self._mode_width, self._iterations = float(mode_width), int(iterations)
        self._slope_ridge = float(slope_ridge)
        self._reselect_mass = float(reselect_mass)
        self._extrapolation_limit = float(extrapolation_limit)
        super().__init__(H_u, H_y, dims, *args, **kwargs)
        if self._sampling != "ode":
            raise ValueError("BD-chi2 initializes its affine fit with sampling='ode'")
        if self._whitening != "global":
            raise ValueError("BD-chi2 tracking requires global whitening so gains share one metric")

    def reset(self, seed=None):
        super().reset(seed)
        # Keep one latent per episode, including when a mode must be selected again.
        dimension = self._A.shape[1]
        self._latent = self._rng.standard_normal(dimension)
        self._tracked = None  # Previous (query, intercept, gain), before leverage limiting.
        self._selections = 0

    def _mode_sample(self, log_kernel_weights, atoms):
        """Sample an initial action using the fixed episode latent."""
        return self._sample_sigma(log_kernel_weights, atoms, self._latent)

    def _affine_fit(self, weights, differences, actions, limit=np.inf):
        """Fit a_i ≈ intercept + gain d_i, optionally limiting the extrapolation step."""
        mean_difference, mean_action = weights @ differences, weights @ actions
        centered = differences - mean_difference
        weighted = centered * weights[:, None]
        covariance = weighted.T @ centered + self._slope_ridge * np.eye(differences.shape[1])
        try:
            if self._slope_ridge == 0:
                np.linalg.cholesky(covariance)  # The exact projection assumes positive definiteness.
            gain = np.linalg.solve(covariance, weighted.T @ (actions - mean_action)).T
        except np.linalg.LinAlgError as error:
            if self._slope_ridge != 0:
                raise
            raise ValueError(
                "BD-chi2 requires a full-rank positive-definite weighted history covariance "
                "for exact consistency; use independent history coordinates or set slope_ridge > 0 "
                "for a regularized, approximate-consistency variant."
            ) from error
        if np.isfinite(limit):
            distance = np.sqrt(mean_difference @ np.linalg.solve(covariance, mean_difference))
            allowed_distance = limit / np.sqrt(np.sum(weights**2))  # r sqrt(n_eff).
            mean_difference = mean_difference * min(1.0, allowed_distance / max(distance, 1e-12))
        return mean_action - gain @ mean_difference, gain

    def _refit(self, log_kernel_weights, differences, actions, intercept, gain):
        """Run fixed-count IRLS and return the line, last-fit weights, and final mass."""
        for _ in range(self._iterations):
            residuals = actions - intercept - differences @ gain.T
            log_fit = -np.sum(residuals**2, axis=1) / (2 * self._mode_width**2)
            weights = softmax(log_kernel_weights + log_fit)
            intercept, gain = self._affine_fit(weights, differences, actions)
        residuals = actions - intercept - differences @ gain.T
        log_fit = -np.sum(residuals**2, axis=1) / (2 * self._mode_width**2)
        mass = float(np.exp(logsumexp(log_kernel_weights + log_fit)))
        return intercept, gain, weights, mass

    def _select_mode(self, log_kernel_weights, differences, actions):
        """Initialize a flat line through a diffusion sample, then refit it."""
        self._selections += 1
        intercept = self._mode_sample(log_kernel_weights, actions)
        gain = np.zeros((actions.shape[1], differences.shape[1]))
        return self._refit(log_kernel_weights, differences, actions, intercept, gain)

    def plan(self, history):
        started = perf_counter()
        query = (history - self._z_mean) / self._z_scale
        indices, differences = self._neighborhood(history)
        log_kernel_weights = self.history_log_weights(differences)
        support = np.isfinite(log_kernel_weights)
        log_kernel_weights = log_kernel_weights[support] - logsumexp(log_kernel_weights[support])
        differences, actions = differences[support], self._A[indices[support]]

        fitted = None
        if self._tracked is not None:
            previous_query, intercept, gain = self._tracked
            carried_intercept = intercept + gain @ (query - previous_query)
            fitted = self._refit(log_kernel_weights, differences, actions, carried_intercept, gain)
            if not fitted[3] >= self._reselect_mass:  # A failed/nonfinite mass also triggers reselection.
                fitted = None
        if fitted is None:
            fitted = self._select_mode(log_kernel_weights, differences, actions)
        intercept, gain, weights, mass = fitted
        self._tracked = (query, intercept, gain)
        if np.isfinite(self._extrapolation_limit):
            intercept = self._affine_fit(weights, differences, actions, self._extrapolation_limit)[0]
        self._last = {
            "effective_support": float(1 / np.sum(weights**2)),
            "mode_mass": mass,
            "mode_selections": self._selections,
            "plan_seconds": perf_counter() - started,
        }
        return (intercept * self._a_scale + self._a_mean).reshape(self._action_horizon, self._m)
