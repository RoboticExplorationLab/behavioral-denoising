"""BD-mean and BD-sample on a dictionary of standardized behavior windows.

BD-mean averages actions with history Gibbs weights. BD-sample uses the
finite sigma-Euler algorithm: start at sigma_max times a Gaussian latent and
integrate the Gaussian-mixture denoiser along decreasing noise levels to zero.
The default schedule has 100 geometric positive levels from 10 to 0.01 followed
by zero. This is a specified finite approximation, not an exact atom sampler.

Histories and actions use global coordinate standardization. Optional nearest
windows restrict the dictionary. The default dictionary prior is uniform.
"""

from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import softmax

from ..data import prepare_dataset
from ..interfaces import Policy
from ..optimization.hankel import HankelMatrixGenerator


def _standardize(columns):
    mean = columns.mean(axis=0)
    scale = columns.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return mean, scale


class BehavioralDenoisingPolicy(Policy):
    def __init__(
        self, H_u, H_y, dims, bandwidth=0.5, action_horizon=1, execute_steps=1,
        sampling="ode", diffusion_steps=100, smoothing=0.0, smoothing_samples=8,
        sampling_scheme="sigma", noise_levels=None, sigma_max=10.0, sigma_min=0.01,
        prior_weights=None,
        neighbors=None, whitening="global", covariance_ridge=1e-3,
        include_past_inputs=True, seed=0,
    ):
        if not bandwidth > 0 or np.isnan(bandwidth):
            raise ValueError("bandwidth must be positive (infinity is allowed)")
        if sampling_scheme != "sigma":
            raise ValueError("The paper implementation requires sampling_scheme='sigma'")
        if int(diffusion_steps) < 1:
            raise ValueError("diffusion_steps must be positive")
        if smoothing != 0:
            raise ValueError("The paper sampler uses smoothing=0")
        if noise_levels is None:
            if not 0 < sigma_min <= sigma_max or not np.isfinite(sigma_max):
                raise ValueError("Require finite 0 < sigma_min <= sigma_max")
            noise_levels = np.r_[np.geomspace(sigma_max, sigma_min, int(diffusion_steps)), 0.0]
        noise_levels = np.asarray(noise_levels, dtype=float)
        if (noise_levels.ndim != 1 or len(noise_levels) < 2
                or not np.all(np.isfinite(noise_levels)) or noise_levels[-1] != 0
                or np.any(noise_levels[:-1] <= 0) or np.any(np.diff(noise_levels) >= 0)):
            raise ValueError("noise_levels must strictly decrease from a positive value to zero")
        self._sampling_scheme = sampling_scheme
        self._noise_levels = noise_levels.copy()
        if sampling not in ("ode", "mean"):
            raise ValueError("sampling must be ode or mean")
        if whitening != "global":
            raise ValueError("The paper implementation uses global standardization")
        if not 1 <= execute_steps <= action_horizon <= dims.T_fut:
            raise ValueError("Require 1 <= execute_steps <= action_horizon <= T_fut")
        self._dims = dims
        self._m, self._p, self._T_past = dims.m, dims.p, dims.T_past
        past_inputs, future_inputs = H_u[:self._T_past * self._m], H_u[self._T_past * self._m:]
        past_outputs = H_y[:self._T_past * self._p]
        self._include_past_inputs = include_past_inputs
        histories = (np.vstack((past_inputs, past_outputs)) if include_past_inputs else past_outputs).T
        actions = future_inputs[:action_horizon * self._m].T
        self._z_mean, self._z_scale = _standardize(histories)
        self._a_mean, self._a_scale = _standardize(actions)
        self._prior_weights = None
        if prior_weights is not None:
            prior_weights = np.asarray(prior_weights, dtype=float)
            if (prior_weights.shape != (len(histories),)
                    or np.any(~np.isfinite(prior_weights)) or np.any(prior_weights < 0)
                    or not np.any(prior_weights > 0)):
                raise ValueError("prior_weights must be nonnegative finite weights, one per window")
            scaled_prior = prior_weights / prior_weights.max()
            self._prior_weights = scaled_prior / scaled_prior.sum()
        self._selected_prior_weights = self._prior_weights
        self._Z = (histories - self._z_mean) / self._z_scale
        self._A = (actions - self._a_mean) / self._a_scale
        self._tree = None if neighbors is None else cKDTree(self._Z)
        self._neighbors = None if neighbors is None else min(int(neighbors), len(self._Z))
        self._bandwidth = float(bandwidth)
        self._action_horizon, self._execute_steps = action_horizon, execute_steps
        self._sampling, self._diffusion_steps = sampling, int(diffusion_steps)
        self._smoothing, self._smoothing_samples = float(smoothing), int(smoothing_samples)
        self._whitening, self._covariance_ridge = whitening, float(covariance_ridge)
        self._seed = seed
        self.reset()

    @classmethod
    def create_from_data(cls, trajectory_data, dims, **options):
        H_u, H_y = HankelMatrixGenerator(dims.T_past, dims.T_fut).generate_hankel_matrices(
            prepare_dataset(trajectory_data))
        return cls(H_u, H_y, dims, **options)

    def reset(self, seed=None):
        self._rng = np.random.default_rng(self._seed if seed is None else seed)
        self._u_past = self._y_past = None
        self._chunk, self._chunk_index = None, 0
        self._last = {}

    # History stage: neighborhood, metric, and weights q(z).
    def _neighborhood(self, history):
        """Indices of candidate windows and their differences z_i - z in the kernel metric."""
        query = (history - self._z_mean) / self._z_scale
        if self._tree is None:
            indices = np.arange(len(self._Z))
        else:
            indices = np.atleast_1d(self._tree.query(query, k=self._neighbors)[1])
        if self._prior_weights is not None:
            self._selected_prior_weights = self._prior_weights[indices]
        differences = self._Z[indices] - query
        return indices, differences

    def history_log_weights(self, differences):
        """Unnormalized log Gibbs weights, retaining tiny but positive prior mass."""
        log_weights = -np.sum(differences**2, axis=1) / (2 * self._bandwidth**2)
        if self._selected_prior_weights is not None:
            prior = self._selected_prior_weights
            if len(prior) != len(differences) or not np.any(prior > 0):
                raise ValueError("The selected neighborhood has no supported prior mass")
            log_prior = np.full_like(prior, -np.inf)
            np.log(prior, out=log_prior, where=prior > 0)
            log_weights = log_weights + log_prior
        return log_weights

    def history_weights(self, differences):
        """Gibbs weights over windows; returns (q, diagnostics)."""
        return softmax(self.history_log_weights(differences)), {}

    # Action stage: Gaussian-mixture denoising along decreasing sigma.
    def _sigma_weights(self, noisy, noise_level, log_kernel_weights, atoms):
        squared = np.sum((atoms - noisy) ** 2, axis=1)
        return softmax(log_kernel_weights - squared / (2 * noise_level**2))

    def _denoise_sigma(self, noisy, noise_level, log_kernel_weights, atoms):
        weights = self._sigma_weights(noisy, noise_level, log_kernel_weights, atoms)
        self._last_denoising_weights = weights
        return weights @ atoms

    def _sample_sigma(self, log_kernel_weights, atoms, latent):
        """Finite sigma-Euler sampler; ``latent`` is a supplied standard Gaussian draw."""
        noisy = self._noise_levels[0] * np.asarray(latent, dtype=float)
        for noise_level, next_noise in zip(self._noise_levels[:-1], self._noise_levels[1:]):
            denoised = self._denoise_sigma(noisy, noise_level, log_kernel_weights, atoms)
            if next_noise == 0:
                noisy = denoised
            else:
                noisy = noisy + (next_noise - noise_level) * (noisy - denoised) / noise_level
        return noisy

    def _sample(self, kernel_weights, indices, log_kernel_weights=None):
        atoms = self._A[indices]
        if self._sampling == "mean":
            return kernel_weights @ atoms
        if log_kernel_weights is None:
            log_kernel_weights = np.full_like(kernel_weights, -np.inf)
            np.log(kernel_weights, out=log_kernel_weights, where=kernel_weights > 0)
        latent = self._rng.standard_normal(atoms.shape[1])
        return self._sample_sigma(log_kernel_weights, atoms, latent)

    def plan(self, history):
        """Return an action chunk (action_horizon, m) for the history vector z."""
        start = perf_counter()
        indices, differences = self._neighborhood(history)
        kernel_weights, weight_info = self.history_weights(differences)
        if self._sampling == "ode":
            # Preserve tiny prior mass until action conditioning: a window with
            # negligible history weight can regain mass near its action.
            log_weights = self.history_log_weights(differences)
            standardized = self._sample(kernel_weights, indices, log_kernel_weights=log_weights)
        else:
            standardized = self._sample(kernel_weights, indices)
        chunk = standardized * self._a_scale + self._a_mean
        self._last = {
            **weight_info,
            "effective_support": float(1 / np.sum(kernel_weights**2)),
            "consistency_residual": float(np.linalg.norm(kernel_weights @ differences)),
            "mean_action": ((kernel_weights @ self._A[indices])[:self._m]
                            * self._a_scale[:self._m] + self._a_mean[:self._m]),
            "plan_seconds": perf_counter() - start,
        }
        return chunk.reshape(self._action_horizon, self._m)

    def history(self):
        if self._y_past is None:
            raise RuntimeError("No history before the first observation")
        return np.r_[self._u_past, self._y_past] if self._include_past_inputs else self._y_past.copy()

    def act(self, observation, reference=None):
        """Imitate the demonstrations; ``reference`` is fixed by the data and unused."""
        observation = np.asarray(observation, dtype=float)
        if self._y_past is None:
            # Same initialization as DeePC: repeated observation, zero past inputs.
            self._y_past = np.tile(observation, self._T_past)
            self._u_past = np.zeros(self._m * self._T_past)
        else:
            self._y_past = np.append(self._y_past[self._p:], observation)
        if self._chunk is None or self._chunk_index >= self._execute_steps:
            self._chunk, self._chunk_index = self.plan(self.history()), 0
        action = self._chunk[self._chunk_index].copy()
        self._chunk_index += 1
        self._u_past = np.append(self._u_past[self._m:], action)
        return action

    def record_applied_action(self, action):
        """Record the actual executed action after any external clipping or filtering."""
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape != (self._m,) or np.any(~np.isfinite(action)):
            raise ValueError("Executed action must be a finite vector of input dimension")
        if self._u_past is not None:
            self._u_past[-self._m:] = action

    def diagnostics(self):
        return {**self._last, "dictionary_columns": len(self._Z),
                "neighbors": self._neighbors or len(self._Z)}
