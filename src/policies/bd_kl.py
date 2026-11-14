"""BD-KL: project every action-conditioned denoiser onto the history constraint.

The action-conditioned KL reference changes at every sigma-Euler step. The hard constraint is solved on its feasible face; redundant history
coordinates are removed by an SVD. Infeasible histories and failed solves raise
an error instead of returning an unconstrained action. Constraints and solver
tolerances are measured in the policy's standardized history coordinates.
"""

from time import perf_counter

import numpy as np
from scipy.optimize import brentq, linprog, minimize
from scipy.special import logsumexp, softmax
from scipy.sparse import eye, hstack

from .behavioral_denoising import BehavioralDenoisingPolicy


class HistoryConstraintError(ValueError):
    """The selected dictionary cannot support the requested hard history constraint."""


class KLHistoryProjection:
    """KL projection of log weights subject to sum(weights * differences) = 0.

    The feasible support is determined once and reused at all denoising levels.
    Linear programs identify boundary faces. A convex dual Newton solve then
    operates only on independent history coordinates of the feasible face.
    """

    def __init__(self, differences, support=None, tolerance=1e-9, max_iterations=100):
        self.differences = np.asarray(differences, dtype=float)
        if self.differences.ndim != 2 or not np.all(np.isfinite(self.differences)):
            raise ValueError("History differences must be a finite matrix")
        if tolerance <= 0 or max_iterations < 1:
            raise ValueError("Projection tolerance and iteration limit must be positive")
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        if support is None:
            support = np.ones(len(self.differences), dtype=bool)
        self.indices = np.flatnonzero(support)
        if not len(self.indices):
            raise HistoryConstraintError("BD-KL has no dictionary atoms with positive prior")
        active = self.differences[self.indices]
        coordinates = self._independent_coordinates(active)
        self.indices = self.indices[self._feasible_support(coordinates)]
        self.coordinates = self._independent_coordinates(self.differences[self.indices])
        self.last_weights = None
        self.last_iterations = 0

    @staticmethod
    def _independent_coordinates(differences):
        _, singular_values, right = np.linalg.svd(differences, full_matrices=False)
        if len(singular_values) == 0:
            return np.empty((len(differences), 0))
        threshold = np.finfo(float).eps * max(differences.shape) * singular_values[0]
        rank = np.count_nonzero(singular_values > threshold)
        # Form coordinates from the original rows so an exactly zero history
        # difference stays zero instead of acquiring SVD left-vector roundoff.
        return (differences @ right[:rank].T) / singular_values[:rank]

    def _feasible_support(self, coordinates):
        count = len(coordinates)
        # HiGHS drops matrix coefficients below its absolute threshold. Scale
        # the zero-moment rows so a near-boundary atom with tiny required mass
        # is not mistaken for an exactly infeasible/absent face direction.
        equality = np.vstack((np.ones(count), 1e8 * coordinates.T))
        target = np.r_[1.0, np.zeros(coordinates.shape[1])]
        # Maximizing the smallest weight quickly recognizes an interior query.
        inequality = hstack((-eye(count, format="csc"), np.ones((count, 1)) / count), format="csc")
        interior = linprog(
            np.r_[np.zeros(count), -1.0], A_ub=inequality, b_ub=np.zeros(count),
            A_eq=np.column_stack((equality, np.zeros(len(target)))), b_eq=target,
            bounds=[(0, None)] * (count + 1), method="highs",
        )
        if not interior.success:
            raise HistoryConstraintError("BD-KL history is outside the convex hull of its supported dictionary")
        if interior.x[-1] > 1e-8:
            return np.arange(count)
        # On a boundary, collect the union of all feasible positive supports.
        # Each LP either adds a feasible atom or certifies that none remain.
        possible = interior.x[:-1] > 0
        for _ in range(count + 1):
            if np.all(possible):
                break
            result = linprog(
                -np.logical_not(possible).astype(float), A_eq=equality, b_eq=target,
                bounds=(0, None), method="highs",
            )
            if not result.success:
                raise HistoryConstraintError("BD-KL failed to determine the feasible history face")
            added = (~possible) & (result.x > 0)
            if not np.any(added):
                break
            possible |= added
        return np.flatnonzero(possible)

    def project(self, log_weights):
        log_weights = np.asarray(log_weights, dtype=float)
        if log_weights.shape != (len(self.differences),):
            raise ValueError("One log weight is required for each dictionary atom")
        selected = log_weights[self.indices]
        if not np.all(np.isfinite(selected)):
            raise HistoryConstraintError("The denoiser lost prior support on the feasible history face")
        selected = selected - np.max(selected)
        features = self.coordinates
        # Remove the large linear component before Newton iterations. This tilt
        # leaves the constrained KL optimum unchanged and improves conditioning.
        multiplier = -features.T @ (selected - selected.mean())
        if features.shape[1]:
            def objective(value):
                tilted = selected + features @ value
                return logsumexp(tilted), softmax(tilted) @ features

            # BFGS crosses flat regions when a narrow action posterior has a
            # numerically singular covariance. Newton then polishes its moment.
            initial = minimize(objective, multiplier, jac=True, method="BFGS",
                               options={"gtol": self.tolerance * 0.1,
                                        "maxiter": self.max_iterations})
            multiplier = initial.x
        weights = softmax(selected + features @ multiplier)
        converged = False
        for iteration in range(self.max_iterations + 1):
            gradient = weights @ features
            residual = weights @ self.differences[self.indices]
            if (np.linalg.norm(gradient) <= self.tolerance
                    and np.linalg.norm(residual) <= self.tolerance):
                converged = True
                break
            if iteration == self.max_iterations:
                break
            centered = features - gradient
            hessian = (centered * weights[:, None]).T @ centered
            try:
                direction = -np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                direction = -gradient
            if (not np.all(np.isfinite(direction)) or gradient @ direction >= 0
                    or np.linalg.norm(direction) > 1e12 * max(1.0, np.linalg.norm(gradient))):
                direction = -gradient
            # The one-dimensional derivative is increasing. Bracketing its zero
            # also handles near-single-atom posteriors whose Hessian underflows.
            base = selected + features @ multiplier
            change = features @ direction
            derivative = lambda step: float(softmax(base + step * change) @ change)
            upper = 1.0
            for _ in range(100):
                if derivative(upper) >= 0:
                    break
                upper *= 2.0
            else:
                raise RuntimeError("BD-KL dual line search failed to bracket a finite solution")
            step = brentq(derivative, 0.0, upper, xtol=1e-12)
            multiplier = multiplier + step * direction
            weights = softmax(selected + features @ multiplier)
        if not converged:
            raise RuntimeError(
                f"BD-KL dual did not converge in {self.max_iterations} iterations "
                f"(history residual {np.linalg.norm(residual):.3g})"
            )
        result = np.zeros(len(self.differences))
        result[self.indices] = weights
        self.last_weights = result
        self.last_iterations = iteration
        return result


class BDKLPolicy(BehavioralDenoisingPolicy):
    """Hard history-consistent BD denoising at every sigma-Euler step.

    ``sampling='mean'`` computes the infinite-noise (history-prior) KL limit.
    A finite noisy sample uses the full action-conditioned posterior at each
    step, including the final denoiser. No infeasibility fallback is implicit.
    """

    def __init__(self, *args, dual_tolerance=1e-9, dual_iterations=100, **kwargs):
        if kwargs.get("sampling_scheme", "sigma") != "sigma":
            raise ValueError("BD-KL requires sampling_scheme='sigma'")
        self._dual_tolerance = float(dual_tolerance)
        self._dual_iterations = int(dual_iterations)
        super().__init__(*args, **kwargs)

    def _sigma_weights(self, noisy, noise_level, log_kernel_weights, atoms):
        squared = np.sum((atoms - noisy) ** 2, axis=1)
        return self._projection.project(log_kernel_weights - squared / (2 * noise_level**2))

    def plan(self, history):
        start = perf_counter()
        indices, differences = self._neighborhood(history)
        log_weights = self.history_log_weights(differences)
        self._projection = KLHistoryProjection(
            differences, support=np.isfinite(log_weights),
            tolerance=self._dual_tolerance, max_iterations=self._dual_iterations,
        )
        atoms = self._A[indices]
        if self._sampling == "mean":
            final_weights = self._projection.project(log_weights)
            standardized = final_weights @ atoms
        else:
            latent = self._rng.standard_normal(atoms.shape[1])
            standardized = self._sample_sigma(log_weights, atoms, latent)
            final_weights = self._last_denoising_weights
        residual = float(np.linalg.norm(final_weights @ differences))
        if residual > self._dual_tolerance:
            raise RuntimeError(f"BD-KL final denoiser violates its history constraint ({residual:.3g})")
        self._last = {
            "dual_converged": True,
            "dual_iterations": self._projection.last_iterations,
            "consistency_residual": residual,
            "effective_support": float(1 / np.sum(final_weights**2)),
            "plan_seconds": perf_counter() - start,
        }
        chunk = standardized * self._a_scale + self._a_mean
        return chunk.reshape(self._action_horizon, self._m)
