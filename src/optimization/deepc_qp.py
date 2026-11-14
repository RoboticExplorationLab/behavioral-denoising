"""The optimization problem shared by the three DeePC policies."""

import cvxpy as cp
import numpy as np


# Update rho after a fixed number of iterations rather than using OSQP's
# timing-based schedule, so machine load does not change the update interval.
OSQP_ADAPTIVE_RHO_INTERVAL = 25
DENSE_PROJECTION_MAX_COLUMNS = 2000


def _row_space_basis(matrix):
    """Orthonormal rows spanning the subspace retained by NumPy's default pinv."""
    _, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=False)
    # np.linalg.pinv uses rcond=1e-15 and retains singular values strictly above
    # that relative cutoff. Do not use matrix_rank's dimension-dependent cutoff.
    retained = singular_values > 1e-15 * singular_values[0]
    return right_vectors[retained]

DEFAULT_SOLVER_OPTIONS = {
    "solver": "OSQP",
    "warm_start": False,
    "max_iter": 6000,
    "eps_abs": 1e-5,
    "eps_rel": 1e-4,
    "adaptive_rho_interval": OSQP_ADAPTIVE_RHO_INTERVAL,
}


class DeePCQP:
    def __init__(
        self, H_u, H_y, dims, costs, constraints, input_reference,
        decompose=False, verbose=False, solver_options=None,
    ):
        requested = dict(solver_options or {})
        if isinstance(requested.get("solver"), str):
            requested["solver"] = requested["solver"].upper()
        # Solver-specific options must not leak into another solver's API.
        defaults = (DEFAULT_SOLVER_OPTIONS if requested.get("solver", "OSQP") == "OSQP"
                    else {"warm_start": False})
        self.options = {**defaults, "verbose": verbose, **requested}
        m, p, tp, tf = dims.m, dims.p, dims.T_past, dims.T_fut
        if decompose:
            # Express coefficients, constraints, and regularization in the
            # Cholesky-factor coordinates.
            H = np.vstack((H_u, H_y))
            jitter = 1e-10
            while True:
                try:
                    factor = np.linalg.cholesky(H @ H.T + jitter * np.eye(H.shape[0]))
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10
            H_u, H_y = factor[:m * (tp + tf)], factor[m * (tp + tf):]

        U_p, U_f = H_u[:m * tp], H_u[m * tp:]
        Y_p, Y_f = H_y[:p * tp], H_y[p * tp:]
        self.g = cp.Variable(U_f.shape[1])
        self.u = cp.Variable(tf * m)
        self.y = cp.Variable(tf * p)
        self.u_past = cp.Parameter(tp * m)
        self.y_past = cp.Parameter(tp * p)
        self.reference = cp.Parameter(tf * p)
        self.y_slack = cp.Variable(tp * p)

        restrictions = [
            U_p @ self.g == self.u_past,
            U_f @ self.g == self.u,
            Y_p @ self.g == self.y_past + self.y_slack,
            Y_f @ self.g == self.y,
            cp.sum(self.g) == 1,
        ]
        objective = costs.slack_cost * cp.norm1(self.y_slack)
        objective += cp.quad_form(
            self.y - self.reference, np.kron(np.eye(tf), costs.Q)
        )
        objective += cp.quad_form(
            self.u - np.tile(input_reference, tf), np.kron(np.eye(tf), costs.R)
        )
        if costs.regularizer_cost_g_1 != 0:
            objective += costs.regularizer_cost_g_1 * cp.norm(self.g, 1)
        if costs.regularizer_cost_g_pi != 0:
            Z = np.vstack((U_p, U_f, Y_p, np.ones((1, Y_p.shape[1]))))
            if Z.shape[1] > DENSE_PROJECTION_MAX_COLUMNS:
                # P = Q.T @ Q. Keep the low-dimensional coordinates explicit:
                # expanding P would allocate a dictionary-size square matrix.
                row_basis = _row_space_basis(Z)
                coordinates = cp.Variable(row_basis.shape[0])
                restrictions.append(coordinates == row_basis @ self.g)
                residual = self.g - row_basis.T @ coordinates
            else:
                # Small dictionaries can use the full projection matrix directly.
                projection = np.linalg.pinv(Z) @ Z
                residual = (np.eye(projection.shape[0]) - projection) @ self.g
            objective += costs.regularizer_cost_g_pi * cp.sum_squares(residual)
        for step in range(tf):
            if constraints.A_u is not None:
                restrictions.append(
                    constraints.A_u @ self.u[step * m:(step + 1) * m] <= constraints.b_u
                )
            if constraints.A_y is not None:
                restrictions.append(
                    constraints.A_y @ self.y[step * p:(step + 1) * p] <= constraints.b_y
                )
        self.problem = cp.Problem(cp.Minimize(objective), restrictions)

    def solve(self):
        self.problem.solve(**self.options)
        return self.problem.status
