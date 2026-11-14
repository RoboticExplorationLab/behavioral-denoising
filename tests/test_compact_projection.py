"""The large-dictionary projection keeps the same DeePC objective and solution."""

import cvxpy as cp
import numpy as np
import pytest

from src.data import DeePCConstraints, DeePCCost, DeePCDims
from src.optimization import deepc_qp


def dictionary(rank_deficient):
    rng = np.random.default_rng(681)
    inputs = rng.normal(size=(4, 28))
    outputs = rng.normal(size=(4, 28))
    if rank_deficient:
        outputs[0] = inputs[0] + 2 * inputs[1]
        outputs[1] = 3 * outputs[0]
    return inputs, outputs


@pytest.mark.parametrize('rank_deficient', [False, True])
def test_compact_projection_has_same_penalty_for_arbitrary_coefficients(rank_deficient):
    inputs, outputs = dictionary(rank_deficient)
    matrix = np.vstack((inputs, outputs[:2], np.ones((1, inputs.shape[1]))))
    basis = deepc_qp._row_space_basis(matrix)
    coefficients = np.random.default_rng(202).normal(size=(matrix.shape[1], 7))
    dense_residual = (np.eye(matrix.shape[1]) - np.linalg.pinv(matrix) @ matrix) @ coefficients
    compact_residual = coefficients - basis.T @ (basis @ coefficients)
    np.testing.assert_allclose(basis @ basis.T, np.eye(len(basis)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(compact_residual, dense_residual, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(np.sum(compact_residual**2, axis=0),
                               np.sum(dense_residual**2, axis=0), rtol=1e-12, atol=1e-12)


def test_basis_uses_pinv_cutoff_not_matrix_rank_default():
    matrix = np.zeros((3, 100))
    matrix[0, 0], matrix[1, 1], matrix[2, 2] = 1., 2e-15, 5e-16
    basis = deepc_qp._row_space_basis(matrix)
    assert basis.shape == (2, 100)
    np.testing.assert_allclose(basis.T @ basis, np.linalg.pinv(matrix) @ matrix, atol=0, rtol=0)


def build_problem(inputs, outputs):
    dims = DeePCDims(T_past=2, T_fut=2, p=1, m=1)
    costs = DeePCCost(np.array([[2.]]), np.array([[.4]]), 5., .03, .7)
    constraints = DeePCConstraints(np.array([[1.], [-1.]]), np.ones(2),
                                   np.array([[1.], [-1.]]), 2 * np.ones(2))
    problem = deepc_qp.DeePCQP(inputs, outputs, dims, costs, constraints, np.array([.1]),
                              solver_options={'eps_abs': 1e-9, 'eps_rel': 1e-9, 'max_iter': 30000,
                                              'polish': True})
    # A known feasible history; soft output-history fitting remains active.
    coefficients = np.ones(inputs.shape[1]) / inputs.shape[1]
    problem.u_past.value = inputs[:2] @ coefficients
    problem.y_past.value = outputs[:2] @ coefficients + np.array([.1, -.1])
    problem.reference.value = np.array([.6, -.3])
    return problem


@pytest.mark.parametrize('rank_deficient', [False, True])
def test_compact_and_dense_problems_have_same_status_objective_and_actions(monkeypatch, rank_deficient):
    inputs, outputs = dictionary(rank_deficient)
    monkeypatch.setattr(deepc_qp, 'DENSE_PROJECTION_MAX_COLUMNS', 1000)
    dense = build_problem(inputs, outputs)
    monkeypatch.setattr(deepc_qp, 'DENSE_PROJECTION_MAX_COLUMNS', 0)
    compact = build_problem(inputs, outputs)
    assert dense.solve() == compact.solve() == cp.OPTIMAL
    np.testing.assert_allclose(compact.problem.value, dense.problem.value, atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(compact.u.value, dense.u.value, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(compact.y.value, dense.y.value, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(compact.g.value, dense.g.value, atol=3e-6, rtol=3e-6)


def test_large_dictionary_never_constructs_dense_projection(monkeypatch):
    rng = np.random.default_rng(9)
    columns = deepc_qp.DENSE_PROJECTION_MAX_COLUMNS + 1
    inputs = rng.normal(size=(4, columns))
    outputs = rng.normal(size=(4, columns))
    monkeypatch.setattr(np.linalg, 'pinv', lambda *args, **kwargs: pytest.fail('dense projector requested'))
    problem = build_problem(inputs, outputs)
    canonical, _, _ = problem.problem.get_problem_data(cp.OSQP)
    # Coefficient matrices should scale with columns times row count, not N².
    assert canonical['P'].nnz < 10 * columns
    assert canonical['A'].nnz < 40 * columns
    assert canonical['F'].nnz < 10 * columns
