"""Independent mathematical checks of the draft's behavioral denoising methods."""

import numpy as np
import pytest
from scipy.special import softmax

from src.data import DeePCDims
from src.policies.behavioral_denoising import BehavioralDenoisingPolicy
from src.policies.bd_kl import BDKLPolicy, HistoryConstraintError, KLHistoryProjection
from src.policies.bd_mean_chi2 import BDMeanChi2Policy


def make_policy(cls=BehavioralDenoisingPolicy, histories=None, actions=None, **options):
    histories = np.array([-1., 0., 1.]) if histories is None else np.asarray(histories)
    actions = 2 * histories + .4 if actions is None else np.asarray(actions)
    count = len(histories)
    inputs = np.vstack((np.zeros(count), actions))
    outputs = np.vstack((histories, np.zeros(count)))
    return cls(inputs, outputs, DeePCDims(1, 1, 1, 1), include_past_inputs=False, **options)


def test_sigma_euler_matches_direct_equations_at_every_step():
    policy = make_policy(noise_levels=[4., 1.2, .3, 0.])
    weights = np.array([.1, .6, .3])
    atoms = np.array([[-1.3, .4], [.1, -.2], [2.1, 1.7]])
    latent = np.array([.7, -.3])
    noisy = 4 * latent
    for level, following in zip([4., 1.2, .3], [1.2, .3, 0.]):
        posterior = weights * np.exp(-np.sum((atoms - noisy)**2, axis=1) / (2 * level**2))
        posterior /= posterior.sum()
        denoised = posterior @ atoms
        noisy += (following - level) * (noisy - denoised) / level
    actual = policy._sample_sigma(np.log(weights), atoms, latent)
    np.testing.assert_allclose(actual, noisy, rtol=1e-14, atol=1e-14)
    assert policy._smoothing == 0 and policy._sampling_scheme == 'sigma'


def test_bd_mean_retains_small_weights_and_obeys_nonuniform_prior():
    policy = make_policy(sampling='mean', bandwidth=np.inf, prior_weights=[1., 2., 7.])
    action = policy.plan(np.array([.2]))
    np.testing.assert_allclose(action, [[1.6]], atol=1e-14)
    weights = np.array([1. - 1e-12, 1e-12])
    policy._A = np.array([[0.], [1e12]])
    np.testing.assert_allclose(policy._sample(weights, np.arange(2)), [1.])


def test_signed_projection_matches_equality_constrained_quadratic_solution():
    policy = make_policy(BDMeanChi2Policy, bandwidth=.7, ridge=0.)
    differences = np.array([[-1.1, -.8], [.1, -.5], [.2, .4], [1., 1.1]])
    weights, _ = policy.history_weights(differences)
    kernel = softmax(-np.sum(differences**2, axis=1) / (2 * .7**2))
    constraints = np.vstack((np.ones(4), differences.T))
    target = np.array([1., 0., 0.])
    expected = kernel * (constraints.T @ np.linalg.solve(
        (constraints * kernel) @ constraints.T, target))
    np.testing.assert_allclose(weights, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weights @ differences, 0., atol=1e-12)


def test_signed_default_ridge_and_explicit_exact_fit_handle_thin_directions():
    exact = make_policy(BDMeanChi2Policy, bandwidth=np.inf, ridge=0.)
    regularized = make_policy(BDMeanChi2Policy, bandwidth=np.inf)
    assert regularized._ridge == 1e-6
    differences = np.array([[.001], [.003]])
    weights, _ = exact.history_weights(differences)
    ridge_weights, _ = regularized.history_weights(differences)
    np.testing.assert_allclose(weights, [1.5, -.5], atol=1e-14)
    np.testing.assert_allclose(ridge_weights @ differences, [.001], atol=1e-14)
    thin = np.array([[-1., -1e-10], [-1., 1e-10], [1., -1e-10], [1., 1e-10]])
    thin_weights, _ = exact.history_weights(thin - [0.1, 1e-11])
    np.testing.assert_allclose(thin_weights @ (thin - [0.1, 1e-11]), 0., atol=1e-14)
    with pytest.raises(ValueError, match='positive-definite'):
        exact.history_weights(np.zeros((3, 2)))
    stabilized_weights, _ = regularized.history_weights(np.zeros((3, 2)))
    np.testing.assert_allclose(stabilized_weights, np.full(3, 1/3))


def test_kl_projection_matches_independent_constrained_solver():
    import cvxpy as cp

    differences = np.array([[-1., -.5], [-.4, .7], [.6, -.8], [1., .9], [.2, .1]])
    reference = np.array([.07, .11, .19, .31, .32])
    weights = KLHistoryProjection(differences).project(np.log(reference))
    variable = cp.Variable(5, nonneg=True)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.kl_div(variable, reference))),
                         [cp.sum(variable) == 1, differences.T @ variable == 0])
    problem.solve(solver='CLARABEL', tol_gap_abs=1e-11, tol_feas=1e-11, tol_gap_rel=1e-11)
    assert problem.status == cp.OPTIMAL
    np.testing.assert_allclose(weights, variable.value, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(weights @ differences, 0., atol=1e-9)
    actual_objective = np.sum(weights * np.log(weights / reference) - weights + reference)
    assert abs(actual_objective - problem.value) < 1e-9


def test_kl_projection_supports_redundant_histories_and_boundary_faces():
    differences = np.array([[0., 0.], [0., 0.], [1., 2.], [2., 4.]])
    weights = KLHistoryProjection(differences).project(np.log([.1, .3, .2, .4]))
    np.testing.assert_allclose(weights, [.25, .75, 0., 0.], atol=1e-14)
    redundant = np.array([[-1., -2.], [0., 0.], [1., 2.]])
    weights = KLHistoryProjection(redundant).project(np.array([0., 1e5, 2e5]))
    np.testing.assert_allclose(weights, [1/3, 1/3, 1/3], atol=1e-9)


def test_kl_rejects_infeasible_or_unsupported_history():
    with pytest.raises(HistoryConstraintError, match='outside the convex hull'):
        KLHistoryProjection(np.array([[1.], [2.]]))
    with pytest.raises(HistoryConstraintError, match='outside the convex hull'):
        KLHistoryProjection(np.array([[-1.], [1.]]), support=np.array([True, False]))
    with pytest.raises(ValueError, match="sampling_scheme='sigma'"):
        make_policy(BDKLPolicy, sampling_scheme='interpolant')


def test_kl_nonconvergence_is_an_error():
    generator = np.random.default_rng(51)
    differences = generator.normal(size=(15, 3))
    differences -= differences.mean(axis=0)
    projector = KLHistoryProjection(differences, max_iterations=1, tolerance=1e-14)
    with pytest.raises(RuntimeError, match='did not converge'):
        projector.project(generator.normal(size=15))


def test_bd_kl_projects_each_action_posterior_including_final_sample():
    policy = make_policy(BDKLPolicy, histories=[-1., 1.], actions=[-1., 1.],
                         noise_levels=[5., 1., .2, 0.], bandwidth=np.inf)
    projected = []
    original = policy._sigma_weights
    def recording(noisy, level, log_weights, atoms):
        weights = original(noisy, level, log_weights, atoms)
        projected.append((level, weights.copy()))
        return weights
    policy._sigma_weights = recording
    action = policy.plan(np.array([0.]))
    assert len(projected) == 3
    for _, weights in projected:
        np.testing.assert_allclose(weights, [.5, .5], atol=1e-9)
    np.testing.assert_allclose(action, [[0.]], atol=1e-9)
    assert policy.diagnostics()['consistency_residual'] < 1e-9


def test_clipped_action_feedback_changes_next_live_history():
    policy = make_policy(sampling='mean')
    policy.act(np.array([0.]))
    policy.record_applied_action(np.array([.15]))
    np.testing.assert_allclose(policy._u_past, [.15])
    with pytest.raises(ValueError, match='finite vector'):
        policy.record_applied_action(np.array([np.nan]))


@pytest.mark.parametrize('levels', [[1., .5], [0., 1.], [1., 1., 0.], [1., np.nan, 0.]])
def test_invalid_sigma_schedule_fails(levels):
    with pytest.raises(ValueError, match='noise_levels'):
        make_policy(noise_levels=levels)


def test_sigma_conditioning_recovers_atom_lost_by_history_softmax_underflow():
    from types import SimpleNamespace

    policy = make_policy(histories=[0., 1.], actions=[0., 1.],
                         bandwidth=1., noise_levels=[1., 0.])
    policy._Z = np.array([[0.], [np.sqrt(2000.)]])
    policy._A = np.array([[0.], [100.]])
    policy._z_mean = policy._a_mean = np.zeros(1)
    policy._z_scale = policy._a_scale = np.ones(1)
    policy._rng = SimpleNamespace(standard_normal=lambda size: np.full(size, 100.))
    history_logs = policy.history_log_weights(policy._Z)
    np.testing.assert_allclose(history_logs, [0., -1000.])
    assert softmax(history_logs)[1] == 0.  # It must survive in log form.
    np.testing.assert_allclose(policy.plan(np.array([0.])), [[100.]], atol=1e-12)
    assert '_sampling_log_weights' not in policy.diagnostics()


def test_large_finite_prior_weights_normalize_without_overflow():
    policy = make_policy(prior_weights=[1e308, 1e308, 1e308], bandwidth=np.inf, sampling='mean')
    np.testing.assert_allclose(policy._prior_weights, [1/3, 1/3, 1/3])


@pytest.mark.parametrize('bandwidth', [0., -1., np.nan])
def test_invalid_bandwidth_rejected(bandwidth):
    with pytest.raises(ValueError, match='bandwidth must be positive'):
        make_policy(bandwidth=bandwidth)


@pytest.mark.parametrize('epsilon', [1e-8, 1e-10, 1e-12])
def test_kl_keeps_atoms_with_tiny_required_near_boundary_mass(epsilon):
    differences = np.array([[-epsilon], [1.]])
    projector = KLHistoryProjection(differences, tolerance=1e-14)
    weights = projector.project(np.zeros(2))
    expected = np.array([1., epsilon]) / (1. + epsilon)
    np.testing.assert_allclose(weights, expected, atol=1e-14, rtol=0.)
    assert np.linalg.norm(weights @ differences) <= 1e-14
