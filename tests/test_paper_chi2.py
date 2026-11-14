"""Mathematical checks for the paper's signed projection and tracked refit."""

from types import MethodType

import numpy as np
import pytest

from src.data import DeePCDims
from src.policies.bd_chi2 import BDChi2Policy


def make_policy(**options):
    histories = np.array([[-1.0, -.4], [-.7, .8], [.2, -.6], [.6, .3], [1.2, 1.1]])
    actions = (histories @ np.array([.8, -.35]) + .2)[:, None]
    count = len(histories)
    inputs = np.vstack((np.zeros((1, count)), actions.T))
    outputs = np.vstack((histories.T, np.zeros((2, count))))
    return BDChi2Policy(inputs, outputs, DeePCDims(1, 1, 2, 1), include_past_inputs=False, **options)


def test_paper_mode_exact_affine_reproduction_and_limited_target():
    policy = make_policy(slope_ridge=0.)
    assert policy._slope_ridge == 0
    weights = np.array([.1, .15, .3, .2, .25])
    histories = np.array([[-1., -.4], [-.7, .8], [.2, -.6], [.6, .3], [1.2, 1.1]])
    query = np.array([2.1, -.8])
    gain = np.array([[1.3, -.7], [-.2, .6]])
    bias = np.array([.4, -.25])
    actions = histories @ gain.T + bias
    intercept, fitted_gain = policy._affine_fit(weights, histories - query, actions)
    np.testing.assert_allclose(fitted_gain, gain, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(intercept, gain @ query + bias, rtol=1e-12, atol=1e-12)
    limited, _ = policy._affine_fit(weights, histories - query, actions, limit=.2)
    mean_history = weights @ histories
    covariance = (histories - mean_history).T @ np.diag(weights) @ (histories - mean_history)
    displacement = query - mean_history
    alpha = min(1., .2 / np.sqrt(np.sum(weights**2) * displacement @ np.linalg.solve(covariance, displacement)))
    assert alpha < 1
    target = mean_history + alpha * displacement
    np.testing.assert_allclose(limited, gain @ target + bias, rtol=1e-12, atol=1e-12)


def test_one_irls_step_increases_mass_and_returns_its_fit_weights():
    policy = make_policy(iterations=1, mode_width=.7, slope_ridge=0.)
    differences = np.array([[-1., -.4], [-.7, .8], [.2, -.6], [.6, .3], [1.2, 1.1]])
    actions = np.array([[-.9], [-.2], [.6], [.7], [2.1]])
    kernel_weights = np.array([.1, .15, .3, .2, .25])
    initial_intercept, initial_gain = np.array([.1]), np.array([[.2, -.1]])
    residual = actions - initial_intercept - differences @ initial_gain.T
    old_factors = np.exp(-np.sum(residual**2, axis=1) / (2*.7**2))
    expected_weights = kernel_weights * old_factors
    expected_weights /= expected_weights.sum()
    old_mass = kernel_weights @ old_factors
    intercept, gain, weights, mass = policy._refit(np.log(kernel_weights), differences, actions,
                                                  initial_intercept, initial_gain)
    np.testing.assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    # Independent least-squares formulation with an explicit intercept column.
    design = np.column_stack((np.ones(len(weights)), differences))
    solution = np.linalg.lstsq(np.sqrt(weights)[:, None] * design,
                              np.sqrt(weights)[:, None] * actions, rcond=None)[0]
    np.testing.assert_allclose(intercept, solution[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gain, solution[1:].T, rtol=1e-12, atol=1e-12)
    assert mass >= old_mass - 1e-13


def test_explicit_exact_mode_rejects_rank_deficiency_default_ridge_handles_it():
    differences = np.array([[-1., -2.], [0., 0.], [1., 2.]])
    actions = np.array([[-1.], [0.], [1.]])
    weights = np.ones(3)/3
    with pytest.raises(ValueError, match='full-rank.*exact consistency'):
        make_policy(slope_ridge=0.)._affine_fit(weights, differences, actions)
    regularized = make_policy()
    assert regularized._slope_ridge == 1e-6
    intercept, gain = regularized._affine_fit(weights, differences, actions)
    assert np.isfinite(intercept).all() and np.isfinite(gain).all()


def test_ridge_is_not_an_exact_projection_or_monotone_mass_claim():
    differences = np.array([[-1e-4], [1e-4]])
    actions = np.array([[-1.], [1.]])
    weights = np.array([.5, .5])
    exact = make_policy(slope_ridge=0.)
    _, exact_gain = exact._affine_fit(weights, differences, actions)
    regularized = make_policy(slope_ridge=1e-6)
    intercept, gain = regularized._affine_fit(weights, differences, actions)
    np.testing.assert_allclose(exact_gain, [[10000.]])
    np.testing.assert_allclose(gain, [[99.009900990099]], rtol=1e-12)
    mass = weights @ np.exp(-np.sum((actions-intercept-differences@gain.T)**2, axis=1)/2)
    assert mass < .7  # The preceding exact line has mass one.


def test_tracking_transports_and_stores_unlimited_line():
    policy = make_policy(mode_width=np.inf, extrapolation_limit=.01)
    first_history, second_history = np.array([2., -.7]), np.array([2.1, -.65])
    policy.plan(first_history)
    previous, previous_intercept, previous_gain = [value.copy() for value in policy._tracked]
    seen = []
    original_refit = policy._refit

    def recording_refit(self, log_weights, differences, actions, intercept, gain):
        seen.append((intercept.copy(), gain.copy()))
        return original_refit(log_weights, differences, actions, intercept, gain)

    policy._refit = MethodType(recording_refit, policy)
    action = policy.plan(second_history)
    query = (second_history - policy._z_mean) / policy._z_scale
    np.testing.assert_allclose(seen[0][0], previous_intercept + previous_gain @ (query-previous))
    np.testing.assert_allclose(seen[0][1], previous_gain)
    assert policy._selections == 1
    stored_intercept = policy._tracked[1]
    assert not np.allclose(action.ravel(), stored_intercept * policy._a_scale + policy._a_mean)


def test_thin_but_positive_definite_history_is_not_silently_regularized():
    differences = np.array([[-1e-10, 0.], [1e-10, 0.], [0., -1.], [0., 1.]])
    weights = np.ones(4)/4
    true_gain = np.array([[2., 3.]])
    actions = differences @ true_gain.T
    intercept, gain = make_policy(slope_ridge=0.)._affine_fit(weights, differences, actions)
    np.testing.assert_allclose(intercept, [0.], atol=1e-12)
    np.testing.assert_allclose(gain, true_gain, rtol=1e-12, atol=1e-12)


def test_action_posterior_can_recover_history_mass_below_float_range():
    # The second history has log weight -1000, but the action posterior's
    # squared-distance term favors its action by 20000 at this noise level.
    inputs = np.array([[0., 0.], [0., 100.]])
    outputs = np.array([[0., 1.], [0., 0.]])
    policy = BDChi2Policy(inputs, outputs, DeePCDims(1, 1, 1, 1), include_past_inputs=False,
                 bandwidth=np.sqrt(.002), noise_levels=[.01, 0.],
                 mode_width=.02, slope_ridge=1e-6)
    policy._latent = np.array([100.])
    np.testing.assert_allclose(policy.plan(np.array([0.])), [[100.]], atol=1e-12)


def test_default_ridge_stabilizes_rank_deficient_policy_histories():
    inputs = np.array([[0., 0., 0.], [-1., 0., 1.]])
    outputs = np.array([[-1., 0., 1.], [-2., 0., 2.], [0., 0., 0.], [0., 0., 0.]])
    policy = BDChi2Policy(inputs, outputs, DeePCDims(1, 1, 2, 1),
                          include_past_inputs=False, bandwidth=np.inf,
                          mode_width=np.inf, diffusion_steps=2)
    assert policy._slope_ridge == 1e-6
    action = policy.plan(np.array([.2, .4]))
    # Standardized histories have identical coordinates. Adding epsilon I
    # yields the two-dimensional ridge prediction 2 / (2 + epsilon) times u.
    expected = .2 * 2 / (2 + 1e-6)
    np.testing.assert_allclose(action, [[expected]], rtol=1e-10, atol=1e-12)


def test_low_mass_refit_reselects_with_the_same_episode_latent():
    policy = make_policy(bandwidth=np.inf, mode_width=1e-4, reselect_mass=.99,
                         noise_levels=[.01, 0.])
    policy._latent = np.zeros(1)
    latent = policy._latent.copy()
    policy.plan(np.array([0., 0.]))
    assert policy.diagnostics()['mode_mass'] < .99
    assert policy.diagnostics()['mode_selections'] == 1
    policy.plan(np.array([.01, -.01]))
    assert policy.diagnostics()['mode_selections'] == 2
    np.testing.assert_array_equal(policy._latent, latent)
    policy.reset()
    assert policy._tracked is None and policy._selections == 0
