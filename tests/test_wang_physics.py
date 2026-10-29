import numpy as np
from kinetics.wang_physics import wang_project, WangThermo


def test_paper_equations_sign_and_minimum_norm():
    thermo = WangThermo()
    rng = np.random.default_rng(51)
    current = rng.dirichlet(np.ones(9), size=30)
    proposal = current * rng.uniform(.99, 1.01, current.shape)
    result, residual = wang_project(proposal, current, thermo.matrix)
    for v, c, actual in zip(proposal, current, result):
        d = v ** .9
        b = thermo.matrix * d
        delta = b.T @ np.linalg.solve(b @ b.T, thermo.matrix @ (c - v))
        np.testing.assert_allclose(actual, v + d * delta, atol=1e-14)
    assert residual.max() < 1e-13


def test_zero_and_extremely_small_species_no_floor_or_nan():
    t = WangThermo()
    y = np.array([[0., 1e-300, 1e-100, .2, 1e-20, .1, 0., 0., .7]])
    result, error = wang_project(y, y, t.matrix)
    np.testing.assert_array_equal(result, y)
    assert error[0] == 0
    # Missing support cannot invent an absent element; caller must reject.
    v = np.zeros((1, 9)); v[0, 8] = 1
    _, error = wang_project(v, y, t.matrix)
    assert error[0] > .01


def test_negative_correction_is_not_silently_clipped():
    # One constraint: a trace component may cross zero with unconstrained solve.
    matrix = np.array([[1., 1.]])
    result, error = wang_project(np.array([[.99, .01]]), np.array([[.001, 0.]]), matrix)
    assert result.min() < 0
    assert error[0] < 1e-14


def test_wang_thermal_closure_conserves_h_elements_and_mass():
    t = WangThermo()
    y = np.array([.03, 1e-6, 1e-6, .2, 1e-5, .02, 1e-6, 1e-6, 0.])
    y[-1] = 1 - y.sum()
    x = np.r_[1200., y, 101325.][None, :]
    raw = np.r_[1300., y * 1.0001][None, None, :]
    actual = t.repair(raw, x)[0, 0]
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(t.matrix @ actual[1:], t.matrix @ y, atol=1e-13)
    np.testing.assert_allclose(actual[1:].sum(), 1., atol=1e-13)
    h0 = np.sum(t.species_h_cp(np.array([1200.]))[0][0] * y)
    h1 = np.sum(t.species_h_cp(actual[:1])[0][0] * actual[1:])
    np.testing.assert_allclose(h1, h0, atol=1e-5)
