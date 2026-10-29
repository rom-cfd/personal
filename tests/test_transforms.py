import numpy as np
import pytest
from kinetics.transforms import Transform, Standardizer, species_epsilon


@pytest.mark.parametrize('kind', ['boxcox', 'softlog', 'asinh', 'shifted_boxcox'])
@pytest.mark.parametrize('gamma', [0.1, 0.3, 0.5])
@pytest.mark.parametrize('signed', [False, True])
def test_roundtrip(kind, gamma, signed):
    x = np.r_[0., np.logspace(-300, 0, 100)]
    if signed:
        x = np.r_[-x[::-1], x]
    t = Transform(kind, gamma, 1e-8, signed)
    z = t.forward(x)
    assert np.all(np.isfinite(z))
    np.testing.assert_allclose(t.inverse(z), x, rtol=2e-12, atol=1e-310)
    assert np.all(np.diff(z) >= 0)


@pytest.mark.parametrize('kind', ['boxcox', 'softlog', 'asinh', 'shifted_boxcox'])
def test_positive_rejects_negative(kind):
    with pytest.raises(ValueError):
        Transform(kind).forward([-1e-20])


def test_boxcox_affine_equivalence():
    x = np.geomspace(1e-8, 1, 30)
    for g in [0.1, 0.3, 0.5]:
        z = Transform('boxcox', g).forward(x)
        conventional = (x**g - 1) / g
        np.testing.assert_allclose(Standardizer().fit(z).forward(z),
                                   Standardizer().fit(conventional).forward(conventional), atol=1e-13)


def test_domains_and_no_silent_repair():
    assert np.isnan(Transform('boxcox').inverse(-0.1))
    assert np.isnan(Transform('shifted_boxcox', gamma=.5).inverse(-3))
    assert Transform('softlog').inverse(-1) < 0
    assert Transform('asinh').inverse(-1) < 0


def test_species_scales_and_constant_columns():
    y = np.array([[0, 0, 1e-8], [1e-4, 0, 1e-6], [1e-2, 0, 1e-4]])
    e = species_epsilon(y)
    assert np.all(e > 0) and e[0] != e[2]
    assert e[1] == 1e-16
    s = Standardizer().fit(y)
    np.testing.assert_allclose(s.inverse(s.forward(y)), y, atol=1e-18)


@pytest.mark.parametrize('epsilon', [0, -1, float('nan')])
def test_bad_epsilon(epsilon):
    with pytest.raises(ValueError):
        Transform('softlog', epsilon=epsilon)
