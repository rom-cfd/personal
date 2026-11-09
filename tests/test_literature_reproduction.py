"""Check the scientific metric and published affine correction independently."""

import numpy as np
import pytest

from kinetics.literature_reproduction import bct_element_projection, ratio


def test_nrmse_uses_reference_rms_not_range_or_percentage():
    reference = np.array([[3.0], [4.0]])
    prediction = np.array([[3.0], [3.0]])
    np.testing.assert_allclose(ratio(reference, prediction), [0.2])


def test_published_projection_conserves_elements_and_minimizes_bct_distance():
    # H2, O2, H2O elemental mass fractions. The third column sums to one.
    matrix = np.array([[1.0, 0.0, 1 / 9], [0.0, 1.0, 8 / 9]])
    current = np.array([0.1, 0.2, 0.7])
    proposal = np.array([0.11, 0.22, 0.66])
    gamma = 0.1
    result = bct_element_projection(proposal, current, matrix, gamma)
    np.testing.assert_allclose(matrix @ result, matrix @ current, atol=2e-15)
    assert np.all(result > 0)

    # Feasible displacements preserve H and O. At the optimum the first-order
    # BCT squared-distance has zero directional derivative along this nullspace.
    direction = np.array([-1 / 9, -8 / 9, 1.0])
    derivative = proposal ** (gamma - 1)
    cost = lambda y: np.sum(((y - proposal) * derivative) ** 2)
    for distance in [-0.01, -0.001, 0.001, 0.01]:
        alternative = result + distance * direction
        np.testing.assert_allclose(matrix @ alternative, matrix @ current)
        assert cost(alternative) > cost(result)


def test_invalid_published_projection_does_not_silently_clip():
    with pytest.raises(ValueError, match='nonnegative'):
        bct_element_projection(np.array([-0.1, 1.1]), np.array([0.2, 0.8]), np.eye(2))
