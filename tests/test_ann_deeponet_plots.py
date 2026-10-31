import numpy as np
from scripts.plot_ann_deeponet import statistics


def test_plot_bands_are_observed_ranges_and_missing_values_are_not_zero():
    values = np.array([[1., 0., np.nan, 9.], [4., 2., np.nan, np.nan]])
    result = statistics(values)
    np.testing.assert_allclose(result['minimum'], [1, 0, np.nan, 9], equal_nan=True)
    np.testing.assert_allclose(result['geometric_mean'], [2, 0, np.nan, 9], equal_nan=True)
    np.testing.assert_allclose(result['maximum'], [4, 2, np.nan, 9], equal_nan=True)
    np.testing.assert_array_equal(result['count'], [2, 2, 0, 1])
