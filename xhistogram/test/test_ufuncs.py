import numpy as np

from ..ufuncs import _histogram_2d_vectorized


def test_histogram_2d_vectorized():
    nrows, ncols = 5, 20
    data = np.random.randn(nrows, ncols)
    bins = np.linspace(-4, 4, 10)

    h2d = _histogram_2d_vectorized(data, bins)
    assert h2d.shape == (nrows, len(bins)-1)

    # make sure we get the same thing as histogram
    hist, _ = np.histogram(data, bins=bins)
    np.testing.assert_array_equal(hist, h2d.sum(axis=0))

    # check that weights works
    h2d_d = _histogram_2d_vectorized(data, bins, weights=2*np.ones_like(data))
    np.testing.assert_array_equal(2*h2d, h2d_d)
