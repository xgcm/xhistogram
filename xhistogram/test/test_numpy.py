import numpy as np

from ..numpy import _histogram_2d_vectorized, histogram
from itertools import combinations


def test_histogram_results_1d():
    nrows, ncols = 5, 20
    data = np.random.randn(nrows, ncols)
    bins = np.linspace(-4, 4, 10)

    h2d = histogram(data, bins=bins, axis=1)
    assert h2d.shape == (nrows, len(bins)-1)

    # make sure we get the same thing as histogram
    hist, _ = np.histogram(data, bins=bins)
    np.testing.assert_array_equal(hist, h2d.sum(axis=0))

    # check that weights works
    h2d_d = histogram(data, bins=bins, axis=1, weights=2*np.ones_like(data))
    np.testing.assert_array_equal(2*h2d, h2d_d)


def test_histogram_shape():
    b = np.random.randn(10, 15, 12, 20)
    bins = np.linspace(-4, 4, 27)

    # no axis
    c = histogram(b, bins=bins)
    assert c.shape == (len(bins) - 1,)
    # same thing
    for axis in [(0, 1, 2, 3), (0, 1, 3, 2), (3, 2, 1, 0), (3, 2, 0, 1)]:
        c = histogram(b, bins=bins, axis=axis)
        assert c.shape == (len(bins) - 1,)

    # scalar axis
    for axis in range(4):
        c = histogram(b, bins=bins, axis=axis)
        shape = list(b.shape)
        del shape[axis]
        expected_shape = tuple(shape) + (len(bins) - 1,)
        assert c.shape == expected_shape

    # two axes
    for i, j in combinations(range(4), 2):
        print(b.shape)
        axis = (i, j)
        print(axis)
        c = histogram(b, bins=bins, axis=axis)
        shape = list(b.shape)
        partial_shape = [shape[k] for k in range(b.ndim) if k not in axis]
        expected_shape = tuple(partial_shape) + (len(bins) - 1,)
        print(c.shape)
        print(expected_shape)
        assert c.shape == expected_shape
