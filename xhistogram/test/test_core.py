import numpy as np

from itertools import combinations
import dask.array as dsa

from ..core import _bincount_kernel, histogram
from .fixtures import empty_dask_array

import pytest


# TODO parametrize over density
# TODO parametrize over number of chunks N
class TestBincountKernel:
    def test_bincount_1d_single_chunk(self):
        # Ensure input array has a leading length 1 axis
        np.random.seed(2)
        data = np.random.randn(1, 30)
        reduce_axis = 1
        bins = np.linspace(-4, 4, 10)

        h = _bincount_kernel(data, bins, reduce_axis,
                             weights=None).sum((0, reduce_axis))

        # compare to numpy
        expected = np.histogram(data[0], bins)[0]
        np.testing.assert_allclose(h, expected)

    def test_bincount_1d_stacked_chunks(self):
        # Same but split the data up into two chunks
        data = np.random.randn(1, 30)
        chunk1, chunk2 = data[:, 15], data[:, 15:]
        input = np.stack(chunk1, chunk2)
        reduce_axis = 1
        bins = np.linspace(-4, 4, 10)

        h = _bincount_kernel(input, bins, reduce_axis,
                             weights=None).sum(axis=(0, reduce_axis))

        expected = np.histogram(data[0], bins)[0]
        np.testing.assert_allclose(h, expected)

    def test_bincount_2d_binned_along_1d(self):
        # Leave one axis alone
        data = np.random.randn(1, 30, 3)
        axis = 0
        bins = np.linspace(-4, 4, 10)
        h = _bincount_kernel(data, bins, axis, weights=None).sum(axis)

        expected = np.stack(
            [np.histogram(data[:,i,:], bins=bins)[0] for i in range(3)]
        )
        np.testing.assert_allclose(h, expected)

    def test_bincount_2d(self):
        nrows, ncols = 5, 20
        data_a = np.random.randn(1, nrows, ncols)
        data_b = np.random.randn(1, nrows, ncols)
        nbins_a = 9
        bins_a = np.linspace(-4, 4, nbins_a + 1)
        nbins_b = 10
        bins_b = np.linspace(-4, 4, nbins_b + 1)
        reduce_axes = (1, 2)

        h = _bincount_kernel(np.stack(data_a, data_b), bins=[bins_a, bins_b],
                             axis=reduce_axes, weights=None).sum((0, reduce_axes))
        assert h.shape == (1, nbins_a, nbins_b)

        hist, _, _ = np.histogram2d(data_a.ravel(), data_b.ravel(), bins=[bins_a, bins_b])
        np.testing.assert_array_equal(hist, h)

    def test_bincount_2d_stacked_chunks(self):
        ...

    def test_bincount_2d_bystander(self):
        ...

    def test_bincount_3d(self):
        ...

    # TODO test weighted


# TODO parameterize to run all tests with both dask and numpy
class TestHistogramResults:
    @pytest.mark.parametrize("density", [False, True])
    @pytest.mark.parametrize("block_size", [None, 1, 2])
    @pytest.mark.parametrize("axis", [1, None])
    def test_histogram_results_1d(self, block_size, density, axis):
        nrows, ncols = 5, 20
        # Setting the random seed here prevents np.testing.assert_allclose
        # from failing beow. We should investigate this further.
        np.random.seed(2)
        data = np.random.randn(nrows, ncols)
        bins = np.linspace(-4, 4, 10)

        h = histogram(data, bins=bins, axis=axis, block_size=block_size, density=density)

        expected_shape = (nrows, len(bins) - 1) if axis == 1 else (len(bins) - 1,)
        assert h.shape == expected_shape

        # make sure we get the same thing as numpy.histogram
        if axis:
            expected = np.stack(
                [np.histogram(data[i], bins=bins, density=density)[0] for i in range(nrows)]
            )
        else:
            expected = np.histogram(data, bins=bins, density=density)[0]
        norm = nrows if (density and axis) else 1
        np.testing.assert_allclose(h, expected / norm)

        if density:
            widths = np.diff(bins)
            integral = np.sum(h * widths)
            np.testing.assert_allclose(integral, 1.0)


    @pytest.mark.parametrize("block_size", [None, 1, 2])
    def test_histogram_results_1d_weighted(self, block_size):
        nrows, ncols = 5, 20
        data = np.random.randn(nrows, ncols)
        bins = np.linspace(-4, 4, 10)
        h = histogram(data, bins=bins, axis=1, block_size=block_size)
        weights = 2 * np.ones_like(data)
        h_w = histogram(data, bins=bins, axis=1, weights=weights, block_size=block_size)
        np.testing.assert_array_equal(2 * h, h_w)


    # @pytest.mark.skip(reason="Weight broadcasting on numpy arrays is not yet implemented")
    @pytest.mark.parametrize("block_size", [None, 1, 2, "auto"])
    def test_histogram_results_1d_weighted_broadcasting(self, block_size):
        nrows, ncols = 5, 20
        data = np.random.randn(nrows, ncols)
        bins = np.linspace(-4, 4, 10)
        h = histogram(data, bins=bins, axis=1, block_size=block_size)
        weights = 2 * np.ones((1, ncols))
        h_w = histogram(data, bins=bins, axis=1, weights=weights, block_size=block_size)
        np.testing.assert_array_equal(2 * h, h_w)


    @pytest.mark.parametrize("block_size", [None, 1, 2])
    def test_histogram_right_edge(self, block_size):
        """Test that last bin is both left- and right-edge inclusive as it
        is for numpy.histogram
        """
        nrows, ncols = 5, 20
        data = np.ones((nrows, ncols))
        bins = np.array([0, 0.5, 1])  # All data at rightmost edge

        h = histogram(data, bins=bins, axis=1, block_size=block_size)
        assert h.shape == (nrows, len(bins) - 1)

        # make sure we get the same thing as histogram (all data in the last bin)
        hist, _ = np.histogram(data, bins=bins)
        np.testing.assert_array_equal(hist, h.sum(axis=0))

        # now try with no axis
        h_na = histogram(data, bins=bins, block_size=block_size)
        np.testing.assert_array_equal(hist, h_na)


    def test_histogram_results_2d(self):
        nrows, ncols = 5, 20
        data_a = np.random.randn(nrows, ncols)
        data_b = np.random.randn(nrows, ncols)
        nbins_a = 9
        bins_a = np.linspace(-4, 4, nbins_a + 1)
        nbins_b = 10
        bins_b = np.linspace(-4, 4, nbins_b + 1)

        h = histogram(data_a, data_b, bins=[bins_a, bins_b])
        assert h.shape == (nbins_a, nbins_b)

        hist, _, _ = np.histogram2d(data_a.ravel(), data_b.ravel(), bins=[bins_a, bins_b])
        np.testing.assert_array_equal(hist, h)


    def test_histogram_results_2d_density(self):
        nrows, ncols = 5, 20
        data_a = np.random.randn(nrows, ncols)
        data_b = np.random.randn(nrows, ncols)
        nbins_a = 9
        bins_a = np.linspace(-4, 4, nbins_a + 1)
        nbins_b = 10
        bins_b = np.linspace(-4, 4, nbins_b + 1)

        h = histogram(data_a, data_b, bins=[bins_a, bins_b], density=True)
        assert h.shape == (nbins_a, nbins_b)

        hist, _, _ = np.histogram2d(
            data_a.ravel(), data_b.ravel(), bins=[bins_a, bins_b], density=True
        )
        np.testing.assert_allclose(hist, h)

        # check integral is 1
        widths_a = np.diff(bins_a)
        widths_b = np.diff(bins_b)
        areas = np.outer(widths_a, widths_b)
        integral = np.sum(hist * areas)
        np.testing.assert_allclose(integral, 1.0)


    def test_histogram_results_3d_density(self):
        nrows, ncols = 5, 20
        data_a = np.random.randn(nrows, ncols)
        data_b = np.random.randn(nrows, ncols)
        data_c = np.random.randn(nrows, ncols)
        nbins_a = 9
        bins_a = np.linspace(-4, 4, nbins_a + 1)
        nbins_b = 10
        bins_b = np.linspace(-4, 4, nbins_b + 1)
        nbins_c = 9
        bins_c = np.linspace(-4, 4, nbins_c + 1)

        h = histogram(data_a, data_b, data_c, bins=[bins_a, bins_b, bins_c], density=True)

        assert h.shape == (nbins_a, nbins_b, nbins_c)

        hist, _ = np.histogramdd(
            (data_a.ravel(), data_b.ravel(), data_c.ravel()),
            bins=[bins_a, bins_b, bins_c],
            density=True,
        )

        np.testing.assert_allclose(hist, h)

        # check integral is 1
        widths_a = np.diff(bins_a)
        widths_b = np.diff(bins_b)
        widths_c = np.diff(bins_c)
        areas = np.einsum("i,j,k", widths_a, widths_b, widths_c)
        integral = np.sum(hist * areas)
        np.testing.assert_allclose(integral, 1.0)


@pytest.mark.parametrize("block_size", [None, 5, "auto"])
@pytest.mark.parametrize("use_dask", [False, True])
def test_histogram_shape(use_dask, block_size):
    """These tests just verify that arrays with the right shape come out.
    They don't verify correctness."""

    shape = 10, 15, 12, 20
    if use_dask:
        b = empty_dask_array(shape, chunks=(1,) + shape[1:])
    else:
        b = np.random.randn(*shape)
    bins = np.linspace(-4, 4, 27)

    # no axis
    c = histogram(b, bins=bins, block_size=block_size)
    assert c.shape == (len(bins) - 1,)
    # same thing
    for axis in [(0, 1, 2, 3), (0, 1, 3, 2), (3, 2, 1, 0), (3, 2, 0, 1)]:
        c = histogram(b, bins=bins, axis=axis)
        assert c.shape == (len(bins) - 1,)
        if use_dask:
            assert isinstance(c, dsa.Array)

    # scalar axis (check positive and negative)
    for axis in list(range(4)) + list(range(-1, -5, -1)):
        c = histogram(b, bins=bins, axis=axis, block_size=block_size)
        shape = list(b.shape)
        del shape[axis]
        expected_shape = tuple(shape) + (len(bins) - 1,)
        assert c.shape == expected_shape
        if use_dask:
            assert isinstance(c, dsa.Array)

    # two axes
    for i, j in combinations(range(4), 2):
        axis = (i, j)
        c = histogram(b, bins=bins, axis=axis, block_size=block_size)
        shape = list(b.shape)
        partial_shape = [shape[k] for k in range(b.ndim) if k not in axis]
        expected_shape = tuple(partial_shape) + (len(bins) - 1,)
        assert c.shape == expected_shape
        if use_dask:
            assert isinstance(c, dsa.Array)
