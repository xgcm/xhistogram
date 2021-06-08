import numpy as np

from itertools import combinations
import dask.array as dsa

from ..core import (
    histogram,
    _ensure_correctly_formatted_bins,
    _ensure_correctly_formatted_range,
)
from .fixtures import empty_dask_array

import pytest


bins_int = 10
bins_str = "auto"
bins_arr = np.linspace(-4, 4, 10)
range_ = (0, 1)


@pytest.mark.parametrize("density", [False, True])
@pytest.mark.parametrize("block_size", [None, 1, 2])
@pytest.mark.parametrize("axis", [1, None])
@pytest.mark.parametrize("bins", [10, np.linspace(-4, 4, 10), "auto"])
@pytest.mark.parametrize("range_", [None, (-4, 4)])
def test_histogram_results_1d(block_size, density, axis, bins, range_):
    nrows, ncols = 5, 20
    # Setting the random seed here prevents np.testing.assert_allclose
    # from failing beow. We should investigate this further.
    np.random.seed(2)
    data = np.random.randn(nrows, ncols)

    h, bin_edges = histogram(
        data, bins=bins, range=range_, axis=axis, block_size=block_size, density=density
    )

    expected_shape = (
        (nrows, len(bin_edges[0]) - 1) if axis == 1 else (len(bin_edges[0]) - 1,)
    )
    assert h.shape == expected_shape

    # make sure we get the same thing as numpy.histogram
    if axis:
        bins_np = np.histogram_bin_edges(
            data, bins=bins, range=range_
        )  # Use same bins for all slices below
        expected = np.stack(
            [
                np.histogram(data[i], bins=bins_np, range=range_, density=density)[0]
                for i in range(nrows)
            ]
        )
    else:
        expected = np.histogram(data, bins=bins, range=range_, density=density)[0]
    norm = nrows if (density and axis) else 1
    np.testing.assert_allclose(h, expected / norm)

    if density:
        widths = np.diff(bin_edges)
        integral = np.sum(h * widths)
        np.testing.assert_allclose(integral, 1.0)


@pytest.mark.parametrize("block_size", [None, 1, 2])
def test_histogram_results_1d_weighted(block_size):
    nrows, ncols = 5, 20
    data = np.random.randn(nrows, ncols)
    bins = np.linspace(-4, 4, 10)
    h, _ = histogram(data, bins=bins, axis=1, block_size=block_size)
    weights = 2 * np.ones_like(data)
    h_w, _ = histogram(data, bins=bins, axis=1, weights=weights, block_size=block_size)
    np.testing.assert_array_equal(2 * h, h_w)


# @pytest.mark.skip(reason="Weight broadcasting on numpy arrays is not yet implemented")
@pytest.mark.parametrize("block_size", [None, 1, 2, "auto"])
def test_histogram_results_1d_weighted_broadcasting(block_size):
    nrows, ncols = 5, 20
    data = np.random.randn(nrows, ncols)
    bins = np.linspace(-4, 4, 10)
    h, _ = histogram(data, bins=bins, axis=1, block_size=block_size)
    weights = 2 * np.ones((1, ncols))
    h_w, _ = histogram(data, bins=bins, axis=1, weights=weights, block_size=block_size)
    np.testing.assert_array_equal(2 * h, h_w)


@pytest.mark.parametrize("block_size", [None, 1, 2])
def test_histogram_right_edge(block_size):
    """Test that last bin is both left- and right-edge inclusive as it
    is for numpy.histogram
    """
    nrows, ncols = 5, 20
    data = np.ones((nrows, ncols))
    bins = np.array([0, 0.5, 1])  # All data at rightmost edge

    h, _ = histogram(data, bins=bins, axis=1, block_size=block_size)
    assert h.shape == (nrows, len(bins) - 1)

    # make sure we get the same thing as histogram (all data in the last bin)
    hist, _ = np.histogram(data, bins=bins)
    np.testing.assert_array_equal(hist, h.sum(axis=0))

    # now try with no axis
    h_na, _ = histogram(data, bins=bins, block_size=block_size)
    np.testing.assert_array_equal(hist, h_na)


def test_histogram_results_2d():
    nrows, ncols = 5, 20
    data_a = np.random.randn(nrows, ncols)
    data_b = np.random.randn(nrows, ncols)
    nbins_a = 9
    bins_a = np.linspace(-4, 4, nbins_a + 1)
    nbins_b = 10
    bins_b = np.linspace(-4, 4, nbins_b + 1)

    h, _ = histogram(data_a, data_b, bins=[bins_a, bins_b])
    assert h.shape == (nbins_a, nbins_b)

    hist, _, _ = np.histogram2d(data_a.ravel(), data_b.ravel(), bins=[bins_a, bins_b])
    np.testing.assert_array_equal(hist, h)


def test_histogram_results_2d_density():
    nrows, ncols = 5, 20
    data_a = np.random.randn(nrows, ncols)
    data_b = np.random.randn(nrows, ncols)
    nbins_a = 9
    bins_a = np.linspace(-4, 4, nbins_a + 1)
    nbins_b = 10
    bins_b = np.linspace(-4, 4, nbins_b + 1)

    h, _ = histogram(data_a, data_b, bins=[bins_a, bins_b], density=True)
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


def test_histogram_results_3d_density():
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

    h, _ = histogram(
        data_a, data_b, data_c, bins=[bins_a, bins_b, bins_c], density=True
    )

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


# TODO parametrize this over axes so there is only one assert per test
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
    c, _ = histogram(b, bins=bins, block_size=block_size)
    assert c.shape == (len(bins) - 1,)
    # same thing
    for axis in [(0, 1, 2, 3), (0, 1, 3, 2), (3, 2, 1, 0), (3, 2, 0, 1)]:
        c, _ = histogram(b, bins=bins, axis=axis)
        assert c.shape == (len(bins) - 1,)
        if use_dask:
            assert isinstance(c, dsa.Array)

    # scalar axis (check positive and negative)
    for axis in list(range(4)) + list(range(-1, -5, -1)):
        c, _ = histogram(b, bins=bins, axis=axis, block_size=block_size)
        shape = list(b.shape)
        del shape[axis]
        expected_shape = tuple(shape) + (len(bins) - 1,)
        assert c.shape == expected_shape
        if use_dask:
            assert isinstance(c, dsa.Array)

    # two axes
    for i, j in combinations(range(4), 2):
        axis = (i, j)
        c, _ = histogram(b, bins=bins, axis=axis, block_size=block_size)
        shape = list(b.shape)
        partial_shape = [shape[k] for k in range(b.ndim) if k not in axis]
        expected_shape = tuple(partial_shape) + (len(bins) - 1,)
        assert c.shape == expected_shape
        if use_dask:
            assert isinstance(c, dsa.Array)


def test_histogram_dask():
    """ Test that fails with dask arrays and inappropriate bins"""
    shape = 10, 15, 12, 20
    b = empty_dask_array(shape, chunks=(1,) + shape[1:])
    histogram(b, bins=bins_arr)  # Should work when bins is all numpy arrays
    with pytest.raises(TypeError):  # Should fail otherwise
        histogram(b, bins=bins_int)
        histogram(b, bins=bins_str)
        histogram(b, b, bins=[bins_arr, bins_int])


@pytest.mark.parametrize(
    "in_out",
    [
        (bins_int, 1, [bins_int]),  # ( bins_in, n_args, bins_out )
        (bins_str, 1, [bins_str]),
        (bins_arr, 1, [bins_arr]),
        ([bins_int], 1, [bins_int]),
        (bins_int, 2, 2 * [bins_int]),
        (bins_str, 2, 2 * [bins_str]),
        (bins_arr, 2, 2 * [bins_arr]),
        ([bins_int, bins_str, bins_arr], 3, [bins_int, bins_str, bins_arr]),
        ([bins_arr], 2, None),
        (None, 1, None),
        ([bins_arr, bins_arr], 1, None),
    ],
)
def test_ensure_correctly_formatted_bins(in_out):
    """ Test the helper function _ensure_correctly_formatted_bins"""
    bins_in, n, bins_expected = in_out
    if bins_expected is not None:
        bins = _ensure_correctly_formatted_bins(bins_in, n)
        assert bins == bins_expected
    else:
        with pytest.raises((ValueError, TypeError)):
            _ensure_correctly_formatted_bins(bins_in, n)


@pytest.mark.parametrize(
    "in_out",
    [
        (range_, 1, [range_]),  # ( range_in, n_args, range_out )
        (range_, 2, [range_, range_]),
        ([range_, range_], 2, [range_, range_]),
        ([(range_[0],)], 1, None),
        ([range_], 2, None),
        ([range_, range_], 1, None),
    ],
)
def test_ensure_correctly_formatted_range(in_out):
    """ Test the helper function _ensure_correctly_formatted_range"""
    range_in, n, range_expected = in_out
    if range_expected is not None:
        range_ = _ensure_correctly_formatted_range(range_in, n)
        assert range_ == range_expected
    else:
        with pytest.raises(ValueError):
            _ensure_correctly_formatted_range(range_in, n)
