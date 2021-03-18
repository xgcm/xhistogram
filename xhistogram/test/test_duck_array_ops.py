import numpy as np
import dask.array as dsa
from ..duck_array_ops import (
    digitize,
    bincount,
    reshape,
    ravel_multi_index,
    broadcast_arrays,
)
from .fixtures import empty_dask_array
import pytest


@pytest.mark.parametrize(
    "function, args",
    [
        (digitize, [np.random.rand(5, 12), np.linspace(0, 1, 7)]),
        (bincount, [np.arange(10)]),
    ],
)
def test_eager(function, args):
    a = function(*args)
    assert isinstance(a, np.ndarray)


@pytest.mark.parametrize(
    "function, args, kwargs",
    [
        (digitize, [empty_dask_array((5, 12)), np.linspace(0, 1, 7)], {}),
        (bincount, [empty_dask_array((10,))], {"minlength": 5}),
        (reshape, [empty_dask_array((10, 5)), (5, 10)], {}),
        (ravel_multi_index, (empty_dask_array((10,)), empty_dask_array((10,))), {}),
    ],
)
def test_lazy(function, args, kwargs):
    # make sure nothing computes
    a = function(*args, **kwargs)
    assert isinstance(a, dsa.core.Array)


@pytest.mark.parametrize("chunks", [(5, 12), (1, 12), (5, 1)])
def test_digitize_dask_correct(chunks):
    a = np.random.rand(5, 12)
    da = dsa.from_array(a, chunks=chunks)
    bins = np.linspace(0, 1, 7)
    d = digitize(a, bins)
    dd = digitize(da, bins)
    np.testing.assert_array_equal(d, dd.compute())


def test_ravel_multi_index_correct():
    arr = np.array([[3, 6, 6], [4, 5, 1]])
    expected = np.ravel_multi_index(arr, (7, 6))
    actual = ravel_multi_index(arr, (7, 6))
    np.testing.assert_array_equal(expected, actual)

    expected = np.ravel_multi_index(arr, (7, 6), order="F")
    actual = ravel_multi_index(arr, (7, 6), order="F")
    np.testing.assert_array_equal(expected, actual)


def test_broadcast_arrays_numpy():
    a1 = np.empty((1, 5, 25))
    a2 = np.empty((4, 1, 1))

    a1b, a2b = broadcast_arrays(a1, a2)
    assert a1b.shape == (4, 5, 25)
    assert a2b.shape == (4, 5, 25)


@pytest.mark.parametrize("d1_chunks", [(5 * (1,), (25,)), ((2, 3), (25,))])
def test_broadcast_arrays_dask(d1_chunks):
    d1 = dsa.empty((5, 25), chunks=d1_chunks)
    d2 = dsa.empty((1, 25), chunks=(1, 25))

    d1b, d2b = broadcast_arrays(d1, d2)
    assert d1b.shape == (5, 25)
    assert d2b.shape == (5, 25)
    assert d1b.chunks == d1_chunks
    assert d2b.chunks == d1_chunks
