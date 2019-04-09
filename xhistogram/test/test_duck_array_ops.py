import numpy as np
import dask
import dask.array as dsa
from ..duck_array_ops import digitize, bincount, ravel_multi_index
import pytest

def _empty_dask_array(shape, dtype=float, chunks=None):
    # a dask array that errors if you try to comput it
    def raise_if_computed():
        raise ValueError('Triggered forbidden computation')

    a = dsa.from_delayed(dask.delayed(raise_if_computed)(), shape, dtype)
    if chunks is not None:
        a = a.rechunk(chunks)

    return a

@pytest.mark.parametrize('function, args', [
    (digitize, [np.random.rand(5, 12), np.linspace(0, 1, 7)]),
    (bincount, [np.arange(10)])
])
def test_eager(function, args):
    a = function(*args)
    assert isinstance(a, np.ndarray)


@pytest.mark.parametrize('function, args, kwargs', [
    (digitize, [_empty_dask_array((5, 12)), np.linspace(0, 1, 7)], {}),
    (bincount, [_empty_dask_array((10,))], {'minlength': 5}),
    (ravel_multi_index, (_empty_dask_array((10,)), _empty_dask_array((10,))), {})
])
def test_lazy(function, args, kwargs):
    # make sure nothing computes
    a = function(*args, **kwargs)
    assert isinstance(a, dsa.core.Array)


@pytest.mark.parametrize('chunks', [(5, 12), (1, 12), (5, 1)])
def test_digitize_dask_correct(chunks):
    a = np.random.rand(5, 12)
    da = dsa.from_array(a, chunks=chunks)
    bins = np.linspace(0, 1, 7)
    d = digitize(a, bins)
    dd = digitize(da, bins)
    np.testing.assert_array_equal(d, dd.compute())
    

def test_ravel_multi_index_correct():
    arr = np.array([[3,6,6],[4,5,1]])
    expected = np.ravel_multi_index(arr, (7,6))
    actual = ravel_multi_index(arr, (7,6))
    np.testing.assert_array_equal(expected, actual)

    expected = np.ravel_multi_index(arr, (7,6), order='F')
    actual = ravel_multi_index(arr, (7,6), order='F')
    np.testing.assert_array_equal(expected, actual)
