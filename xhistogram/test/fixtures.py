import random
import string

import pytest
import dask
import dask.array as dsa
import numpy as np
import xarray as xr


def empty_dask_array(shape, dtype=float, chunks=None):
    # a dask array that errors if you try to comput it
    def raise_if_computed():
        raise ValueError("Triggered forbidden computation")

    a = dsa.from_delayed(dask.delayed(raise_if_computed)(), shape, dtype)
    if chunks is not None:
        a = a.rechunk(chunks)

    return a


@pytest.fixture(scope="module")
def dataarray_factory():
    def _dataarray_factory(shape=(5,20)):
        data = np.random.randn(*shape)
        dims = [random.choice(string.ascii_lowercase) for ax in shape]
        da = xr.DataArray(
            data, dims=dims, name="T"
        )
        return da
    return _dataarray_factory


@pytest.fixture(scope="module")
def dataset_factory():
    """Random dataset with every variable having the same shape"""
    def _dataset_factory(ndim=2, n_vars=2):
        shape = (8, 9, 10, 11)[:ndim]
        dims = [random.choice(string.ascii_lowercase) for ax in shape]
        ds = xr.Dataset()
        for _ in range(n_vars):
            name = random.choice(string.ascii_uppercase)
            data = np.random.randn(*shape)
            da = xr.DataArray(
                data, dims=dims, name=name
            )
            ds[name] = da
        return ds

    return _dataset_factory
