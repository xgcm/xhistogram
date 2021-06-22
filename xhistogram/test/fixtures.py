import uuid
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


def example_dataarray(shape=(5, 20)):
    data = np.random.randn(*shape)
    dims = [f"dim_{i}" for i in range(len(shape))]
    da = xr.DataArray(data, dims=dims, name="T")
    return da


def example_dataset(n_dim=2, n_vars=2):
    """Random dataset with every variable having the same shape"""

    shape = tuple(range(8, 8 + n_dim))
    dims = [f"dim_{i}" for i in range(len(shape))]
    var_names = [uuid.uuid4().hex for _ in range(n_vars)]
    ds = xr.Dataset()
    for i in range(n_vars):
        name = var_names[i]
        data = np.random.randn(*shape)
        da = xr.DataArray(data, dims=dims, name=name)
        ds[name] = da
    return ds
