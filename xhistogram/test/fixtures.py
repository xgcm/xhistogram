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
