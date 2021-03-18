"""Compatibility module defining operations on duck numpy-arrays.
Shamelessly copied from xarray."""

import numpy as np

try:
    import dask.array as dsa

    has_dask = True
except ImportError:
    has_dask = False


def _dask_or_eager_func(name, eager_module=np, list_of_args=False, n_array_args=1):
    """Create a function that dispatches to dask for dask array inputs."""
    if has_dask:

        def f(*args, **kwargs):
            dispatch_args = args[0] if list_of_args else args
            if any(isinstance(a, dsa.Array) for a in dispatch_args[:n_array_args]):
                module = dsa
            else:
                module = eager_module
            return getattr(module, name)(*args, **kwargs)

    else:

        def f(data, *args, **kwargs):
            return getattr(eager_module, name)(data, *args, **kwargs)

    return f


digitize = _dask_or_eager_func("digitize")
bincount = _dask_or_eager_func("bincount")
reshape = _dask_or_eager_func("reshape")
concatenate = _dask_or_eager_func("concatenate", list_of_args=True)
broadcast_arrays = _dask_or_eager_func("broadcast_arrays")
ravel_multi_index = _dask_or_eager_func("ravel_multi_index")
