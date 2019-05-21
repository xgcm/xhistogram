"""Compatibility module defining operations on duck numpy-arrays.
Shamelessly copied from xarray."""

import numpy as np
from functools import reduce

try:
    import dask.array as dsa
    has_dask = True
except ImportError:
    has_dask = False


def _dask_or_eager_func(name, eager_module=np, list_of_args=False,
                        n_array_args=1):
    """Create a function that dispatches to dask for dask array inputs."""
    if has_dask:
        def f(*args, **kwargs):
            dispatch_args = args[0] if list_of_args else args
            if any(isinstance(a, dsa.Array)
                   for a in dispatch_args[:n_array_args]):
                module = dsa
            else:
                module = eager_module
            return getattr(module, name)(*args, **kwargs)
    else:
        def f(data, *args, **kwargs):
            return getattr(eager_module, name)(data, *args, **kwargs)
    return f


digitize = _dask_or_eager_func('digitize')
bincount = _dask_or_eager_func('bincount')
reshape = _dask_or_eager_func('reshape')
concatenate = _dask_or_eager_func('concatenate', list_of_args=True)
#broadcast_to = _dask_or_eager_func('broadcast_to')
broadcast_arrays = _dask_or_eager_func('broadcast_arrays')

# dask doesn't yet have this
# https://github.com/dask/dask/issues/2557
# ravel_multi_index = _dask_or_eager_func('ravel_multi_index')
def ravel_multi_index(multi_index, dims, order='C'):
    # poor clone of the numpy function
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel_multi_index.html
    # but written in a dask-friendly way
    # implementation is from this: https://stackoverflow.com/a/20266350/3266235

    assert len(multi_index) == len(dims)
    if order=='C':
        offset_factors = [reduce(lambda x, y: x*y, dims[n:])
                          for n in range(1, len(dims))] + [1,]
    elif order=='F':
        offset_factors = [1,] + [reduce(lambda x, y: x*y, dims[:n])
                                 for n in range(1, len(dims))]
    full_index = [of * ix for of, ix in zip(offset_factors, multi_index)]
    return sum(full_index)


# def broadcast_arrays(*args):
#     all_shapes = np.array([a.shape for a in args])
#     max_shape = all_shapes.max(axis=0)
#     for shape in all_shapes:
#         for n, nmax in zip(shape, max_shape):
#             if n==1 or n==max_shape:
#                 pass
#             else:
#                 raise ValueError('Incompatible shape for broadcasting. '
#                                  f'shape: {shape}, max_shape: {max_shape}')
#     is_dask = [isinstance(a, dsa.Array) for a in args]
#     chunk_shapes = [a.chunks for a in all_shapes]
