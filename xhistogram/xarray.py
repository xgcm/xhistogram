"""
Xarray API for xhistogram.
"""

import xarray as xr
import numpy as np
from .core import histogram as _histogram


def histogram(*args, bins=None, dims=None, weights=None, density=False,
              right=False, bin_dim_suffix='_bin'):
    """Histogram applied along specified dimensions.

    Parameters
    ----------
    args : xarray.DataArray objects
        Input data. The number of input arguments determines the dimensonality of
        the histogram. For example, two arguments prodocue a 2D histogram. All
        args must be aligned and have the same dimensions.
    bins :  int or array_like or [int, int, ...] or [array, array, ...], optional
        The bin specification:

          * If int, the number of bins for the two dimensions (nx=ny=bins).
          * If array_like, the bin edges for the two dimensions
            (x_edges=y_edges=bins).
          * If [int, int], the number of bins in each dimension
            (nx, ny = bins).
          * If [array, array], the bin edges in each dimension
            (x_edges, y_edges = bins).
          * A combination [int, array] or [array, int], where int
            is the number of bins and array is the bin edges.

        For now, only explicit bins are supported
    dims : tuple of strings, optional
        Dimensions over which which the histogram is computed. The default is to
        compute the histogram of the flattened array.
    weights : array_like, optional
        An array of weights, of the same shape as `a`.  Each value in
        `a` only contributes its associated weight towards the bin count
        (instead of 1). If `density` is True, the weights are
        normalized, so that the integral of the density over the range
        remains 1.
    density : bool, optional
        If ``False``, the result will contain the number of samples in
        each bin. If ``True``, the result is the value of the
        probability *density* function at the bin, normalized such that
        the *integral* over the range is 1. Note that the sum of the
        histogram values will not be equal to 1 unless bins of unity
        width are chosen; it is not a probability *mass* function.
    right : bool, optional
        Indicating whether the intervals include the right or the left bin
        edge. Default behavior is (right==False) indicating that the interval
        does not include the right edge.

    Returns
    -------
    hist : array
        The values of the histogram.

    """

    N_args = len(args)

    # some sanity checks
    # TODO: replace this with a more robust function
    assert len(bins)==N_args
    for bin in bins:
        assert isinstance(bin, np.ndarray)

    args = [da.reset_coords(drop=True) for da in args]
    args = xr.align(*args, join='exact')
    a0 = args[0]
    a_dims = a0.dims

    if dims is not None:
        axis = [a0.get_axis_num(dim) for dim in dims]
        dims_to_keep = [dim for dim in a_dims if dim not in dims]
        input_core_dims = N_args*(dims,)
    else:
        axis = None
        dims_to_keep = []
        input_core_dims = (a0.dims,)

    # create output dims
    new_dims = [a.name + bin_dim_suffix for a in args]
    bin_centers = [0.5*(bin[:-1] + bin[1:]) for bin in bins]
    new_coords = {name: ((name,), bin_center, a.attrs)
                  for name, bin_center, a in zip(new_dims, bin_centers, args)}

    res = xr.apply_ufunc(_histogram, *args,
                         kwargs=dict(bins=bins, axis=axis),
                         input_core_dims=input_core_dims,
                         output_core_dims=[new_dims],
                         vectorize=False,
                         join='exact',
                         dask='allowed')
    res = res.rename('histogram')

    res.coords.update(new_coords)

    return res
