"""
Xarray API for xhistogram.
"""

import xarray as xr
import numpy as np
from .core import histogram as _histogram


def histogram(*args, bins=None, dim=None, weights=None, density=False,
              block_size=None, bin_dim_suffix='_bin',
              bin_edge_suffix='_bin_edge'):
    """Histogram applied along specified dimensions.

    Parameters
    ----------
    args : xarray.DataArray objects
        Input data. The number of input arguments determines the dimensonality of
        the histogram. For example, two arguments prodocue a 2D histogram. All
        args must be aligned and have the same dimensions.
    bins :  int or array_like or a list of ints or arrays, optional
        If a list, there should be one entry for each item in ``args``.
        The bin specification:

          * If int, the number of bins for all arguments in ``args``.
          * If array_like, the bin edges for all arguments in ``args``.
          * If a list of ints, the number of bins  for every argument in ``args``.
          * If a list arrays, the bin edges for each argument in ``args``
            (required format for Dask inputs).
          * A combination [int, array] or [array, int], where int
            is the number of bins and array is the bin edges.

        A ``TypeError`` will be raised if ``args`` contains dask arrays and
        ``bins`` are not specified explicitly as a list of arrays.
    dim : tuple of strings, optional
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
    block_size : int, optional
        A parameter which governs the algorithm used to compute the histogram.
        Using a nonzero value splits the histogram calculation over the
        non-histogram dims into blocks of size ``block_size``, iterating over
        them with a loop (numpy inputs) or in parallel (dask inputs).

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
        assert isinstance(bin, np.ndarray), 'all bins must be numpy arrays'

    # we drop coords to simplify alignment
    args = [da.reset_coords(drop=True) for da in args]
    if weights is not None:
        args += [weights]
    # explicitly align just so we understand what is going into apply_ufunc
    # (apply_ufunc might be doing this by itself again)
    args = xr.align(*args, join='exact')
    if weights is not None:
        weights = args.pop()
    a0 = args[0]
    a_dims = a0.dims

    if dim is not None:
        dims_to_keep = [d for d in a_dims if d not in dim]
        input_core_dims = N_args*(dim,)
        # "Core dimensions are assumed to appear as the last dimensions of each
        # output in the provided order." - xarray docs
        # Therefore, we need to tell xhistogram to operate on the final axes
        axis = tuple(range(-1, -len(dim) - 1, -1))
    else:
        axis = None
        dims_to_keep = []
        input_core_dims = (a0.dims,)

    for a in args:
        # TODO: make this a more robust check
        assert a.name is not None, 'all arrays must have a name'

    # create output dims
    new_dims = [a.name + bin_dim_suffix for a in args]
    bin_centers = [0.5*(bin[:-1] + bin[1:]) for bin in bins]
    new_coords = {name: ((name,), bin_center, a.attrs)
                  for name, bin_center, a in zip(new_dims, bin_centers, args)}

    # CF conventions tell us how to specify cell boundaries
    # http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#cell-boundaries
    # However, they require introduction of an additional dimension.
    # I don't like that.
    edge_dims = [a.name + bin_edge_suffix for a in args]
    edge_coords = {name: ((name,), bin_edge, a.attrs)
                  for name, bin_edge, a in zip(edge_dims, bins, args)}

    histogram_kwargs = dict(bins=bins, axis=axis, block_size=block_size)
    res = xr.apply_ufunc(_histogram, *args,
                         kwargs=histogram_kwargs,
                         input_core_dims=input_core_dims,
                         output_core_dims=[new_dims],
                         vectorize=False,
                         join='exact',
                         dask='allowed')
    res = res.rename('histogram')

    res.coords.update(new_coords)

    # raises ValueError: cannot add coordinates with new dimensions to a DataArray
    #res.coords.update(edge_coords)

    return res
