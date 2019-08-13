"""
Xarray API for xhistogram.
"""

import xarray as xr
import numpy as np
from collections import OrderedDict
from .core import histogram as _histogram


def histogram(*args, bins=None, dim=None, weights=None, density=False,
              block_size='auto', bin_dim_suffix='_bin',
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
    block_size : int or 'auto', optional
        A parameter which governs the algorithm used to compute the histogram.
        Using a nonzero value splits the histogram calculation over the
        non-histogram axes into blocks of size ``block_size``, iterating over
        them with a loop (numpy inputs) or in parallel (dask inputs). If
        ``'auto'``, blocks will be determined either by the underlying dask
        chunks (dask inputs) or an experimental built-in heuristic (numpy inputs).

    Returns
    -------
    hist : array
        The values of the histogram.

    """

    N_args = len(args)

    # TODO: allow list of weights as well
    N_weights = 1 if weights is not None else 0

    # some sanity checks
    # TODO: replace this with a more robust function
    assert len(bins)==N_args
    for bin in bins:
        assert isinstance(bin, np.ndarray), 'all bins must be numpy arrays'

    for a in args:
        # TODO: make this a more robust check
        assert a.name is not None, 'all arrays must have a name'

    # we drop coords to simplify alignment
    args = [da.reset_coords(drop=True) for da in args]
    if N_weights:
        args += [weights.reset_coords(drop=True)]
    # explicitly broadcast so we understand what is going into apply_ufunc
    # (apply_ufunc might be doing this by itself again)
    args = list(xr.align(*args, join='exact'))



    # what happens if we skip this?
    #args = list(xr.broadcast(*args))
    a0 = args[0]
    a_dims = a0.dims

    # roll our own broadcasting
    # now manually expand the arrays
    all_dims = [d for a in args for d in a.dims]
    all_dims_ordered = list(OrderedDict.fromkeys(all_dims))
    args_expanded = []
    for a in args:
        expand_keys = [d for d in all_dims_ordered if d not in a.dims]
        a_expanded = a.expand_dims({k: 1 for k in expand_keys})
        args_expanded.append(a_expanded)

    # only transpose if necessary, to avoid creating unnecessary dask tasks
    args_transposed = []
    for a in args_expanded:
        if a.dims != all_dims_ordered:
            args_transposed.append(a.transpose(*all_dims_ordered))
        else:
            args.transposed.append(a)
    args_data = [a.data for a in args_transposed]

    if N_weights:
        weights_data = args_data.pop()
    else:
        weights_data = None

    if dim is not None:
        dims_to_keep = [d for d in all_dims_ordered if d not in dim]
        axis = [args_transposed[0].get_axis_num(d) for d in dim]
    else:
        dims_to_keep = []
        axis = None

    h_data = _histogram(*args_data, weights=weights_data, bins=bins, axis=axis,
                        block_size=block_size)

    # create output dims
    new_dims = [a.name + bin_dim_suffix for a in args[:N_args]]
    output_dims = dims_to_keep + new_dims

    # create new coords
    bin_centers = [0.5*(bin[:-1] + bin[1:]) for bin in bins]
    new_coords = {name: ((name,), bin_center, a.attrs)
                  for name, bin_center, a in zip(new_dims, bin_centers, args)}

    old_coords = {name: a0[name]
                  for name in dims_to_keep if name in a0.coords}
    all_coords = {}
    all_coords.update(old_coords)
    all_coords.update(new_coords)

    # CF conventions tell us how to specify cell boundaries
    # http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#cell-boundaries
    # However, they require introduction of an additional dimension.
    # I don't like that.
    edge_dims = [a.name + bin_edge_suffix for a in args[:N_args]]
    edge_coords = {name: ((name,), bin_edge, a.attrs)
                  for name, bin_edge, a in zip(edge_dims, bins, args)}

    output_name = '_'.join(['histogram'] + [a.name for a in args[:N_args]])

    da_out = xr.DataArray(h_data, dims=output_dims, coords=all_coords,
                          name=output_name)
    return da_out

    # we need weights to be passed through apply_func's alignment algorithm,
    # so we include it as an arg, so we create a wrapper function to do so
    # this feels like a hack
    # def _histogram_wrapped(*args, **kwargs):
    #     alist = list(args)
    #     weights = [alist.pop() for n in range(N_weights)]
    #     if N_weights == 0:
    #         weights = None
    #     elif N_weights == 1:
    #         weights = weights[0] # squeeze
    #     return _histogram(*alist, weights=weights, **kwargs)
