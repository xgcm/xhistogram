"""
Xarray API for xhistogram.
"""

import xarray as xr
from collections import OrderedDict
from .core import histogram as _histogram

# range is a keyword so save the builtin so they can use it.
_range = range


def histogram(
    *args,
    bins=None,
    range=None,
    dim=None,
    weights=None,
    density=False,
    block_size="auto",
    keep_coords=False,
    bin_dim_suffix="_bin",
):
    """Histogram applied along specified dimensions.

    Parameters
    ----------
    args : xarray.DataArray objects
        Input data. The number of input arguments determines the dimensonality of
        the histogram. For example, two arguments prodocue a 2D histogram. All
        args must be aligned and have the same dimensions.
    bins :  int, str or numpy array or a list of ints, strs and/or arrays, optional
        If a list, there should be one entry for each item in ``args``.
        The bin specifications are as follows:

          * If int; the number of bins for all arguments in ``args``.
          * If str; the method used to automatically calculate the optimal bin width
            for all arguments in ``args``, as defined by numpy `histogram_bin_edges`.
          * If numpy array; the bin edges for all arguments in ``args``.
          * If a list of ints, strs and/or arrays; the bin specification as
            above for every argument in ``args``.

        When bin edges are specified, all but the last (righthand-most) bin include
        the left edge and exclude the right edge. The last bin includes both edges.

        A TypeError will be raised if args or weights contains dask arrays and bins
        are not specified explicitly as an array or list of arrays. This is because
        other bin specifications trigger computation.
    range : (float, float) or a list of (float, float), optional
        If a list, there should be one entry for each item in ``args``.
        The range specifications are as follows:

          * If (float, float); the lower and upper range(s) of the bins for all
            arguments in ``args``. Values outside the range are ignored. The first
            element of the range must be less than or equal to the second. `range`
            affects the automatic bin computation as well. In this case, while bin
            width is computed to be optimal based on the actual data within `range`,
            the bin count will fill the entire range including portions containing
            no data.
          * If a list of (float, float); the ranges as above for every argument in
            ``args``.
          * If not provided, range is simply ``(arg.min(), arg.max())`` for each
            arg.
    dim : tuple of strings, optional
        Dimensions over which which the histogram is computed. The default is to
        compute the histogram of the flattened array.
    weights : array_like, optional
        An array of weights, of the same shape as `a`.  Each value in
        `a` only contributes its associated weight towards the bin count
        (instead of 1). If `density` is True, the weights are
        normalized, so that the integral of the density over the range
        remains 1. NaNs in the weights input will fill the entire bin with
        NaNs. If there are NaNs in the weights input call ``.fillna(0.)``
        before running ``histogram()``.
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
    keep_coords : bool, optional
        If ``True``, keep all coordinates. Default: ``False``
    bin_dim_suffix : str, optional
        Suffix to append to input arg names to define names of output bin
        dimensions

    Returns
    -------
    hist : xarray.DataArray
        The values of the histogram. For each bin, the midpoint of the bin edges
        is given along the bin coordinates.

    """

    args = list(args)
    N_args = len(args)

    # TODO: allow list of weights as well
    N_weights = 1 if weights is not None else 0

    for a in args:
        if not isinstance(a, xr.DataArray):
            raise TypeError(
                "xhistogram.xarray.histogram accepts only xarray.DataArray "
                + f"objects but a {type(a).__name__} was provided"
            )

    for a in args:
        assert a.name is not None, "all arrays must have a name"

    # we drop coords to simplify alignment
    if not keep_coords:
        args = [da.reset_coords(drop=True) for da in args]
    if N_weights:
        args += [weights.reset_coords(drop=True)]
    # explicitly broadcast so we understand what is going into apply_ufunc
    # (apply_ufunc might be doing this by itself again)
    args = list(xr.align(*args, join="exact"))

    # what happens if we skip this?
    # args = list(xr.broadcast(*args))
    a0 = args[0]
    a_coords = a0.coords

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

    h_data, bins = _histogram(
        *args_data,
        weights=weights_data,
        bins=bins,
        range=range,
        axis=axis,
        density=density,
        block_size=block_size,
    )

    # create output dims
    new_dims = [a.name + bin_dim_suffix for a in args[:N_args]]
    output_dims = dims_to_keep + new_dims

    # create new coords
    bin_centers = [0.5 * (bin[:-1] + bin[1:]) for bin in bins]
    new_coords = {
        name: ((name,), bin_center, a.attrs)
        for name, bin_center, a in zip(new_dims, bin_centers, args)
    }

    # old coords associated with dims
    old_dim_coords = {name: a0[name] for name in dims_to_keep if name in a_coords}

    all_coords = {}
    all_coords.update(old_dim_coords)
    all_coords.update(new_coords)
    # add compatible coords
    if keep_coords:
        for c in a_coords:
            if c not in all_coords and set(a0[c].dims).issubset(output_dims):
                all_coords[c] = a0[c]

    output_name = "_".join(["histogram"] + [a.name for a in args[:N_args]])

    da_out = xr.DataArray(h_data, dims=output_dims, coords=all_coords, name=output_name)

    return da_out

    # we need weights to be passed through apply_func's alignment algorithm,
    # so we include it as an arg, so we create a wrapper function to do so
    # this feels like a hack
    # def _histogram_wrapped(*args, **kwargs):
    #     alist = list(args)
    #     weights = [alist.pop() for n in _range(N_weights)]
    #     if N_weights == 0:
    #         weights = None
    #     elif N_weights == 1:
    #         weights = weights[0] # squeeze
    #     return _histogram(*alist, weights=weights, **kwargs)
