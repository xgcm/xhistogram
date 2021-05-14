"""
Numpy API for xhistogram.
"""


import numpy as np
from functools import reduce
from numpy import (
    digitize,
    bincount,
    reshape,
    ravel_multi_index,
    concatenate,
    broadcast_arrays,
)
from .duck_array_ops import _any_dask_array


def _ensure_bins_is_a_list_of_arrays(bins, N_expected):
    if len(bins) == N_expected:
        return bins
    elif N_expected == 1:
        return [bins]
    else:
        raise ValueError("Can't figure out what to do with bins.")


def _bincount_2d(bin_indices, weights, N, hist_shapes):
    # a trick to apply bincount on an axis-by-axis basis
    # https://stackoverflow.com/questions/40591754/vectorizing-numpy-bincount
    # https://stackoverflow.com/questions/40588403/vectorized-searchsorted-numpy
    M = bin_indices.shape[0]
    if weights is not None:
        weights = weights.ravel()
    bin_indices_offset = (bin_indices + (N * np.arange(M)[:, None])).ravel()
    bc_offset = bincount(bin_indices_offset, weights=weights, minlength=N * M)
    final_shape = (M,) + tuple(hist_shapes)
    return bc_offset.reshape(final_shape)


def _bincount_loop(bin_indices, weights, N, hist_shapes, block_chunks):
    M = bin_indices.shape[0]
    assert sum(block_chunks) == M
    block_counts = []
    # iterate over chunks
    bounds = np.cumsum((0,) + block_chunks)
    for m_start, m_end in zip(bounds[:-1], bounds[1:]):
        bin_indices_block = bin_indices[m_start:m_end]
        weights_block = weights[m_start:m_end] if weights is not None else None
        bc_block = _bincount_2d(bin_indices_block, weights_block, N, hist_shapes)
        block_counts.append(bc_block)
    all_counts = concatenate(block_counts)
    final_shape = (bin_indices.shape[0],) + tuple(hist_shapes)
    return all_counts.reshape(final_shape)


def _determine_block_chunks(bin_indices, block_size):
    M, N = bin_indices.shape
    if block_size is None:
        return (M,)
    if block_size == "auto":
        try:
            # dask arrays - use the pre-existing chunks
            chunks = bin_indices.chunks
            return chunks[0]
        except AttributeError:
            # automatically pick a chunk size
            # this a a heueristic without much basis
            _MAX_CHUNK_SIZE = 10_000_000
            block_size = min(_MAX_CHUNK_SIZE // N, M)
    assert isinstance(block_size, int)
    num_chunks = M // block_size
    block_chunks = num_chunks * (block_size,)
    residual = M % block_size
    if residual:
        block_chunks += (residual,)
    assert sum(block_chunks) == M
    return block_chunks


def _dispatch_bincount(bin_indices, weights, N, hist_shapes, block_size=None):
    # block_chunks is like a dask chunk, a tuple that divides up the first
    # axis of bin_indices
    block_chunks = _determine_block_chunks(bin_indices, block_size)
    if len(block_chunks) == 1:
        # single global chunk, don't need a loop over chunks
        return _bincount_2d(bin_indices, weights, N, hist_shapes)
    else:
        return _bincount_loop(bin_indices, weights, N, hist_shapes, block_chunks)


def _bincount_2d_vectorized(
    *args, bins=None, weights=None, density=False, right=False, block_size=None
):
    """Calculate the histogram independently on each row of a 2D array"""

    N_inputs = len(args)
    a0 = args[0]

    # consistency checks for inputa
    for a, b in zip(args, bins):
        assert a.ndim == 2
        assert b.ndim == 1
        assert a.shape == a0.shape
    if weights is not None:
        assert weights.shape == a0.shape

    nrows, ncols = a0.shape
    nbins = [len(b) for b in bins]
    hist_shapes = [nb + 1 for nb in nbins]

    # a marginally faster implementation would be to use searchsorted,
    # like numpy histogram itself does
    # https://github.com/numpy/numpy/blob/9c98662ee2f7daca3f9fae9d5144a9a8d3cabe8c/numpy/lib/histograms.py#L864-L882
    # for now we stick with `digitize` because it's easy to understand how it works

    # Add small increment to the last bin edge to make the final bin right-edge inclusive
    # Note, this is the approach taken by sklearn, e.g.
    # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/calibration.py#L592
    # but a better approach would be to use something like _search_sorted_inclusive() in
    # numpy histogram. This is an additional motivation for moving to searchsorted
    bins = [np.concatenate((b[:-1], b[-1:] + 1e-8)) for b in bins]

    # the maximum possible value of of digitize is nbins
    # for right=False:
    #   - 0 corresponds to a < b[0]
    #   - i corresponds to bins[i-1] <= a < b[i]
    #   - nbins corresponds to a a >= b[1]
    each_bin_indices = [digitize(a, b) for a, b in zip(args, bins)]
    # product of the bins gives the joint distribution
    if N_inputs > 1:
        bin_indices = ravel_multi_index(each_bin_indices, hist_shapes)
    else:
        bin_indices = each_bin_indices[0]
    # total number of unique bin indices
    N = reduce(lambda x, y: x * y, hist_shapes)

    bin_counts = _dispatch_bincount(
        bin_indices, weights, N, hist_shapes, block_size=block_size
    )

    # just throw out everything outside of the bins, as np.histogram does
    # TODO: make this optional?
    slices = (slice(None),) + (N_inputs * (slice(1, -1),))
    bin_counts = bin_counts[slices]

    return bin_counts


def _bincount(all_arrays, weights, axis, bins, density):

    if weights is not None:
        all_arrays.append(weights)

    all_arrays_broadcast = broadcast_arrays(*all_arrays)

    if weights is not None:
        weights_broadcast = all_arrays_broadcast.pop()
    else:
        weights_broadcast = None
    a0 = all_arrays_broadcast[0]

    do_full_array = (axis is None) or (set(axis) == set(range(a0.ndim)))
    if do_full_array:
        kept_axes_shape = None
    else:
        kept_axes_shape = tuple([a0.shape[i] for i in range(a0.ndim) if i not in axis])

    def reshape_input(a):
        if do_full_array:
            d = a.ravel()[None, :]
        else:
            # reshape the array to 2D
            # axis 0: preserved axis after histogram
            # axis 1: calculate histogram along this axis
            new_pos = tuple(range(-len(axis), 0))
            c = np.moveaxis(a, axis, new_pos)
            split_idx = c.ndim - len(axis)
            dims_0 = c.shape[:split_idx]
            assert dims_0 == kept_axes_shape
            dims_1 = c.shape[split_idx:]
            new_dim_0 = np.prod(dims_0)
            new_dim_1 = np.prod(dims_1)
            d = reshape(c, (new_dim_0, new_dim_1))
        return d

    all_arrays_reshaped = [reshape_input(a) for a in all_arrays_broadcast]

    if weights_broadcast is not None:
        weights_reshaped = reshape_input(weights_broadcast)
    else:
        weights_reshaped = None

    bin_counts = _bincount_2d_vectorized(
        *all_arrays_reshaped, bins=bins, weights=weights_reshaped, density=density
    )

    if bin_counts.shape[0] == 1:
        assert do_full_array
        bin_counts = bin_counts.squeeze()
    else:
        final_shape = kept_axes_shape + bin_counts.shape[1:]
        bin_counts = reshape(bin_counts, final_shape)

    return bin_counts


def _bincount_kernel(arr_stack, bins, axis, weights):
    # TODO: bincount function here that behaves like this:
    # Takes a stack of N chunks (where `N == len(arrs)`), returns their joint distribution (bincounts)
    # over `axes`. But leaves those axes as 1-lengh. Non-aggregated axes are left as full length, in the same position.
    # So the result has `input_ndim + len(arrs)` dimensions.
    # The way the function transforms the shape of the input array would be like:
    # (N, x, y, z, t) -> (1, 1, 1, t, bin0, bin1) for the current example here
    # Or, very formally (ignoring strs vs ints):
    # (len(arrs),) + input_inds -> (1 if d in used_inds else arr.shape[d] for d in input_inds) + (bin.size for bin in bins)
    raise NotImplementedError
    return bincounts


def histogram(
    *arrs, bins=None, axis=None, weights=None, density=False, block_size="auto"
):
    """Histogram applied along specified axis / axes.

    Parameters
    ----------
    arrs : array_like
        Input data. The number of input arguments determines the dimensonality
        of the histogram. For example, two arguments prodocue a 2D histogram.
        All args must have the same size.
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

        When bin edges are specified, all but the last (righthand-most) bin include
        the left edge and exclude the right edge. The last bin includes both edges.

        A ``TypeError`` will be raised if ``args`` contains dask arrays and
        ``bins`` are not specified explicitly as a list of arrays.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the histogram is computed. The default is to
        compute the histogram of the flattened array
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

    See Also
    --------
    numpy.histogram, numpy.bincount, numpy.digitize
    """

    all_arrays = list(arrs)

    n_inputs = len(all_arrays)
    bins = _ensure_bins_is_a_list_of_arrays(bins, n_inputs)

    if len(bins) != len(arrs):
        raise ValueError
    for b in bins:
        if b.ndim != 1:
            raise ValueError

    # Decide whether to use dask
    use_dask = _any_dask_array(weights, *all_arrays)
    if use_dask:
        import dask.array as da
        barrs = da.broadcast_arrays(*arrs)
    else:
        barrs = np.broadcast_arrays(*arrs)

    # Set up for internal histogram algorithm
    input_ndim = barrs[0].ndim
    if axis is None:
        axis = tuple(range(input_ndim))
    elif np.isscalar(axis):
        axis = (axis,)
    # TODO full axis validation & negative handling

    if weights:
        raise NotImplementedError

    if use_dask:
        # Map step: bring together `len(bins)` chunks per task, and calculate their joint distribution.
        # Leave each axis we aggregate over as 1-length.
        # We basically end up with an array with 1 _element_ per _chunk_ of the original array:
        # `bincounts.shape == input_array.numblocks + (bin.size for bin in bins)`.
        # So we end up with a per-chunk(s) bincount.
        from dask.core import flatten

        input_inds = tuple(str(i) for i in range(input_ndim))
        used_inds = tuple(str(i) for i in axis)
        # TODO handle chunks when bins are dask arrays
        # TODO handle different bin edges options (this assumes the last item in each bin is a right edge)
        new_axes = {f"bins_{i}": bin.size - 1 for i, bin in enumerate(bins)}
        out_ind = input_inds + tuple(new_axes)

        bin_counts = da.blockwise(
            _bincount_kernel,
            out_ind,
            *flatten([arr, input_inds] for arr in barrs),
            new_axes=new_axes,
            dtype=np.intp,
            adjust_chunks={i: lambda c: 1 for i in used_inds}
        )

        # TODO weights are left as an exercise to the reader

    else:
        # numpy case is as if there was only a single chunk
        stacked = np.stack(barrs)
        bin_counts = _bincount_kernel(stacked, bins, axis, weights)

    # Sum up the per-chunk bincounts over all chunks and remaining unti length axes to get result
    # In the numpy case merely removes the length-1 axes
    bin_counts = bin_counts.sum(axis=(0, *axis))  # `axis` must all be positive

    if density:
        # Normalise by dividing by bin counts and areas such that all the
        # histogram data integrated over all dimensions = 1
        bin_widths = [np.diff(b) for b in bins]
        if n_inputs == 1:
            bin_areas = bin_widths[0]
        elif n_inputs == 2:
            bin_areas = np.outer(*bin_widths)
        else:
            # Slower, but N-dimensional logic
            bin_areas = np.prod(np.ix_(*bin_widths))

        h = bin_counts / bin_areas / bin_counts.sum()
    else:
        h = bin_counts

    return h
