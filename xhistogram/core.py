"""
Numpy API for xhistogram.
"""


import dask
import numpy as np
from functools import reduce
from collections.abc import Iterable
from numpy import (
    searchsorted,
    bincount,
    reshape,
    ravel_multi_index,
    concatenate,
    broadcast_arrays,
)

# range is a keyword so save the builtin so they can use it.
_range = range

try:
    import dask.array as dsa

    has_dask = True
except ImportError:
    has_dask = False


def _any_dask_array(*args):
    if not has_dask:
        return False
    else:
        return any(isinstance(a, dsa.core.Array) for a in args)


def _ensure_correctly_formatted_bins(bins, N_expected):
    # TODO: This could be done better / more robustly
    if bins is None:
        raise ValueError("bins must be provided")
    if isinstance(bins, (int, str, np.ndarray)):
        bins = N_expected * [bins]
    if len(bins) == N_expected:
        return bins
    else:
        raise ValueError(
            "The number of bin definitions doesn't match the number of args"
        )


def _ensure_correctly_formatted_range(range_, N_expected):
    # TODO: This could be done better / more robustly
    def _iterable_nested(x):
        return all(isinstance(i, Iterable) for i in x)

    if range_ is not None:
        if (len(range_) == 2) & (not _iterable_nested(range_)):
            return N_expected * [range_]
        elif N_expected == len(range_):
            if all(len(x) == 2 for x in range_):
                return range_
            else:
                raise ValueError(
                    "range should be provided as (lower_range, upper_range). In the "
                    + "case of multiple args, range should be a list of such tuples"
                )
        else:
            raise ValueError("The number of ranges doesn't match the number of args")
    else:
        return N_expected * [range_]


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

    # The maximum possible value of searchsorted is nbins
    # For _searchsorted_inclusive:
    #   - 0 corresponds to a < b[0]
    #   - i corresponds to b[i-1] <= a < b[i]
    #   - nbins-1 corresponds to b[-2] <= a <= b[-1]
    #   - nbins corresponds to a >= b[-1]
    def _searchsorted_inclusive(a, b):
        """
        Like `searchsorted`, but where the last bin is also right-edge inclusive.
        """
        # Similar to implementation in np.histogramdd
        # see https://github.com/numpy/numpy/blob/9c98662ee2f7daca3f9fae9d5144a9a8d3cabe8c/numpy/lib/histograms.py#L1056
        # This assumes the bins (b) are sorted
        bin_indices = searchsorted(b, a, side="right")
        on_edge = a == b[-1]
        # Shift these points one bin to the left.
        bin_indices[on_edge] -= 1
        return bin_indices

    each_bin_indices = [_searchsorted_inclusive(a, b) for a, b in zip(args, bins)]
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


def _bincount(
    *all_arrays, weights=False, axis=None, bins=None, density=None, block_size=None
):
    a0 = all_arrays[0]

    do_full_array = (axis is None) or (set(axis) == set(_range(a0.ndim)))

    if do_full_array:
        kept_axes_shape = (1,) * a0.ndim
    else:
        kept_axes_shape = tuple(
            [a0.shape[i] if i not in axis else 1 for i in _range(a0.ndim)]
        )

    def reshape_input(a):
        if do_full_array:
            d = a.ravel()[None, :]
        else:
            # reshape the array to 2D
            # axis 0: preserved axis after histogram
            # axis 1: calculate histogram along this axis
            new_pos = tuple(_range(-len(axis), 0))
            c = np.moveaxis(a, axis, new_pos)
            split_idx = c.ndim - len(axis)
            dims_0 = c.shape[:split_idx]
            # assert dims_0 == kept_axes_shape
            dims_1 = c.shape[split_idx:]
            new_dim_0 = np.prod(dims_0)
            new_dim_1 = np.prod(dims_1)
            d = reshape(c, (new_dim_0, new_dim_1))
        return d

    all_arrays_reshaped = [reshape_input(a) for a in all_arrays]

    if weights:
        weights_array = all_arrays_reshaped.pop()
    else:
        weights_array = None

    bin_counts = _bincount_2d_vectorized(
        *all_arrays_reshaped,
        bins=bins,
        weights=weights_array,
        density=density,
        block_size=block_size,
    )

    final_shape = kept_axes_shape + bin_counts.shape[1:]
    bin_counts = reshape(bin_counts, final_shape)

    return bin_counts


def histogram(
    *args,
    bins=None,
    range=None,
    axis=None,
    weights=None,
    density=False,
    block_size="auto",
):
    """Histogram applied along specified axis / axes.

    Parameters
    ----------
    args : array_like
        Input data. The number of input arguments determines the dimensionality
        of the histogram. For example, two arguments produce a 2D histogram.
        All args must have the same size.
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
    bin_edges : list of arrays
        Return the bin edges for each input array.

    See Also
    --------
    numpy.histogram, numpy.bincount, numpy.searchsorted
    """

    a0 = args[0]
    ndim = a0.ndim
    n_inputs = len(args)

    is_dask_array = any([dask.is_dask_collection(a) for a in list(args) + [weights]])

    if axis is not None:
        axis = np.atleast_1d(axis)
        assert axis.ndim == 1
        axis_normed = []
        for ax in axis:
            if ax >= 0:
                ax_positive = ax
            else:
                ax_positive = ndim + ax
            assert ax_positive < ndim, "axis must be less than ndim"
            axis_normed.append(ax_positive)
        axis = [int(i) for i in axis_normed]

    all_arrays = list(args)
    n_inputs = len(all_arrays)

    if weights is not None:
        all_arrays.append(weights)
        has_weights = True
    else:
        has_weights = False

    dtype = "i8" if not has_weights else weights.dtype

    # Broadcast input arrays. Note that this dispatches to `dsa.broadcast_arrays` as necessary.
    all_arrays = broadcast_arrays(*all_arrays)
    # Since all arrays now have the same shape, just get the axes of the first.
    input_axes = tuple(_range(all_arrays[0].ndim))

    # Some sanity checks and format bins and range correctly
    bins = _ensure_correctly_formatted_bins(bins, n_inputs)
    range = _ensure_correctly_formatted_range(range, n_inputs)

    # histogram_bin_edges triggers computation on dask arrays. It would be possible
    # to write a version of this that doesn't trigger when `range` is provided, but
    # for now let's just use np.histogram_bin_edges
    if is_dask_array:
        if not all(isinstance(b, np.ndarray) for b in bins):
            raise TypeError(
                "When using dask arrays, bins must be provided as numpy array(s) of edges"
            )
    else:
        bins = [
            np.histogram_bin_edges(
                a, bins=b, range=r, weights=all_arrays[-1] if has_weights else None
            )
            for a, b, r in zip(all_arrays, bins, range)
        ]
    bincount_kwargs = dict(
        weights=has_weights,
        axis=axis,
        bins=bins,
        density=density,
        block_size=block_size,
    )

    # remove these axes from the inputs
    if axis is not None:
        drop_axes = tuple(axis)
    else:
        drop_axes = input_axes

    if _any_dask_array(weights, *all_arrays):
        # We should be able to just apply the bin_count function to every
        # block and then sum over all blocks to get the total bin count.
        # The main challenge is to figure out the chunk shape that will come
        # out of _bincount. We might also need to add dummy dimensions to sum
        # over in the _bincount function
        import dask.array as dsa

        # Important note from blockwise docs
        # > Any index, like i missing from the output index is interpreted as a contraction...
        # > In the case of a contraction the passed function should expect an iterable of blocks
        # > on any array that holds that index.
        # This means that we need to have all the input indexes present in the output index
        # However, they will be reduced to singleton (len 1) dimensions

        adjust_chunks = {i: (lambda x: 1) for i in drop_axes}

        new_axes_start = max(input_axes) + 1
        new_axes = {new_axes_start + i: len(bin) - 1 for i, bin in enumerate(bins)}
        out_index = input_axes + tuple(new_axes)

        blockwise_args = []
        for arg in all_arrays:
            blockwise_args.append(arg)
            blockwise_args.append(input_axes)

        bin_counts = dsa.blockwise(
            _bincount,
            out_index,
            *blockwise_args,
            new_axes=new_axes,
            adjust_chunks=adjust_chunks,
            meta=np.array((), dtype),
            **bincount_kwargs,
        )
        # sum over the block dims
        bin_counts = bin_counts.sum(drop_axes)
    else:
        # drop the extra axis used for summing over blocks
        bin_counts = _bincount(*all_arrays, **bincount_kwargs).squeeze(drop_axes)

    if density:
        # Normalize by dividing by bin counts and areas such that all the
        # histogram data integrated over all dimensions = 1
        bin_widths = [np.diff(b) for b in bins]
        if n_inputs == 1:
            bin_areas = bin_widths[0]
        elif n_inputs == 2:
            bin_areas = np.outer(*bin_widths)
        else:
            # Slower, but N-dimensional logic
            bin_areas = np.prod(np.ix_(*bin_widths))

        # Sum over the last n_inputs axes, which correspond to the bins. All other axes
        # are "bystander" axes. Sums must be done independently for each bystander axes
        # so that nans are dealt with correctly (#51)
        bin_axes = tuple(_range(-n_inputs, 0))
        bin_count_sums = bin_counts.sum(axis=bin_axes)
        bin_count_sums_shape = bin_count_sums.shape + len(bin_axes) * (1,)
        h = bin_counts / bin_areas / reshape(bin_count_sums, bin_count_sums_shape)
    else:
        h = bin_counts

    return h, bins
