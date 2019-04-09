"""
Numpy API for xhistogram.
"""


import numpy as np
from functools import reduce
from .duck_array_ops import digitize, bincount, ravel_multi_index


def _ensure_bins_is_a_list_of_arrays(bins, N_expected):
    if len(bins) == N_expected:
        return bins
    elif N_expected==1:
        return [bins]
    else:
        raise ValueError("Can't figure out what to do with bins.")


def _histogram_2d_vectorized(*args, bins=None, weights=None, density=False, right=False):
    """Calculate the histogram independently on each row of a 2D array"""

    N_inputs = len(args)
    bins = _ensure_bins_is_a_list_of_arrays(bins, N_inputs)
    a0 = args[0]

    # consistency checks for inputa
    for a, b in zip(args, bins):
        assert a.ndim == 2
        assert b.ndim == 1
        assert a.shape == a0.shape
    if weights is not None:
        assert weights.shape == a0.shape
        weights = weights.ravel()

    nrows, ncols = a0.shape
    nbins = [len(b) for b in bins]
    hist_shapes = [nb+1 for nb in nbins]
    final_shape = (nrows,) + tuple(hist_shapes)

    # a marginally faster implementation would be to use searchsorted,
    # like numpy histogram itself does
    # https://github.com/numpy/numpy/blob/9c98662ee2f7daca3f9fae9d5144a9a8d3cabe8c/numpy/lib/histograms.py#L864-L882
    # for now we stick with `digitize` because it's easy to understand how it works

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
    N = reduce(lambda x, y: x*y, hist_shapes)

    # now the tricks to apply bincount on an axis-by-axis basis
    # https://stackoverflow.com/questions/40591754/vectorizing-numpy-bincount
    # https://stackoverflow.com/questions/40588403/vectorized-searchsorted-numpy

    M = nrows
    bin_indices_offset = (bin_indices + (N * np.arange(M)[:, None])).ravel()
    bc_offset = bincount(bin_indices_offset, weights=weights,
                            minlength=N*M)

    bc_offset_reshape = bc_offset.reshape(final_shape)

    # just throw out everything outside of the bins, as np.histogram does
    # TODO: make this optional?
    slices = (slice(None),) + (N_inputs * (slice(1, -1),))
    return bc_offset_reshape[slices]


def histogram(*args, bins=None, axis=None, weights=None, density=False, right=False):
    """Histogram applied along specified axis / axes.

    Parameters
    ----------
    args : array_like
        Input data. The histogram is computed over the specified axes.
    bins : array_like
        Bin edges. Must be specified explicitly. The size of the output
        dimension will be ``len(bins) - 1``.
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
    right : bool, optional
        Indicating whether the intervals include the right or the left bin
        edge. Default behavior is (right==False) indicating that the interval
        does not include the right edge.

    Returns
    -------
    hist : array
        The values of the histogram.

    See Also
    --------
    bincount, histogram, digitize
    """

    if axis is not None:
        axis = np.atleast_1d(axis)
        assert axis.ndim == 1

    a0 = args[0]
    do_full_array = (axis is None) or (set(axis) == set(range(a0.ndim)))
    if do_full_array:
        kept_axes_shape = None
    else:
        kept_axes_shape = tuple([a0.shape[i]
                                 for i in range(a0.ndim) if i not in axis])

    def reshape_input(a):
        if do_full_array:
            # should be the same thing
            #f, _ = np.histogram(a, bins=bins, weights=weights, density=density)
            d = a.ravel()[None, :]
        else:
            new_pos = tuple(range(-len(axis), 0))
            c = np.moveaxis(a, axis, new_pos)
            split_idx = c.ndim - len(axis)
            dims_0 = c.shape[:split_idx]
            assert dims_0 == kept_axes_shape
            dims_1 = c.shape[split_idx:]
            new_dim_0 = np.prod(dims_0)
            new_dim_1 = np.prod(dims_1)
            d = np.reshape(c, (new_dim_0, new_dim_1))
        return d

    args_reshaped = [reshape_input(a) for a in args]
    if weights is not None:
        weights = reshape_input(weights)

    h = _histogram_2d_vectorized(*args_reshaped, bins=bins, weights=weights,
                                 density=density, right=right)

    if h.shape[0] == 1:
        assert do_full_array
        h = h.squeeze()
    else:
        final_shape = kept_axes_shape + h.shape[1:]
        h = np.reshape(h, final_shape)

    return h
