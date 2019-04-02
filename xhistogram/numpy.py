"""Numpy functions for vectorized histograms."""

import numpy as np
from .duck_array_ops import digitize, bincount

def _histogram_2d_vectorized(a, bins, weights=None, density=False, right=False):
    """Calculate the histogram independently on each row of a 2D array"""
    assert a.ndim == 2
    assert bins.ndim == 1
    if weights is not None:
        assert weights.shape == a.shape
        weights = weights.ravel()

    nrows, ncols = a.shape
    nbins = len(bins)

    # a marginally faster implementation would be to use searchsorted,
    # like numpy histogram itself does
    # https://github.com/numpy/numpy/blob/9c98662ee2f7daca3f9fae9d5144a9a8d3cabe8c/numpy/lib/histograms.py#L864-L882
    # for now we stick with `digitize` because it's easy to understand how it works

    # the maximum possible value of of bin_indices is nbins
    # for right=False:
    #   - 0 corresponds to a < bins[0]
    #   - i corresponds to bins[i-1] <= a < bins[i]
    #   - nbins corresponds to a a >= bins[1]
    bin_indices = digitize(a, bins)

    # now the tricks to apply bincount on an axis-by-axis basis
    # https://stackoverflow.com/questions/40591754/vectorizing-numpy-bincount
    # https://stackoverflow.com/questions/40588403/vectorized-searchsorted-numpy

    M = nrows
    N = nbins+1
    bin_indices_offset = (bin_indices + (N * np.arange(M)[:, None])).ravel()
    bc_offset = bincount(bin_indices_offset, weights=weights,
                            minlength=N*M)
    bc_offset_reshape = bc_offset.reshape(M, -1)

    # just throw out everything outside of the bins, as np.histogram does
    # TODO: make this optional?
    return bc_offset_reshape[:, 1:-1]


def histogram(a, bins, axis=None, weights=None, density=False, right=False):
    """Histogram applied along specified axis / axes.

    Parameters
    ----------
    a : array_like
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

    keep_shape = axis is None or (len(axis) == a.ndim)
    if keep_shape:
        # should be the same thing
        #f, _ = np.histogram(a, bins=bins, weights=weights, density=density)
        d = a.ravel()[None, :]
    else:
        new_pos = tuple(range(-len(axis), 0))
        c = np.moveaxis(a, axis, new_pos)
        split_idx = c.ndim - len(axis)
        dims_0 = c.shape[:split_idx]
        dims_1 = c.shape[split_idx:]
        new_dim_0 = np.prod(dims_0)
        new_dim_1 = np.prod(dims_1)
        d = np.reshape(c, (new_dim_0, new_dim_1))
    e = _histogram_2d_vectorized(d, bins, weights=weights, density=density,
                                 right=right)

    if keep_shape:
        f = e.squeeze()
    else:
        final_shape = dims_0 + (-1,)
        f = np.reshape(e, final_shape)

    return f
