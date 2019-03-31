"""Numpy functions for vectorized histograms."""

import numpy as np

def _histogram_2d_vectorized(a, bins, weights=None, density=False, right=False):
    """Calculate the histogram independently on each row of a 2D array"""
    assert a.ndim == 2
    assert bins.ndim == 1
    if weights is not None:
        assert weights.shape == a.shape
        weights = weights.ravel()

    nrows, ncols = a.shape
    nbins = len(bins)

    # the maximum possible value of of bin_indices is nbins
    # for right=False:
    #   - 0 corresponds to a < bins[0]
    #   - i corresponds to bins[i-1] <= a < bins[i]
    #   - nbins corresponds to a a >= bins[1]
    bin_indices = np.digitize(a, bins)

    # now the tricks to apply bincount on an axis-by-axis basis
    # https://stackoverflow.com/questions/40591754/vectorizing-numpy-bincount
    # https://stackoverflow.com/questions/40588403/vectorized-searchsorted-numpy

    M = nrows
    N = nbins+1
    bin_indices_offset = (bin_indices + (N * np.arange(M)[:, None])).ravel()
    bc_offset = np.bincount(bin_indices_offset, weights=weights,
                            minlength=N*M)
    bc_offset_reshape = bc_offset.reshape(M, -1)

    # just throw out everything outside of the bins, as np.histogram does
    # TODO: make this optional?
    return bc_offset_reshape[:, 1:-1]
