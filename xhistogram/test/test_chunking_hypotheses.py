import numpy as np
import pytest

from .fixtures import dataarray_factory, dataset_factory
from ..xarray import histogram

pytest.importorskip("hypothesis")

import hypothesis.strategies as st
from hypothesis import given


@st.composite
def chunk_shapes(draw, ndim=3, max_arr_len=10):
    """Generate different chunking patterns for an N-D array of data."""
    chunks = []
    for n in range(ndim):
        shape = draw(st.integers(min_value=1, max_value=max_arr_len))
        chunks.append(shape)
    return tuple(chunks)


class TestChunkingHypotheses:
    @given(chunk_shapes(ndim=1, max_arr_len=20))
    def test_all_chunking_patterns_1d(self, dataarray_factory, chunks):

        data = dataarray_factory(shape=(20,)).chunk(chunks)

        nbins_a = 8
        bins = np.linspace(-4, 4, nbins_a + 1)

        h = histogram(data, bins=[bins])

        assert h.shape == (nbins_a,)

        hist, _ = np.histogram(
            data.values.ravel(),
            bins=bins,
        )

        np.testing.assert_allclose(hist, h)

    # TODO mark as slow?
    @given(chunk_shapes(ndim=2, max_arr_len=8))
    def test_all_chunking_patterns_2d(self, dataarray_factory, chunks):

        data_a = dataarray_factory(shape=(5, 20)).chunk(chunks)
        data_b = dataarray_factory(shape=(5, 20)).chunk(chunks)

        nbins_a = 8
        nbins_b = 9
        bins_a = np.linspace(-4, 4, nbins_a + 1)
        bins_b = np.linspace(-4, 4, nbins_b + 1)

        h = histogram(data_a, data_b, bins=[bins_a, bins_b])

        assert h.shape == (nbins_a, nbins_b)

        hist, _, _ = np.histogram2d(
            data_a.values.ravel(),
            data_b.values.ravel(),
            bins=[bins_a, bins_b],
        )

        np.testing.assert_allclose(hist, h.values)

    # TODO mark as slow?
    @pytest.mark.parametrize("n_vars", [1, 2, 3, 4])
    @given(chunk_shapes(ndim=2, max_arr_len=7))
    def test_all_chunking_patterns_dd_hist(self, dataset_factory, n_vars, chunk_shapes):
        ds = dataset_factory(ndim=2, n_vars=n_vars)
        ds = ds.chunk({d: c for d, c in zip(ds.dims.keys(), chunk_shapes)})

        n_bins = (7, 8, 9, 10)[:n_vars]
        bins = [np.linspace(-4, 4, n + 1) for n in n_bins]

        h = histogram(*[da for name, da in ds.data_vars.items()], bins=bins)

        assert h.shape == n_bins

        input_data = np.stack(
            [da.values.ravel() for name, da in ds.data_vars.items()], axis=-1
        )
        hist, _ = np.histogramdd(input_data, bins=bins)

        np.testing.assert_allclose(hist, h.values)
