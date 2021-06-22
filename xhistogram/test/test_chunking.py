import numpy as np
import pytest

from .fixtures import example_dataarray
from ..xarray import histogram


@pytest.mark.parametrize("weights", [False, True])
@pytest.mark.parametrize("chunksize", [1, 2, 3, 10])
@pytest.mark.parametrize("shape", [(10,), (10, 4)])
def test_chunked_weights(chunksize, shape, weights):

    data_a = example_dataarray(shape).chunk((chunksize,))

    if weights:
        weights = example_dataarray(shape).chunk((chunksize,))
        weights_arr = weights.values
    else:
        weights = weights_arr = None

    nbins_a = 6
    bins_a = np.linspace(-4, 4, nbins_a + 1)

    h = histogram(data_a, bins=[bins_a], weights=weights)

    assert h.shape == (nbins_a,)

    hist, _ = np.histogram(data_a.values, bins=bins_a, weights=weights_arr)

    np.testing.assert_allclose(hist, h.values)


@pytest.mark.parametrize("xchunksize", [1, 2, 3, 10])
@pytest.mark.parametrize("ychunksize", [1, 2, 3, 12])
class TestFixedSize2DChunks:
    def test_2d_chunks(self, xchunksize, ychunksize):

        data_a = example_dataarray(shape=(10, 12)).chunk((xchunksize, ychunksize))

        nbins_a = 8
        bins_a = np.linspace(-4, 4, nbins_a + 1)

        h = histogram(data_a, bins=[bins_a])

        assert h.shape == (nbins_a,)

        hist, _ = np.histogram(data_a.values, bins=bins_a)

        np.testing.assert_allclose(hist, h.values)

    @pytest.mark.parametrize("reduce_dim", ["dim_0", "dim_1"])
    def test_2d_chunks_broadcast_dim(
        self,
        xchunksize,
        ychunksize,
        reduce_dim,
    ):
        data_a = example_dataarray(shape=(10, 12)).chunk((xchunksize, ychunksize))
        dims = list(data_a.dims)
        broadcast_dim = [d for d in dims if d != reduce_dim][0]

        nbins_a = 8
        bins_a = np.linspace(-4, 4, nbins_a + 1)

        h = histogram(data_a, bins=[bins_a], dim=(reduce_dim,))

        assert h.shape == (data_a.sizes[broadcast_dim], nbins_a)

        def _np_hist(*args, **kwargs):
            h, _ = np.histogram(*args, **kwargs)
            return h

        hist = np.apply_along_axis(
            _np_hist, axis=dims.index(reduce_dim), arr=data_a.values, bins=bins_a
        )

        if reduce_dim == "dim_0":
            h = h.transpose()
        np.testing.assert_allclose(hist, h.values)

    def test_2d_chunks_2d_hist(self, xchunksize, ychunksize):

        data_a = example_dataarray(shape=(10, 12)).chunk((xchunksize, ychunksize))
        data_b = example_dataarray(shape=(10, 12)).chunk((xchunksize, ychunksize))

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


@pytest.mark.parametrize("xchunksize", [1, 2, 3, 10])
@pytest.mark.parametrize("ychunksize", [1, 2, 3, 12])
class TestUnalignedChunks:
    def test_unaligned_data_chunks(self, xchunksize, ychunksize):
        data_a = example_dataarray(shape=(10, 12)).chunk((xchunksize, ychunksize))
        data_b = example_dataarray(shape=(10, 12)).chunk(
            (xchunksize + 1, ychunksize + 1)
        )

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

    def test_unaligned_weights_chunks(self, xchunksize, ychunksize):

        data_a = example_dataarray(shape=(10, 12)).chunk((xchunksize, ychunksize))
        weights = example_dataarray(shape=(10, 12)).chunk(
            (xchunksize + 1, ychunksize + 1)
        )

        nbins_a = 8
        bins_a = np.linspace(-4, 4, nbins_a + 1)

        h = histogram(data_a, bins=[bins_a], weights=weights)

        assert h.shape == (nbins_a,)

        hist, _ = np.histogram(data_a.values, bins=bins_a, weights=weights.values)

        np.testing.assert_allclose(hist, h.values)
