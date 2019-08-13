import xarray as xr
import numpy as np
import pytest
import pandas as pd
from functools import reduce
from itertools import combinations

from ..xarray import histogram


# example dimensions
DIMS = {'time': 5, 'depth': 10, 'lat': 45, 'lon': 90}
COORDS = {'time': ('time', pd.date_range(start='2000-01-01', periods=DIMS['time'])),
          'depth': ('depth', np.arange(DIMS['depth']) * 100. + 50),
          'lat': ('lat', np.arange(DIMS['lat']) * 180/DIMS['lat'] - 90 + 90/DIMS['lat']),
          'lon': ('lon', np.arange(DIMS['lon']) * 360/DIMS['lon'] + 180/DIMS['lon'])
          }


@pytest.fixture(params=[('lon',),
                        ('lat', 'lon'),
                        ('depth', 'lat', 'lon'),
                        ('time', 'depth', 'lat', 'lon')],
                ids=['1D', '2D', '3D', '4D'])
def ones(request):
    dims = request.param
    shape = [DIMS[d] for d in dims]
    coords = {k: v for k, v in COORDS.items() if k in dims}
    data = np.ones(shape, dtype='f8')
    da = xr.DataArray(data, dims=dims, coords=coords, name='ones')
    return da


@pytest.mark.parametrize('ndims', [1, 2, 3, 4])
def test_histogram_ones(ones, ndims):
    dims = ones.dims
    if ones.ndim < ndims:
        pytest.skip("Don't need to test when number of dimension combinations "
                    "exceeds the number of array dimensions")

    # everything should be in the middle bin (index 1)
    bins = np.array([0, 0.9, 1.1, 2])
    bins_c = 0.5 * (bins[1:] + bins[:-1])

    def _check_result(h, d):
        other_dims = [dim for dim in ones.dims if dim not in d]
        if len(other_dims) > 0:
            assert set(other_dims) <= set(h.dims)
        # check that all values are in the central bin
        h_sum = h.sum(other_dims)
        h_sum_expected = xr.DataArray([0, ones.size, 0],
                                      dims=['ones_bin'],
                                      coords={'ones_bin': ('ones_bin', bins_c)},
                                      name='histogram_ones')
        xr.testing.assert_identical(h_sum, h_sum_expected)

    for d in combinations(dims, ndims):
        h = histogram(ones, bins=[bins], dim=d)
        _check_result(h, d)


# TODO: refactor this test to use better fixtures
# (it currently has a ton of loops)
@pytest.mark.parametrize('ndims', [1, 2, 3, 4])
def test_weights(ones, ndims):
    dims = ones.dims
    if ones.ndim < ndims:
        pytest.skip("Don't need to test when number of dimension combinations "
                    "exceeds the number of array dimensions")

    bins = np.array([0, 0.9, 1.1, 2])
    bins_c = 0.5 * (bins[1:] + bins[:-1])

    weight_value = 0.5

    def _check_result(h, d):
        other_dims = [dim for dim in ones.dims if dim not in d]
        if len(other_dims) > 0:
            assert set(other_dims) <= set(h.dims)
        # check that all values are in the central bin
        h_sum = h.sum(other_dims)
        h_sum_expected = xr.DataArray([0, weight_value * ones.size, 0],
                                      dims=['ones_bin'],
                                      coords={'ones_bin': ('ones_bin', bins_c)},
                                      name='histogram_ones')
        xr.testing.assert_identical(h_sum, h_sum_expected)

    # get every possible combination of sub-dimensions
    for n_combinations in range(ones.ndim):
        for weight_dims in combinations(dims, n_combinations):
            i_selector = {dim: 0 for dim in weight_dims}
            weights = xr.full_like(ones.isel(**i_selector), weight_value)
            for nc in range(ndims):
                for d in combinations(dims, nc+1):
                    h = histogram(ones, weights=weights, bins=[bins], dim=d)
                    _check_result(h, d)


# test for issue #5
def test_dims_and_coords():
    time_axis = np.arange(4)
    depth_axis = np.arange(10)
    X_axis = np.arange(30)
    Y_axis = np.arange(30)

    dat1 = np.random.randint(low=0, high=100,
                             size=(len(time_axis), len(depth_axis),
                                   len(X_axis), len(Y_axis)))
    array1 = xr.DataArray(dat1, coords=[time_axis,depth_axis,X_axis,Y_axis],
                          dims=['time', 'depth', 'X', 'Y'], name='one')

    dat2 = np.random.randint(low=0, high=50,
                             size=(len(time_axis), len(depth_axis),
                                   len(X_axis), len(Y_axis)))
    array2 = xr.DataArray(dat2, coords=[time_axis,depth_axis,X_axis,Y_axis],
                          dims=['time','depth','X','Y'], name='two')

    bins1 = np.linspace(0, 100, 50)
    bins2 = np.linspace(0,50,25)

    result = histogram(array1,array2,dim = ['X','Y'] , bins=[bins1,bins2])
    assert result.dims == ('time', 'depth', 'one_bin', 'two_bin')
    assert result.time.identical(array1.time)
    assert result.depth.identical(array2.depth)
