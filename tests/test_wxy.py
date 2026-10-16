import numpy as np
import pytest

import wxy

STERE = "+proj=stere +lat_0=-90 +lat_ts=-71 +datum=WGS84"
GLOBAL = (-180, 180, -90, 90)


def lat_grid():
    """A 1-degree global grid whose value is the cell-centre latitude."""
    lat = np.arange(89.5, -90, -1.0)
    return np.repeat(lat[:, None], 360, axis=1)


def test_affine():
    assert wxy.affine((0, 10, 0, 5), (10, 5)) == (0, 1, 0, 5, 0, -1)
    with pytest.raises(ValueError):
        wxy.affine((1, 2, 3), (1, 1))
    with pytest.raises(ValueError):
        wxy.affine((0, 1, 0, 1), (0, 1))


def test_regular():
    # cell centres of a 1-degree global grid, as a netCDF file might store them
    lon = np.arange(-179.5, 180, 1.0)
    r = wxy.regular(lon)
    assert r["regular"] and r["origin"] == -180 and r["step"] == 1

    # a float32-sized wobble is regular at float32 tolerance, not float64
    lon32 = lon.copy()
    lon32[1] += 1e-6
    assert wxy.regular(lon32)["regular"]
    assert not wxy.regular(lon32, tolerance="float64")["regular"]

    # a stretched axis is not
    assert not wxy.regular([0, 1, 2, 4, 8])["regular"]

    # edges: n + 1 values
    e = wxy.regular(np.arange(0, 11, 2.0), cell="edge")
    assert e["regular"] and e["origin"] == 0 and e["step"] == 2

    with pytest.raises(ValueError):
        wxy.regular(lon, tolerance="float16")


def test_warp_identity():
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    w = wxy.warp(m, (0, 2, 0, 2), "EPSG:4326", (0, 2, 0, 2), (2, 2), "EPSG:4326")
    np.testing.assert_array_equal(w, m)

    # the north-west quarter is the first value
    w1 = wxy.warp(m, (0, 2, 0, 2), "EPSG:4326", (0, 1, 1, 2), (1, 1), "EPSG:4326")
    assert w1.shape == (1, 1) and w1[0, 0] == 1


def test_warp_layout_and_dtype():
    # a Fortran-ordered integer array warps the same as a C-ordered double
    m = np.arange(12).reshape(3, 4)
    a = wxy.warp(m, (0, 4, 0, 3), "EPSG:4326", (0, 4, 0, 3), (4, 3), "EPSG:4326")
    b = wxy.warp(np.asfortranarray(m), (0, 4, 0, 3), "EPSG:4326",
                 (0, 4, 0, 3), (4, 3), "EPSG:4326")
    assert a.dtype == np.float64 and a.shape == (3, 4)
    np.testing.assert_array_equal(a, m)
    np.testing.assert_array_equal(a, b)


def test_warp_south_pole():
    # the four cells around the pole take the southernmost row
    p = wxy.warp(lat_grid(), GLOBAL, "EPSG:4326", (-5e6, 5e6, -5e6, 5e6), (100, 100), STERE)
    assert p.shape == (100, 100)
    assert np.all(p[49:51, 49:51] == -89.5)
    assert not np.isnan(p).any()


def test_warp_antimeridian():
    # an odd size puts a column of cell centres on the antimeridian; a global
    # source wraps, so the column is filled. (The one cell exactly on the
    # pole, lat -90, may still be NaN with nearest: it sits on the bottom
    # edge of the source.)
    p = wxy.warp(lat_grid(), GLOBAL, "EPSG:4326", (-5e6, 5e6, -5e6, 5e6), (101, 101), STERE)
    assert not np.isnan(p[51:, 50]).any()
    assert np.isnan(p).sum() <= 1


def test_warp_bilinear_rectangular():
    # a non-square target comes back (nrow, ncol)
    p = wxy.warp(lat_grid(), GLOBAL, "EPSG:4326", (-5e6, 5e6, -2e6, 2e6), (50, 20),
                 STERE, resample="bilinear")
    assert p.shape == (20, 50)
    # latitude is symmetric about the pole (to within the approximate
    # transformer's error), and the far corners are ~45 S
    np.testing.assert_allclose(p, p[::-1, ::-1], atol=0.1)
    assert -46 < p[0, 0] < -44 and p.min() < -85


def test_warp_errors():
    m = np.ones((2, 2))
    with pytest.raises(ValueError):
        wxy.warp(m, (0, 2, 0, 2), "EPSG:4326", (0, 2, 0, 2), (2, 2), "+proj=nonsense")
    with pytest.raises(ValueError):
        wxy.warp(m, (0, 2, 0, 2), "EPSG:4326", (0, 2, 0, 2), (2, 2), "EPSG:4326",
                 resample="mode")
    with pytest.raises(ValueError):
        wxy.warp(np.ones(4), (0, 2, 0, 2), "EPSG:4326", (0, 2, 0, 2), (2, 2), "EPSG:4326")
