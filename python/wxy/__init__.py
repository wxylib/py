"""Python bindings for wxylib: a minimal spike.

Three functions from wxylib/core, chosen because they are pure Rust: no
I/O, no C libraries, and CRS work through the proj4rs backend that rwarp
uses. The same three are bound in R by wxylib/r.
"""

import numpy as np

from . import _wxy

__all__ = ["affine", "regular", "warp"]

_RESAMPLE = ("near", "bilinear", "cubic", "lanczos")


def affine(extent, dim):
    """Geotransform for an extent and a dimension, north up.

    Parameters
    ----------
    extent : sequence of 4 floats
        ``(xmin, xmax, ymin, ymax)``.
    dim : sequence of 2 ints
        ``(ncol, nrow)``.

    Returns
    -------
    tuple of 6 floats
        ``(x0, dx, rx, y0, ry, dy)``, the GDAL-style geotransform.

    Examples
    --------
    >>> affine((0, 10, 0, 5), (10, 5))
    (0.0, 1.0, 0.0, 5.0, 0.0, -1.0)
    """
    return tuple(_wxy.affine(_floats(extent), _floats(dim)))


def regular(x, cell="centre", tolerance="float32"):
    """Is a coordinate vector evenly spaced?

    This is the R2 -> R1 test of the position ladder in wxylib design doc
    02: does an explicit coordinate array reduce to an origin and a step?

    Parameters
    ----------
    x : 1-d array_like of float
        Coordinates, increasing or decreasing.
    cell : {"centre", "edge"}
        Whether ``x`` holds cell centres (n values) or cell edges (n + 1).
    tolerance : {"float32", "float64"} or float
        How far a value may sit from the fitted line: within the rounding
        of that storage type, or an absolute distance in ``x``'s units.

    Returns
    -------
    dict
        ``regular`` (bool), ``origin`` and ``step`` (the edge-based origin
        and spacing, NaN when not regular), ``max_deviation``,
        ``max_deviation_steps``, ``tolerance`` and a text ``report``.

    Examples
    --------
    >>> r = regular(np.arange(-179.5, 180, 1.0))
    >>> r["regular"], r["origin"], r["step"]
    (True, -180.0, 1.0)
    """
    if cell not in ("centre", "center", "edge"):
        raise ValueError("cell must be 'centre' or 'edge'")
    if not isinstance(tolerance, str):
        tolerance = float(tolerance)
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    return _wxy.regular(x, cell, tolerance)


def warp(x, extent, crs, dst_extent, dst_dim, dst_crs, resample="near", error_threshold=0.125):
    """Warp a 2-d array from one grid and CRS to another.

    Parameters
    ----------
    x : 2-d array_like of float
        Values with shape ``(nrow, ncol)``, row 0 at the north edge (the
        usual image order).
    extent : sequence of 4 floats
        ``(xmin, xmax, ymin, ymax)`` of ``x`` in ``crs``.
    crs : str
        Source CRS, as ``"EPSG:4326"`` or a PROJ string (parsed by
        proj4rs, so PROJ strings and a set of EPSG codes).
    dst_extent : sequence of 4 floats
        ``(xmin, xmax, ymin, ymax)`` of the target grid in ``dst_crs``.
    dst_dim : sequence of 2 ints
        ``(ncol, nrow)`` of the target grid.
    dst_crs : str
        Target CRS.
    resample : {"near", "bilinear", "cubic", "lanczos"}
    error_threshold : float
        Maximum error, in target pixels, of the approximate transformer
        (GDAL's default, 0.125). ``0`` transforms every pixel exactly:
        slower, and the reference when comparing results.

    Returns
    -------
    numpy.ndarray
        float64, shape ``(nrow, ncol)`` of ``dst_dim``, row 0 north. Cells
        nothing maps to are NaN.

    Examples
    --------
    >>> lat = np.arange(89.5, -90, -1.0)
    >>> z = np.repeat(lat[:, None], 360, axis=1)
    >>> w = warp(z, (-180, 180, -90, 90), "EPSG:4326",
    ...          (-5e6, 5e6, -5e6, 5e6), (100, 100),
    ...          "+proj=stere +lat_0=-90 +lat_ts=-71 +datum=WGS84")
    >>> w.shape, float(w[50, 50])
    ((100, 100), -89.5)
    """
    if resample not in _RESAMPLE:
        raise ValueError(f"resample must be one of {_RESAMPLE}")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be two-dimensional (nrow, ncol)")
    return _wxy.warp(
        x,
        _floats(extent),
        str(crs),
        _floats(dst_extent),
        _floats(dst_dim),
        str(dst_crs),
        resample,
        float(error_threshold),
    )


def _floats(v):
    return [float(a) for a in v]
