//! A minimal spike of Python bindings to wxylib: two functions from
//! wxy-domain and one from wxy-warp, the same three as the R spike in
//! wxylib/r. All pure Rust: no I/O, no C libraries, and CRS work through
//! rwarp's proj4rs backend.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use wxy_domain::affine::Affine;
use wxy_domain::mapping::CellConvention;
use wxy_domain::normalise::{regular_axis, Tolerance};
use wxy_domain::positions::Positions;
use wxy_domain::{DataType, Dim, Domain, Frame, Mapping};
use wxy_source::{Chunk, ChunkGrid, ChunkKey, ChunkSource, Encoded, Values};
use wxy_warp::{GridSpec, ResampleAlg, WarpOptions};

/// A numpy array as a one-chunk source: values in C order (row 0 north).
struct Memory {
    grid: ChunkGrid,
    values: Vec<f64>,
}

#[async_trait::async_trait]
impl ChunkSource for Memory {
    fn grid(&self) -> &ChunkGrid {
        &self.grid
    }
    fn dtype(&self) -> DataType {
        DataType::F64
    }
    async fn fetch(&self, keys: &[ChunkKey]) -> wxy_source::Result<Vec<Encoded>> {
        Ok(keys
            .iter()
            .map(|k| Encoded {
                key: k.clone(),
                payload: Box::new(()),
            })
            .collect())
    }
    fn decode(&self, e: Encoded) -> wxy_source::Result<Chunk> {
        Ok(Chunk {
            window: self.grid.window(&e.key),
            key: e.key,
            stored: self.grid.shape.clone(),
            values: Values::F64(self.values.clone()),
        })
    }
}

fn err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn extent4(name: &str, v: &[f64]) -> PyResult<[f64; 4]> {
    <[f64; 4]>::try_from(v).map_err(|_| err(format!("{name} must have 4 values")))
}

fn dim2(name: &str, v: &[f64]) -> PyResult<[usize; 2]> {
    if v.len() != 2
        || v.iter()
            .any(|d| !d.is_finite() || *d < 1.0 || d.fract() != 0.0)
    {
        return Err(err(format!("{name} must be two positive whole numbers")));
    }
    Ok([v[0] as usize, v[1] as usize])
}

/// Geotransform `[x0, dx, rx, y0, ry, dy]` for an extent
/// `[xmin, xmax, ymin, ymax]` and dimension `[ncol, nrow]`, north up.
#[pyfunction]
fn affine(extent: Vec<f64>, dim: Vec<f64>) -> PyResult<Vec<f64>> {
    let a = Affine::from_extent(extent4("extent", &extent)?, dim2("dim", &dim)?);
    Ok(a.0.to_vec())
}

/// Test whether a coordinate vector is evenly spaced (the R2 -> R1 test of
/// the position ladder). `tolerance` is "float32", "float64", or a number
/// (an absolute distance in the coordinates' units).
#[pyfunction]
fn regular<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    cell: &str,
    tolerance: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let cell = match cell {
        "centre" | "center" => CellConvention::Centre,
        "edge" => CellConvention::Edge,
        other => return Err(err(format!("unknown cell convention '{other}'"))),
    };
    let tolerance = if let Ok(s) = tolerance.extract::<String>() {
        match s.as_str() {
            "float32" => Tolerance::Float32,
            "float64" => Tolerance::Float64,
            other => return Err(err(format!("unknown tolerance '{other}'"))),
        }
    } else if let Ok(t) = tolerance.extract::<f64>() {
        Tolerance::Absolute(t)
    } else {
        return Err(err(
            "tolerance must be \"float32\", \"float64\" or a number",
        ));
    };
    let x = x.as_array().to_vec();
    let (found, report) = regular_axis(&x, cell, tolerance);
    let (origin, step) = match found {
        Some(Positions::Regular1 { origin, step }) => (origin, step),
        _ => (f64::NAN, f64::NAN),
    };
    let d = PyDict::new(py);
    d.set_item("regular", report.accepted)?;
    d.set_item("origin", origin)?;
    d.set_item("step", step)?;
    d.set_item("max_deviation", report.max_deviation)?;
    d.set_item("max_deviation_steps", report.max_deviation_steps)?;
    d.set_item("tolerance", report.tolerance)?;
    d.set_item("report", report.to_string())?;
    Ok(d)
}

/// Warp `x` (`nrow` x `ncol`, row 0 north) on `extent` in `crs` to a grid
/// of `dst_dim` cells over `dst_extent` in `dst_crs`. Returns an
/// `nrow` x `ncol` float64 array; cells nothing maps to are NaN.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn warp<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    extent: Vec<f64>,
    crs: &str,
    dst_extent: Vec<f64>,
    dst_dim: Vec<f64>,
    dst_crs: &str,
    resample: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let resampling = match resample {
        "near" => ResampleAlg::NearestNeighbour,
        "bilinear" => ResampleAlg::Bilinear,
        "cubic" => ResampleAlg::Cubic,
        "lanczos" => ResampleAlg::Lanczos,
        other => return Err(err(format!("unknown resampling '{other}'"))),
    };
    let extent = extent4("extent", &extent)?;
    let dst_extent = extent4("dst_extent", &dst_extent)?;
    let dst_dim = dim2("dst_dim", &dst_dim)?;
    let a = x.as_array();
    let (nrow, ncol) = a.dim();
    if nrow == 0 || ncol == 0 {
        return Err(err("x must have at least one row and one column"));
    }
    // Logical (row-major) order whatever the array's memory layout.
    let values: Vec<f64> = a.iter().copied().collect();

    let mut domain = Domain::new(vec![Dim::new("y", nrow as u64), Dim::new("x", ncol as u64)]);
    domain.add_mapping(Mapping::new(
        vec![0, 1],
        Positions::Affine(Affine::from_extent(extent, [ncol, nrow])),
        Frame::crs(crs),
    ));
    let source = Memory {
        grid: ChunkGrid::new(
            vec![nrow as u64, ncol as u64],
            vec![nrow as u64, ncol as u64],
        ),
        values,
    };
    let dst = GridSpec::from_extent(dst_crs, dst_extent, dst_dim);
    let opts = WarpOptions {
        resampling,
        nodata: Some(f64::NAN),
        ..Default::default()
    };
    // The warp is pure Rust, so other Python threads may run meanwhile.
    let out = py
        .detach(|| futures::executor::block_on(wxy_warp::warp(&source, &domain, &dst, &opts, None)))
        .map_err(err)?;
    let v = match out.values {
        Values::F64(v) => v,
        _ => return Err(err("unexpected output type")),
    };
    PyArray1::from_vec(py, v).reshape([dst_dim[1], dst_dim[0]])
}

#[pymodule]
fn _wxy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(affine, m)?)?;
    m.add_function(wrap_pyfunction!(regular, m)?)?;
    m.add_function(wrap_pyfunction!(warp, m)?)?;
    Ok(())
}
