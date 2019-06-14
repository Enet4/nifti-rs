//! Module holding a lazy implementation of a NIfTI volume.

use super::util::coords_to_index;
use super::NiftiVolume;
use byteordered::{ByteOrdered, Endianness};
use error::{NiftiError, Result};
use extension::{Extender, ExtensionSequence};
use flate2::bufread::GzDecoder;
use header::NiftiHeader;
use num_traits::{AsPrimitive, Num};
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::ops::{Add, Mul};
use std::path::Path;
use typedef::NiftiType;
use util::nb_bytes_for_data;
use volume::element::DataElement;

#[cfg(feature = "ndarray_volumes")]
use ndarray::{Array, Ix, IxDyn, ShapeBuilder};
#[cfg(feature = "ndarray_volumes")]
use volume::ndarray::IntoNdArray;

/// A data type for a NIFTI-1 volume which reads data from a file on-demand.
/// The underlying data source is enquired when using one of the available
/// reading methods or [converting it to an `ndarray`] (only with the
/// `ndarray_volumes` feature).
///
/// [converting it to an `ndarray`]: ../ndarray/index.html
///
#[derive(Debug, PartialEq, Clone)]
pub struct LazyNiftiVolume<F> {
    dim: [u16; 8],
    datatype: NiftiType,
    scl_slope: f32,
    scl_inter: f32,
    endianness: Endianness,
    voxel_offset: u64,
    source: F,
}

impl<F> LazyNiftiVolume<F>
where
    F: Read,
    F: Seek,
{

}