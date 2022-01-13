//! This module defines the voxel volume API, as well as data
//! types for reading volumes from files.
//! An integration with `ndarray` allows for more elegant and
//! efficient approaches, and should be preferred when possible.
//! In order to do so, you must add the `ndarray_volumes` feature
//! to this crate.

pub mod element;
pub mod inmem;
pub mod shape;
pub mod streamed;
pub use self::inmem::*;
pub use self::streamed::StreamedNiftiVolume;

mod util;
use crate::error::{NiftiError, Result};
use crate::header::NiftiHeader;
use crate::typedef::NiftiType;
use std::io::Read;

#[cfg(feature = "ndarray_volumes")]
pub mod ndarray;

/// Public API for a NIFTI volume.
///
/// This API is currently experimental and will likely be subjected to
/// various changes and additions in future versions.
pub trait NiftiVolume {
    /// Get the dimensions of the volume. Unlike how NIFTI-1
    /// stores dimensions, the returned slice does not include
    /// `dim[0]` and is clipped to the effective number of dimensions.
    fn dim(&self) -> &[u64];

    /// Get the volume's number of dimensions. In a fully compliant file,
    /// this is equivalent to the corresponding header's `dim[0]` field
    /// (with byte swapping already applied).
    fn dimensionality(&self) -> usize {
        self.dim().len()
    }

    /// Get this volume's data type.
    fn data_type(&self) -> NiftiType;
}

/// Public API for a NIFTI volume with full random access to data.
///
/// This API is currently experimental and will likely be subjected to
/// various changes and additions in future versions.
pub trait RandomAccessNiftiVolume: NiftiVolume {
    /// Fetch a single voxel's value in the given voxel index coordinates
    /// as a double precision floating point value.
    /// All necessary conversions and transformations are made
    /// when reading the voxel, including scaling. Note that using this
    /// function continuously to traverse the volume is inefficient.
    /// Prefer using iterators or the `ndarray` API for volume traversal.
    ///
    /// # Errors
    ///
    /// - `NiftiError::OutOfBounds` if the given coordinates surpass this
    /// volume's boundaries.
    fn get_f64(&self, coords: &[u64]) -> Result<f64>;

    /// Fetch a single voxel's value in the given voxel index coordinates
    /// as a single precision floating point value.
    /// All necessary conversions and transformations are made
    /// when reading the voxel, including scaling. Note that using this
    /// function continuously to traverse the volume is inefficient.
    /// Prefer using iterators or the `ndarray` API for volume traversal.
    ///
    /// # Errors
    ///
    /// - `NiftiError::OutOfBounds` if the given coordinates surpass this
    /// volume's boundaries.
    #[inline]
    fn get_f32(&self, coords: &[u64]) -> Result<f32> {
        self.get_f64(coords).map(|v| v as f32)
    }

    /// Fetch a single voxel's value in the given voxel index coordinates
    /// as an unsigned 8-bit value.
    /// All necessary conversions and transformations are made
    /// when reading the voxel, including scaling. Note that using this
    /// function continuously to traverse the volume is inefficient.
    /// Prefer using iterators or the `ndarray` API for volume traversal.
    ///
    /// # Errors
    ///
    /// - `NiftiError::OutOfBounds` if the given coordinates surpass this
    /// volume's boundaries.
    #[inline]
    fn get_u8(&self, coords: &[u64]) -> Result<u8> {
        self.get_f64(coords).map(|v| v as u8)
    }

    /// Fetch a single voxel's value in the given voxel index coordinates
    /// as a signed 8-bit value.
    /// All necessary conversions and transformations are made
    /// when reading the voxel, including scaling. Note that using this
    /// function continuously to traverse the volume is inefficient.
    /// Prefer using iterators or the `ndarray` API for volume traversal.
    ///
    /// # Errors
    ///
    /// - `NiftiError::OutOfBounds` if the given coordinates surpass this
    /// volume's boundaries.
    #[inline]
    fn get_i8(&self, coords: &[u64]) -> Result<i8> {
        self.get_f64(coords).map(|v| v as i8)
    }

    /// Fetch a single voxel's value in the given voxel index coordinates
    /// as an unsigned 16-bit value.
    /// All necessary conversions and transformations are made
    /// when reading the voxel, including scaling. Note that using this
    /// function continuously to traverse the volume is inefficient.
    /// Prefer using iterators or the `ndarray` API for volume traversal.
    ///
    /// # Errors
    ///
    /// - `NiftiError::OutOfBounds` if the given coordinates surpass this
    /// volume's boundaries.
    #[inline]
    fn get_u16(&self, coords: &[u64]) -> Result<u16> {
        self.get_f64(coords).map(|v| v as u16)
    }

    /// Fetch a single voxel's value in the given voxel index coordinates
    /// as a signed 16-bit value.
    /// All necessary conversions and transformations are made
    /// when reading the voxel, including scaling. Note that using this
    /// function continuously to traverse the volume is inefficient.
    /// Prefer using iterators or the `ndarray` API for volume traversal.
    ///
    /// # Errors
    ///
    /// - `NiftiError::OutOfBounds` if the given coordinates surpass this
    /// volume's boundaries.
    #[inline]
    fn get_i16(&self, coords: &[u64]) -> Result<i16> {
        self.get_f64(coords).map(|v| v as i16)
    }

    /// Fetch a single voxel's value in the given voxel index coordinates
    /// as an unsigned 32-bit value.
    /// All necessary conversions and transformations are made
    /// when reading the voxel, including scaling. Note that using this
    /// function continuously to traverse the volume is inefficient.
    /// Prefer using iterators or the `ndarray` API for volume traversal.
    ///
    /// # Errors
    ///
    /// - `NiftiError::OutOfBounds` if the given coordinates surpass this
    /// volume's boundaries.
    #[inline]
    fn get_u32(&self, coords: &[u64]) -> Result<u32> {
        self.get_f64(coords).map(|v| v as u32)
    }

    /// Fetch a single voxel's value in the given voxel index coordinates
    /// as a signed 32-bit value.
    /// All necessary conversions and transformations are made
    /// when reading the voxel, including scaling. Note that using this
    /// function continuously to traverse the volume is inefficient.
    /// Prefer using iterators or the `ndarray` API for volume traversal.
    ///
    /// # Errors
    ///
    /// - `NiftiError::OutOfBounds` if the given coordinates surpass this
    /// volume's boundaries.
    #[inline]
    fn get_i32(&self, coords: &[u64]) -> Result<i32> {
        self.get_f64(coords).map(|v| v as i32)
    }

    /// Fetch a single voxel's value in the given voxel index coordinates
    /// as an unsigned 64-bit value.
    /// All necessary conversions and transformations are made
    /// when reading the voxel, including scaling. Note that using this
    /// function continuously to traverse the volume is inefficient.
    /// Prefer using iterators or the `ndarray` API for volume traversal.
    ///
    /// # Errors
    ///
    /// - `NiftiError::OutOfBounds` if the given coordinates surpass this
    /// volume's boundaries.
    #[inline]
    fn get_u64(&self, coords: &[u64]) -> Result<u64> {
        self.get_f64(coords).map(|v| v as u64)
    }

    /// Fetch a single voxel's value in the given voxel index coordinates
    /// as a signed 64-bit value.
    /// All necessary conversions and transformations are made
    /// when reading the voxel, including scaling. Note that using this
    /// function continuously to traverse the volume is inefficient.
    /// Prefer using iterators or the `ndarray` API for volume traversal.
    ///
    /// # Errors
    ///
    /// - `NiftiError::OutOfBounds` if the given coordinates surpass this
    /// volume's boundaries.
    #[inline]
    fn get_i64(&self, coords: &[u64]) -> Result<i64> {
        self.get_f64(coords).map(|v| v as i64)
    }
}

/// Interface for a volume that can be sliced at an arbitrary position.
pub trait Sliceable {
    /// The type of the resulting slice, which is also a volume.
    type Slice: NiftiVolume;

    /// Obtain a slice of the volume over a certain axis, yielding a
    /// volume of N-1 dimensions.
    fn get_slice(&self, axis: u16, index: u64) -> Result<Self::Slice>;
}

/// Interface for specifying the type for the set of options that are relevent
/// for constructing a volume instance from a data source. The separation
/// between this type and `FromSource` is important because the options are
/// invariant with respect to the reader type `R`.
pub trait FromSourceOptions {
    /// Set of additional options required (or useful) for constructing a volume.
    type Options: Clone + Default;
}

/// Interface for constructing a volume instance from the given data source.
pub trait FromSource<R>: FromSourceOptions + Sized {
    /// Read a NIfTI volume from a stream of raw voxel data. The header and
    /// expected byte order of the volume's data must be known in advance. It
    /// is also expected that the following bytes represent the first voxels of
    /// the volume (and not part of the extensions).
    fn from_reader(reader: R, header: &NiftiHeader, options: Self::Options) -> Result<Self>
    where
        R: Read;
}

/// A view over a single slice of another volume.
/// Slices are usually created by calling the `get_slice` method on another
/// volume with random access to voxels (see `Sliceable`). This implementation
/// is generic and delegates most operations to the underlying volume.
#[derive(Debug, Clone)]
pub struct SliceView<T> {
    volume: T,
    axis: u16,
    index: u64,
    dim: Vec<u64>,
}

impl<'a, T> Sliceable for &'a T
where
    &'a T: NiftiVolume,
{
    type Slice = SliceView<&'a T>;

    fn get_slice(&self, axis: u16, index: u64) -> Result<Self::Slice> {
        let mut coords: Vec<_> = self.dim().into();
        if let Some(d) = coords.get(axis as usize) {
            if *d <= index {
                return Err(NiftiError::OutOfBounds(util::hot_vector(
                    self.dimensionality(),
                    axis as usize,
                    index,
                )));
            }
        } else {
            return Err(NiftiError::AxisOutOfBounds(axis));
        }

        let _ = coords.remove(axis as usize);

        Ok(SliceView {
            volume: *self,
            axis,
            index,
            dim: coords,
        })
    }
}

impl<V> NiftiVolume for SliceView<V>
where
    V: NiftiVolume,
{
    #[inline]
    fn dim(&self) -> &[u64] {
        &self.dim
    }

    #[inline]
    fn dimensionality(&self) -> usize {
        self.dim.len()
    }

    #[inline]
    fn data_type(&self) -> NiftiType {
        self.volume.data_type()
    }
}

impl<V> RandomAccessNiftiVolume for SliceView<V>
where
    V: RandomAccessNiftiVolume,
{
    fn get_f32(&self, coords: &[u64]) -> Result<f32> {
        let mut coords = Vec::from(coords);
        coords.insert(self.axis as usize, self.index);
        self.volume.get_f32(&coords)
    }

    fn get_f64(&self, coords: &[u64]) -> Result<f64> {
        let mut coords = Vec::from(coords);
        coords.insert(self.axis as usize, self.index);
        self.volume.get_f64(&coords)
    }

    fn get_u8(&self, coords: &[u64]) -> Result<u8> {
        let mut coords = Vec::from(coords);
        coords.insert(self.axis as usize, self.index);
        self.volume.get_u8(&coords)
    }

    fn get_i8(&self, coords: &[u64]) -> Result<i8> {
        let mut coords = Vec::from(coords);
        coords.insert(self.axis as usize, self.index);
        self.volume.get_i8(&coords)
    }

    fn get_u16(&self, coords: &[u64]) -> Result<u16> {
        let mut coords = Vec::from(coords);
        coords.insert(self.axis as usize, self.index);
        self.volume.get_u16(&coords)
    }

    fn get_i16(&self, coords: &[u64]) -> Result<i16> {
        let mut coords = Vec::from(coords);
        coords.insert(self.axis as usize, self.index);
        self.volume.get_i16(&coords)
    }

    fn get_u32(&self, coords: &[u64]) -> Result<u32> {
        let mut coords = Vec::from(coords);
        coords.insert(self.axis as usize, self.index);
        self.volume.get_u32(&coords)
    }

    fn get_i32(&self, coords: &[u64]) -> Result<i32> {
        let mut coords = Vec::from(coords);
        coords.insert(self.axis as usize, self.index);
        self.volume.get_i32(&coords)
    }

    fn get_u64(&self, coords: &[u64]) -> Result<u64> {
        let mut coords = Vec::from(coords);
        coords.insert(self.axis as usize, self.index);
        self.volume.get_u64(&coords)
    }

    fn get_i64(&self, coords: &[u64]) -> Result<i64> {
        let mut coords = Vec::from(coords);
        coords.insert(self.axis as usize, self.index);
        self.volume.get_i64(&coords)
    }
}
