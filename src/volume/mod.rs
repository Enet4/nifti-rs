//! This module defines the voxel volume API, as well as data
//! types for reading volumes from files.
//! An integration with `ndarray` allows for more elegant and
//! efficient approaches, and should be preferred when possible.
//! In order to do so, you must add the `ndarray_volumes` feature
//! to this crate.

pub mod inmem;
pub use self::inmem::*;
mod util;
use error::{NiftiError, Result};
use typedef::NiftiType;

/// Public API for NIFTI volume data, exposed as a multi-dimensional
/// voxel array.
///
/// This API is currently experimental and will likely be subjected to
/// various changes and additions in future versions.
pub trait NiftiVolume {
    /// Get the dimensions of the volume. Unlike how NIFTI-1
    /// stores dimensions, the returned slice does not include
    /// `dim[0]` and is clipped to the effective number of dimensions.
    fn dim(&self) -> &[u16];

    /// Get the volume's number of dimensions. In a fully compliant file,
    /// this is equivalent to the corresponding header's `dim[0]` field
    /// (with byte swapping already applied).
    fn dimensionality(&self) -> usize {
        self.dim().len()
    }

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
    fn get_f64(&self, coords: &[u16]) -> Result<f64>;

    /// Get this volume's data type.
    fn data_type(&self) -> NiftiType;

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
    fn get_f32(&self, coords: &[u16]) -> Result<f32> {
        let v = self.get_f64(coords)?;
        Ok(v as f32)
    }
}

/// Interface for a volume that can be sliced.
pub trait Sliceable {
    /// The type of the resulting slice, which is also a volume.
    type Slice: NiftiVolume;

    /// Obtain a slice of the volume over a certain axis, yielding a
    /// volume of N-1 dimensions.
    fn get_slice(&self, axis: u16, index: u16) -> Result<Self::Slice>;
}

/// A view over a single slice of another volume.
/// Slices are usually created by calling the `get_slice` method (see `Sliceable`).
/// This implementation is generic and delegates most operations to the underlying volume.
#[derive(Debug, Clone)]
pub struct SliceView<T> {
    volume: T,
    axis: u16,
    index: u16,
    dim: Vec<u16>,
}

impl<'a, T> Sliceable for &'a T
where
    &'a T: NiftiVolume,
{
    type Slice = SliceView<&'a T>;

    fn get_slice(&self, axis: u16, index: u16) -> Result<Self::Slice> {
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
    fn dim(&self) -> &[u16] {
        &self.dim
    }

    fn get_f32(&self, coords: &[u16]) -> Result<f32> {
        let mut coords = Vec::from(coords);
        coords.insert(self.axis as usize, self.index);
        self.volume.get_f32(&coords)
    }

    fn get_f64(&self, coords: &[u16]) -> Result<f64> {
        let mut coords = Vec::from(coords);
        coords.insert(self.axis as usize, self.index);
        self.volume.get_f64(&coords)
    }

    /// Get this volume's data type.
    fn data_type(&self) -> NiftiType {
        self.volume.data_type()
    }
}
