use std::io::{Read, BufReader};
use std::fs::File;
use std::path::Path;
use header::NiftiHeader;
use error::{NiftiError, Result};
use util::{Endianness, raw_to_value};
use flate2::bufread::GzDecoder;
use typedef::NiftiType;
use num::FromPrimitive;

#[cfg(feature = "ndarray_volumes")] use ndarray::{Array, Ix, IxDyn, ShapeBuilder};
#[cfg(feature = "ndarray_volumes")] use std::ops::{Add, Mul};
#[cfg(feature = "ndarray_volumes")] use num::Num;

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

    /// Fetch a single voxel's value in the given coordinates.
    /// All necessary conversions and transformations are made
    /// when reading the voxel, including scaling.
    fn get_f32(&self, coords: &[u16]) -> Result<f32>;

    /// Get this volume's data type.
    fn data_type(&self) -> NiftiType;
}

/// A data type for a NIFTI-1 volume contained in memory.
/// Objects of this type contain raw image data, which
/// is converted automatically when reading the
///
#[derive(Debug, PartialEq, Clone)]
pub struct InMemNiftiVolume {
    dim: [u16; 8],
    datatype: NiftiType,
    scl_slope: f32,
    scl_inter: f32,
    raw_data: Vec<u8>,
    endianness: Endianness,
}

impl InMemNiftiVolume {
    
    /// Read a NIFTI volume from a stream of data. The header and expected byte order
    /// of the volume's data must be known in advance.
    pub fn from_stream<R: Read>(mut source: R, header: &NiftiHeader, endianness: Endianness) -> Result<Self> {
        let ndims = header.dim[0];
        let resolution: usize = header.dim[1..(ndims+1) as usize].iter()
            .map(|d| *d as usize)
            .product();
        let nbytes = resolution * header.bitpix as usize / 8;
        let mut raw_data = vec![0u8; nbytes];
        source.read_exact(raw_data.as_mut_slice())?;

        let datatype: NiftiType = NiftiType::from_i16(header.datatype)
            .ok_or_else(|| NiftiError::InvalidFormat)?;

        Ok(InMemNiftiVolume {
            dim: header.dim,
            datatype,
            scl_slope: header.scl_slope,
            scl_inter: header.scl_inter,
            raw_data,
            endianness,
        })
    }

    /// Read a NIFTI volume from an image file. NIFTI-1 volume files usually have the
    /// extension ".img" or ".img.gz". In the latter case, the file is automatically
    /// decoded as a Gzip stream.
    pub fn from_file<P: AsRef<Path>>(path: P, header: &NiftiHeader, endianness: Endianness) -> Result<Self> {
        let gz = path.as_ref().extension()
            .map(|a| a.to_string_lossy() == "gz")
            .unwrap_or(false);
        let file = BufReader::new(File::open(path)?);
        if gz {
            InMemNiftiVolume::from_stream(GzDecoder::new(file)?, &header, endianness)
        } else {
            InMemNiftiVolume::from_stream(file, &header, endianness)
        }
    }

    /// Retrieve the raw data, consuming the volume.
    pub fn to_raw_data(self) -> Vec<u8> {
        self.raw_data
    }

    /// Retrieve a reference to the raw data.
    pub fn get_raw_data(&self) -> &[u8] {
        &self.raw_data
    }

    /// Retrieve a mutable reference to the raw data.
    pub fn get_raw_data_mut(&mut self) -> &mut [u8] {
        &mut self.raw_data
    }
}

#[cfg(feature = "ndarray_volumes")]
// ndarray dependent impl
impl InMemNiftiVolume {

    /// Consume the volume into an ndarray.
    pub fn to_ndarray<T>(self) -> Result<Array<T, IxDyn>>
        where T: From<u8>,
              T: From<f32>,
              T: Clone,
              T: Num,
              T: Mul<Output = T>,
              T: Add<Output = T>,
    {
        if self.datatype != NiftiType::Uint8 {
            return Err(NiftiError::UnsupportedDataType(self.datatype));
        }

        let slope: T = self.scl_slope.into();
        let inter: T = self.scl_inter.into();
        let dim: Vec<_> = self.dim().iter()
            .map(|d| *d as Ix).collect();
        let a = Array::from_shape_vec(IxDyn(&dim).f(), self.raw_data)
            .expect("Inconsistent raw data size")
            .mapv(|v| raw_to_value(v, slope.clone(), inter.clone()));
        Ok(a)
    }
}

impl NiftiVolume for InMemNiftiVolume {
    fn dim(&self) -> &[u16] {
        &self.dim[1..(self.dim[0] + 1) as usize]
    }

    fn dimensionality(&self) -> usize {
        self.dim[0] as usize
    }

    fn data_type(&self) -> NiftiType {
        self.datatype
    }

    /// Fetch a single voxel's value in the given index coordinates.
    /// Scaling is performed when applicable. Note that using this
    /// function continuously to traverse the volume is inefficient.
    /// Prefer using iterators or the `ndarray` API for volume traversal.
    fn get_f32(&self, coords: &[u16]) -> Result<f32> {
        let index = coords_to_index(coords, self.dim())?;
        if self.datatype == NiftiType::Uint8 {
            let byte = self.raw_data[index];
            Ok(raw_to_value(byte as f32, self.scl_slope, self.scl_inter))
        } else {
            let range = &self.raw_data[index..];
            self.datatype.read_primitive_value(range, self.endianness, self.scl_slope, self.scl_inter)
        }
    }
}

fn coords_to_index(coords: &[u16], dim: &[u16]) -> Result<usize> {
    if coords.len() != dim.len() || coords.is_empty() {
        return Err(NiftiError::IncorrectVolumeDimensionality(
            dim.len() as u16,
            coords.len() as u16
        ))
    }

    if !coords.iter().zip(dim).all(|(i, d)| {
        *i < (*d) as u16
    }) {
        return Err(NiftiError::OutOfBounds(Vec::from(coords)));
    }

    let mut crds = coords.into_iter();
    let start = *crds.next_back().unwrap() as usize;
    let index = crds.zip(dim).rev()
        .fold(start, |a, b| {
            a * *b.1 as usize + *b.0 as usize
    });

    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::coords_to_index;

    #[test]
    fn test_coords_to_index() {
        assert!(coords_to_index(&[0, 0], &[10, 10, 5]).is_err());
        assert!(coords_to_index(&[0, 0, 0, 0], &[10, 10, 5]).is_err());
        assert_eq!(
            coords_to_index(&[0, 0, 0], &[10, 10, 5]).unwrap(),
            0
        );

        assert_eq!(
            coords_to_index(&[1, 0, 0], &[16, 16, 3]).unwrap(),
            1
        );
        assert_eq!(
            coords_to_index(&[0, 1, 0], &[16, 16, 3]).unwrap(),
            16
        );
        assert_eq!(
            coords_to_index(&[0, 0, 1], &[16, 16, 3]).unwrap(),
            256
        );
        assert_eq!(
            coords_to_index(&[1, 1, 1], &[16, 16, 3]).unwrap(),
            273
        );

        assert_eq!(
            coords_to_index(&[15, 15, 2], &[16, 16, 3]).unwrap(),
            16 * 16 * 3 - 1
        );

        assert!(coords_to_index(&[16, 15, 2], &[16, 16, 3]).is_err());
    }
}
