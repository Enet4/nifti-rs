use std::io::{Read, BufReader};
use std::fs::File;
use std::ops::{Add, Mul};
use std::path::Path;
use header::NiftiHeader;
use error::{NiftiError, Result};
use util::Endianness;
use flate2::bufread::GzDecoder;
use typedef::NiftiType;
use num::{Num, FromPrimitive};

#[cfg(ndarray_volumes)]
use ndarray;

pub trait NiftiVolume {
    /// Get the dimensions of the volume. Unlike how NIFTI-1
    /// stores dimensions, the returned slice does not include
    /// `dim[0]` and is clipped to the effective number of dimensions.
    fn dim(&self) -> &[i16];

    /// Fetch a single voxel's value in the given coordinates.
    /// All necessary conversions and transformations are made
    /// when reading the voxel, including scaling.
    fn get_f32(&self, coords: &[u16]) -> Result<f32>;

    // TODO
}

/// A data type for a NIFTI-1 volume contained in memory.
/// Objects of this type contain raw image data, which
/// is converted automatically when reading the
///
#[derive(Debug, PartialEq, Clone)]
pub struct InMemNiftiVolume {
    dim: [i16; 8],
    datatype: NiftiType,
    scl_slope: f32,
    scl_inter: f32,
    raw_data: Vec<u8>,
    endianness: Endianness,
}

impl InMemNiftiVolume {
    pub fn from_stream<R: Read>(mut source: R, header: &NiftiHeader, endianness: Endianness) -> Result<Self> {
        let ndims = header.dim[0];
        let resolution: usize = header.dim[1..(ndims+1) as usize].iter()
            .map(|d| *d as usize)
            .product();
        let nbytes = resolution * header.bitpix as usize / 8;
        let mut raw_data = vec![0u8; nbytes];
        source.read_exact(raw_data.as_mut_slice())?;        /// This voxel data type is not supported. Sorry. :(

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

    /// Convert a raw volume value to the scale defined
    /// by the object's scale slope and intercept paramters.
    fn raw_to_value<V: Into<T>, T>(&self, value: V) -> T
        where V: Into<T>,
              T: From<f32> + Mul<Output = T> + Add<Output = T>,
    {
        if self.scl_slope != 0. {
            let slope = T::from(self.scl_slope);
            let inter = T::from(self.scl_inter);
            value.into() * slope + inter
        } else {
            value.into()
        }
    }
}

#[cfg(ndarray_volumes)]
// ndarray dependent impl
impl InMemNiftiVolume {
    /// Consume the volume into an ndarray.
    pub fn to_ndarray_f32<D>(self) -> Result<Array<f32, D>>
        where D: ndarray::Dimension
    {
        unimplemented!()
    }
}

impl NiftiVolume for InMemNiftiVolume {
    fn dim(&self) -> &[i16] {
        &self.dim[1..(self.dim[0] + 1) as usize]
    }

    /// Fetch a single voxel's value in the given coordinates.
    fn get_f32(&self, coords: &[u16]) -> Result<f32> {
        // TODO return error
        debug_assert_eq!(coords.len(), self.dim().len());
        debug_assert!(!coords.is_empty());

        // TODO add support for more data types
        if self.datatype != NiftiType::Uint8 {
            return Err(NiftiError::UnsupportedDataType(self.datatype));
        }

        let mut crds = coords.into_iter();
        let start = *crds.next().unwrap() as usize;
        let index = coords.into_iter().zip(self.dim())
            .fold(start, |a, b| {
                a * *b.1 as usize + *b.0 as usize
        });
        
        let byte = self.raw_data[index];
        Ok(self.raw_to_value(byte))
    }
}