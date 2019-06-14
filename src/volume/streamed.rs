//! Streamed interface of a NIfTI volume and implementation.
//!
//! This API provides slice-by-slice reading of volumes, thus lowering
//! memory requirements and better supporting the manipulation of
//! large volumes.

use super::util::coords_to_index;
use super::NiftiVolume;
use byteordered::{ByteOrdered, Endianness};
use error::{NiftiError, Result};
use extension::{Extender, ExtensionSequence};
use flate2::bufread::GzDecoder;
use header::NiftiHeader;
use num_traits::{AsPrimitive, Num};
use std::fs::File;
use std::io::{BufReader, Read};
use std::ops::{Add, Mul};
use std::path::Path;
use typedef::NiftiType;
use util::nb_bytes_for_data;
use volume::element::DataElement;

/// A NIfTI-1 volume instance that is read slice by slice.
#[derive(Debug)]
pub struct StreamedNiftiVolume<R> {
    source: R,
    /// dimensions starting at 1, dim[0] is the dimensionality
    dim: [u16; 8],
    slice_dim: [u16; 8],
    datatype: NiftiType,
    scl_slope: f32,
    scl_inter: f32,
    endianness: Endianness,
}

impl<R> StreamedNiftiVolume<R> {

    /// Read a NIFTI volume from a stream of data. The header and expected byte order
    /// of the volume's data must be known in advance. It it also expected that the
    /// following bytes represent the first voxels of the volume (and not part of the
    /// extensions).
    pub fn from_reader(source: R, header: &NiftiHeader) -> Result<Self>
    where
        R: Read,
    {
        // TODO recoverable error if #dim == 0
        let _ = header.dim()?; // check dim consistency
        let datatype = header.data_type()?;
        let slice_dim = calculate_slice_dims(&header.dim);
        Ok(StreamedNiftiVolume {
            source,
            dim: header.dim,
            slice_dim,
            datatype,
            scl_slope: header.scl_slope,
            scl_inter: header.scl_inter,
            endianness: header.endianness,
        })
    }

    /// Retrieve the volume dimensions
    pub fn dim(&self) -> &[u16] {
        &self.dim[1..usize::from(self.dim[0]) + 1]
    }

    /// Retrieve the slice dimensions
    pub fn slice_dim(&self) -> &[u16] {
        &self.slice_dim[1..usize::from(self.dim[0] - 1) + 1]
    }

    // TODO make more allocation efficient; would require streaming iterator
    fn read_slice_primitive<T>(&mut self) -> Result<Vec<T>>
    where
        T: DataElement,
        T: Num,
        T: Copy,
        T: Mul<Output = T>,
        T: Add<Output = T>,
        T: AsPrimitive<u8>,
        T: AsPrimitive<f32>,
        T: AsPrimitive<f64>,
        T: AsPrimitive<u16>,
        u8: AsPrimitive<T>,
        i8: AsPrimitive<T>,
        u16: AsPrimitive<T>,
        i16: AsPrimitive<T>,
        u32: AsPrimitive<T>,
        i32: AsPrimitive<T>,
        u64: AsPrimitive<T>,
        i64: AsPrimitive<T>,
        f32: AsPrimitive<T>,
        f64: AsPrimitive<T>,
    {
        let data = vec![T::zero(); 1]; // TODO
        let index = coords_to_index(coords, self.dim())?;
        let range = &self.raw_data[index * self.datatype.size_of()..];
        self.datatype
            .read_primitive_value(range, self.endianness, self.scl_slope, self.scl_inter)
    }
}

fn calculate_slice_dims(dim: &[u16; 8]) -> [u16; 8] {
    assert!(dim[0] > 0);
    [
        dim[0] - 1,
        dim[2],
        dim[3],
        dim[4],
        dim[5],
        dim[6],
        dim[7],
        0,
    ]
}

#[cfg(test)]
mod tests {
    use super::StreamedNiftiVolume;
    use typedef::NiftiType;
    use NiftiHeader;

    #[test]
    fn test_streamed() {
        let volume_data = &[1, 3, 5, 7, 9, 11, 13, 15];
        let header = NiftiHeader {
            dim: [3, 2, 2, 2, 0, 0, 0, 0],
            datatype: NiftiType::Uint8 as i16,
            ..NiftiHeader::default()
        };

        let volume = StreamedNiftiVolume::from_reader(&volume_data[..], &header).unwrap();

    }
}