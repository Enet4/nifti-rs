//! Module holding an in-memory implementation of a NIfTI volume.

use super::NiftiVolume;
use super::util::coords_to_index;
use std::io::{BufReader, Read};
use std::fs::File;
use std::path::Path;
use asprim::AsPrim;
use header::NiftiHeader;
use extension::{Extender, ExtensionSequence};
use error::{NiftiError, Result};
use util::{raw_to_value, raw_to_value_via_f32, Endianness};
use byteorder::{BigEndian, LittleEndian};
use flate2::bufread::GzDecoder;
use typedef::NiftiType;
use num::{FromPrimitive, Num};

#[cfg(feature = "ndarray_volumes")]
use volume::ndarray::IntoNdArray;
#[cfg(feature = "ndarray_volumes")]
use util::{convert_bytes_and_cast_to};
#[cfg(feature = "ndarray_volumes")]
use ndarray::{Array, Ix, IxDyn, ShapeBuilder};
#[cfg(feature = "ndarray_volumes")]
use safe_transmute::PodTransmutable;
#[cfg(feature = "ndarray_volumes")]
use std::ops::{Add, Mul};

/// A data type for a NIFTI-1 volume contained in memory.
/// Objects of this type contain raw image data, which
/// is converted automatically when using reading methods
/// or converting it to an `ndarray` (with the
/// `ndarray_volumes` feature).
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
    /// of the volume's data must be known in advance. It it also expected that the
    /// following bytes represent the first voxels of the volume (and not part of the
    /// extensions).
    pub fn from_stream<R: Read>(
        mut source: R,
        header: &NiftiHeader,
        endianness: Endianness,
    ) -> Result<Self> {
        let ndims = header.dim[0];
        let resolution: usize = header.dim[1..(ndims + 1) as usize]
            .iter()
            .map(|d| *d as usize)
            .product();
        let nbytes = resolution * header.bitpix as usize / 8;
        println!("Reading volume of {:?} bytes", nbytes);
        let mut raw_data = vec![0u8; nbytes];
        source.read_exact(&mut raw_data)?;

        let datatype: NiftiType =
            NiftiType::from_i16(header.datatype).ok_or_else(|| NiftiError::InvalidFormat)?;

        Ok(InMemNiftiVolume {
            dim: header.dim,
            datatype,
            scl_slope: header.scl_slope,
            scl_inter: header.scl_inter,
            raw_data,
            endianness,
        })
    }

    /// Read a NIFTI volume, and extensions, from a stream of data. The header,
    /// extender code and expected byte order of the volume's data must be
    /// known in advance.
    pub fn from_stream_with_extensions<R>(
        mut source: R,
        header: &NiftiHeader,
        extender: Extender,
        endianness: Endianness,
    ) -> Result<(Self, ExtensionSequence)>
    where
        R: Read,
    {
        // fetch extensions
        let len = header.vox_offset as usize;
        let len = if len < 352 { 0 } else { len - 352 };

        let ext = match endianness {
            Endianness::LE => {
                ExtensionSequence::from_stream::<LittleEndian, _>(extender, &mut source, len)
            }
            Endianness::BE => {
                ExtensionSequence::from_stream::<BigEndian, _>(extender, &mut source, len)
            }
        }?;

        // fetch volume (rest of file)
        Ok((Self::from_stream(source, &header, endianness)?, ext))
    }

    /// Read a NIFTI volume from an image file. NIFTI-1 volume files usually have the
    /// extension ".img" or ".img.gz". In the latter case, the file is automatically
    /// decoded as a Gzip stream.
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        header: &NiftiHeader,
        endianness: Endianness,
    ) -> Result<Self> {
        let gz = path.as_ref()
            .extension()
            .map(|a| a.to_string_lossy() == "gz")
            .unwrap_or(false);
        let file = BufReader::new(File::open(path)?);
        if gz {
            InMemNiftiVolume::from_stream(GzDecoder::new(file), &header, endianness)
        } else {
            InMemNiftiVolume::from_stream(file, &header, endianness)
        }
    }

    /// Read a NIFTI volume, along with the extensions, from an image file. NIFTI-1 volume
    /// files usually have the extension ".img" or ".img.gz". In the latter case, the file
    /// is automatically decoded as a Gzip stream.
    pub fn from_file_with_extensions<P>(
        path: P,
        header: &NiftiHeader,
        endianness: Endianness,
        extender: Extender,
    ) -> Result<(Self, ExtensionSequence)>
    where
        P: AsRef<Path>,
    {
        let gz = path.as_ref()
            .extension()
            .map(|a| a.to_string_lossy() == "gz")
            .unwrap_or(false);
        let stream = BufReader::new(File::open(path)?);

        if gz {
            InMemNiftiVolume::from_stream_with_extensions(
                GzDecoder::new(stream),
                &header,
                extender,
                endianness,
            )
        } else {
            InMemNiftiVolume::from_stream_with_extensions(stream, &header, extender, endianness)
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

    fn get_prim<T>(&self, coords: &[u16]) -> Result<T>
    where
        T: AsPrim,
        T: Num,
        T: Copy,
    {
        let index = coords_to_index(coords, self.dim())?;
        if self.datatype == NiftiType::Uint8 {
            let byte = self.raw_data[index];
            Ok(raw_to_value(byte, self.scl_slope.as_(), self.scl_inter.as_()))
        } else if self.datatype == NiftiType::Int8 {
            let byte = self.raw_data[index] as i8;
            Ok(raw_to_value(byte, self.scl_slope.as_(), self.scl_inter.as_()))
        } else {
            let range = &self.raw_data[index * self.datatype.size_of()..];
            self.datatype.read_primitive_value(
                range,
                self.endianness,
                self.scl_slope,
                self.scl_inter,
            )
        }
    }

    fn get_prim_via_f32<T>(&self, coords: &[u16]) -> Result<T>
    where
        T: AsPrim,
        T: Num,
        T: Copy,
    {
        let index = coords_to_index(coords, self.dim())?;
        if self.datatype == NiftiType::Uint8 {
            let byte = self.raw_data[index];
            Ok(raw_to_value_via_f32(byte, self.scl_slope, self.scl_inter))
        } else if self.datatype == NiftiType::Int8 {
            let byte = self.raw_data[index] as i8;
            Ok(raw_to_value_via_f32(byte, self.scl_slope, self.scl_inter))
        } else {
            let range = &self.raw_data[index * self.datatype.size_of()..];
            self.datatype.read_primitive_value(
                range,
                self.endianness,
                self.scl_slope,
                self.scl_inter,
            )
        }
    }

    // Shortcut to avoid repeating the call for all types
    #[cfg(feature = "ndarray_volumes")]
    fn convert_bytes_and_cast_to<I, O>(self) -> Array<O, IxDyn>
        where I: AsPrim + PodTransmutable,
              O: AsPrim + Num
    {
        let dim: Vec<_> = self.dim().iter().map(|d| *d as Ix).collect();
        convert_bytes_and_cast_to::<I, O>(
            self.raw_data, self.endianness, &dim,
            self.scl_slope.as_(), self.scl_inter.as_())
    }
}

#[cfg(feature = "ndarray_volumes")]
impl IntoNdArray for InMemNiftiVolume {
    /// Consume the volume into an ndarray.
    fn to_ndarray<T>(self) -> Result<Array<T, IxDyn>>
    where T: AsPrim + Num + PodTransmutable
    {
        match self.datatype {
            NiftiType::Uint8 | NiftiType::Int8 => {
                let dim: Vec<_> = self.dim().iter().map(|d| *d as Ix).collect();
                let slope = self.scl_slope;
                let inter = self.scl_inter;
                let a = Array::from_shape_vec(IxDyn(&dim).f(), self.raw_data)
                    .expect("Inconsistent raw data size")
                    .mapv(|v| raw_to_value_via_f32(v, slope, inter));
                Ok(a)
            }
            NiftiType::Uint16 => Ok(self.convert_bytes_and_cast_to::<u16, T>()),
            NiftiType::Int16 => Ok(self.convert_bytes_and_cast_to::<i16, T>()),
            NiftiType::Uint32 => Ok(self.convert_bytes_and_cast_to::<u32, T>()),
            NiftiType::Int32 => Ok(self.convert_bytes_and_cast_to::<i32, T>()),
            NiftiType::Uint64 => Ok(self.convert_bytes_and_cast_to::<u64, T>()),
            NiftiType::Int64 => Ok(self.convert_bytes_and_cast_to::<i64, T>()),
            NiftiType::Float32 => Ok(self.convert_bytes_and_cast_to::<f32, T>()),
            NiftiType::Float64 => Ok(self.convert_bytes_and_cast_to::<f64, T>()),
            //NiftiType::Float128 => {}
            //NiftiType::Complex64 => {}
            //NiftiType::Complex128 => {}
            //NiftiType::Complex256 => {}
            //NiftiType::Rgb24 => {}
            //NiftiType::Rgba32 => {}
            _ => Err(NiftiError::UnsupportedDataType(self.datatype)),
        }
    }
}

#[cfg(feature = "ndarray_volumes")]
impl<'a> IntoNdArray for &'a InMemNiftiVolume {
    /// Create an ndarray from the given volume.
    fn to_ndarray<T>(self) -> Result<Array<T, IxDyn>>
    where
        T: AsPrim,
        T: Clone,
        T: Num,
        T: Mul<Output = T>,
        T: Add<Output = T>,
        T: PodTransmutable
    {
        self.clone().to_ndarray()
    }
}

impl<'a> NiftiVolume for &'a InMemNiftiVolume {
    fn dim(&self) -> &[u16] {
        (**self).dim()
    }

    fn dimensionality(&self) -> usize {
        (**self).dimensionality()
    }

    fn data_type(&self) -> NiftiType {
        (**self).data_type()
    }

    fn get_f32(&self, coords: &[u16]) -> Result<f32> {
        (**self).get_f32(coords)
    }

    fn get_f64(&self, coords: &[u16]) -> Result<f64> {
        (**self).get_f64(coords)
    }

    fn get_u8(&self, coords: &[u16]) -> Result<u8> {
        (**self).get_u8(coords)
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

    fn get_f32(&self, coords: &[u16]) -> Result<f32> {
        self.get_prim(coords)
    }

    fn get_f64(&self, coords: &[u16]) -> Result<f64> {
        self.get_prim(coords)
    }

    fn get_u8(&self, coords: &[u16]) -> Result<u8> {
        self.get_prim_via_f32(coords)
    }

    fn get_i8(&self, coords: &[u16]) -> Result<i8> {
        self.get_prim_via_f32(coords)
    }

    fn get_u16(&self, coords: &[u16]) -> Result<u16> {
        self.get_prim_via_f32(coords)
    }

    fn get_i16(&self, coords: &[u16]) -> Result<i16> {
        self.get_prim_via_f32(coords)
    }

    fn get_u32(&self, coords: &[u16]) -> Result<u32> {
        self.get_prim_via_f32(coords)
    }

    fn get_i32(&self, coords: &[u16]) -> Result<i32> {
        self.get_prim_via_f32(coords)
    }

    fn get_u64(&self, coords: &[u16]) -> Result<u64> {
        self.get_prim(coords)
    }

    fn get_i64(&self, coords: &[u16]) -> Result<i64> {
        self.get_prim(coords)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use volume::Sliceable;
    use typedef::NiftiType;
    use util::Endianness;

    #[test]
    fn test_u8_inmem_volume() {
        let data: Vec<u8> = (0..64).map(|x| x * 2).collect();
        let vol = InMemNiftiVolume {
            dim: [3, 4, 4, 4, 0, 0, 0, 0],
            datatype: NiftiType::Uint8,
            scl_slope: 1.,
            scl_inter: -5.,
            raw_data: data,
            endianness: Endianness::LE,
        };

        let v = vol.get_f32(&[3, 1, 0]).unwrap();
        assert_eq!(v, 9.);

        let v = vol.get_f32(&[3, 3, 3]).unwrap();
        assert_eq!(v, 121.);

        let v = vol.get_f32(&[2, 1, 1]).unwrap();
        assert_eq!(v, 39.);

        assert!(vol.get_f32(&[4, 0, 0]).is_err());
    }

    #[test]
    fn test_u8_inmem_volume_slice() {
        let data: Vec<u8> = (0..64).map(|x| x * 2).collect();
        let vol = InMemNiftiVolume {
            dim: [3, 4, 4, 4, 0, 0, 0, 0],
            datatype: NiftiType::Uint8,
            scl_slope: 1.,
            scl_inter: -5.,
            raw_data: data,
            endianness: Endianness::LE,
        };

        let slice = (&vol).get_slice(0, 3).unwrap();
        assert_eq!(slice.dim(), &[4, 4]);
        assert_eq!(slice.dimensionality(), 2);

        let v = slice.get_f32(&[1, 0]).unwrap();
        assert_eq!(v, 9.);
        let v = slice.get_f32(&[3, 3]).unwrap();
        assert_eq!(v, 121.);

        let slice = (&vol).get_slice(1, 1).unwrap();
        assert_eq!(slice.dim(), &[4, 4]);
        assert_eq!(slice.dimensionality(), 2);
        let v = slice.get_f32(&[2, 1]).unwrap();
        assert_eq!(v, 39.);
    }
}
