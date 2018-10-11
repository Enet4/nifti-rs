//! Module holding an in-memory implementation of a NIfTI volume.

use super::NiftiVolume;
use super::util::coords_to_index;
use std::io::{BufReader, Read};
use std::fs::File;
use std::path::Path;
use std::ops::{Add, Mul};
use header::NiftiHeader;
use extension::{Extender, ExtensionSequence};
use error::{NiftiError, Result};
use volume::element::DataElement;
use util::{Endianness, nb_bytes_for_data};
use byteorder::{BigEndian, LittleEndian};
use flate2::bufread::GzDecoder;
use typedef::NiftiType;
use num_traits::{AsPrimitive, FromPrimitive, Num};

#[cfg(feature = "ndarray_volumes")]
use volume::ndarray::IntoNdArray;
#[cfg(feature = "ndarray_volumes")]
use ndarray::{Array, Ix, IxDyn, ShapeBuilder};

/// A data type for a NIFTI-1 volume contained in memory. Objects of this type
/// contain raw image data, which is converted automatically when using reading
/// methods or [converting it to an `ndarray`] (only with the `ndarray_volumes`
/// feature).
///
/// Since NIfTI volumes are stored in disk in column major order (also called
/// Fortran order), this data type will also retain this memory order.
///
/// [converting it to an `ndarray`]: ../ndarray/index.html
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
    /// Build an InMemNiftiVolume from a header and a buffer. The buffer length and the dimensions
    /// declared in the header are expected to fit.
    pub fn from_raw_data(header: &NiftiHeader, raw_data: Vec<u8>) -> Result<Self> {
        if nb_bytes_for_data(header) != raw_data.len() {
            return Err(NiftiError::IncompatibleLengthError);
        }

        let datatype: NiftiType =
            NiftiType::from_i16(header.datatype).ok_or_else(|| NiftiError::InvalidFormat)?;
        Ok(InMemNiftiVolume {
            dim: header.dim,
            datatype,
            scl_slope: header.scl_slope,
            scl_inter: header.scl_inter,
            raw_data,
            endianness: header.endianness
        })
    }

    /// Read a NIFTI volume from a stream of data. The header and expected byte order
    /// of the volume's data must be known in advance. It it also expected that the
    /// following bytes represent the first voxels of the volume (and not part of the
    /// extensions).
    pub fn from_stream<R: Read>(
        mut source: R,
        header: &NiftiHeader,
    ) -> Result<Self> {
        let mut raw_data = vec![0u8; nb_bytes_for_data(header)];
        source.read_exact(&mut raw_data)?;

        let datatype: NiftiType =
            NiftiType::from_i16(header.datatype).ok_or_else(|| NiftiError::InvalidFormat)?;

        Ok(InMemNiftiVolume {
            dim: header.dim,
            datatype,
            scl_slope: header.scl_slope,
            scl_inter: header.scl_inter,
            raw_data,
            endianness: header.endianness,
        })
    }

    /// Read a NIFTI volume, and extensions, from a stream of data. The header,
    /// extender code and expected byte order of the volume's data must be
    /// known in advance.
    pub fn from_stream_with_extensions<R>(
        mut source: R,
        header: &NiftiHeader,
        extender: Extender,
    ) -> Result<(Self, ExtensionSequence)>
    where
        R: Read,
    {
        // fetch extensions
        let len = header.vox_offset as usize;
        let len = if len < 352 { 0 } else { len - 352 };

        let ext = match header.endianness {
            Endianness::LE => {
                ExtensionSequence::from_stream::<LittleEndian, _>(extender, &mut source, len)
            }
            Endianness::BE => {
                ExtensionSequence::from_stream::<BigEndian, _>(extender, &mut source, len)
            }
        }?;

        // fetch volume (rest of file)
        Ok((Self::from_stream(source, &header)?, ext))
    }

    /// Read a NIFTI volume from an image file. NIFTI-1 volume files usually have the
    /// extension ".img" or ".img.gz". In the latter case, the file is automatically
    /// decoded as a Gzip stream.
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        header: &NiftiHeader,
    ) -> Result<Self> {
        let gz = path.as_ref()
            .extension()
            .map(|a| a.to_string_lossy() == "gz")
            .unwrap_or(false);
        let file = BufReader::new(File::open(path)?);
        if gz {
            InMemNiftiVolume::from_stream(GzDecoder::new(file), &header)
        } else {
            InMemNiftiVolume::from_stream(file, &header)
        }
    }

    /// Read a NIFTI volume, along with the extensions, from an image file. NIFTI-1 volume
    /// files usually have the extension ".img" or ".img.gz". In the latter case, the file
    /// is automatically decoded as a Gzip stream.
    pub fn from_file_with_extensions<P>(
        path: P,
        header: &NiftiHeader,
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
            )
        } else {
            InMemNiftiVolume::from_stream_with_extensions(stream, &header, extender)
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
        let index = coords_to_index(coords, self.dim())?;
        let range = &self.raw_data[index * self.datatype.size_of()..];
        self.datatype.read_primitive_value(
            range,
            self.endianness,
            self.scl_slope,
            self.scl_inter,
        )
    }

    // Shortcut to avoid repeating the call for all types
    #[cfg(feature = "ndarray_volumes")]
    fn convert_bytes_and_cast_to<I, O>(self) -> Result<Array<O, IxDyn>>
        where
            I: DataElement,
            I: AsPrimitive<O>,
            O: DataElement,
    {
        use volume::element::LinearTransform;

        let dim: Vec<_> = self.dim().iter().map(|d| *d as Ix).collect();

        let data: Vec<_> = <I as DataElement>::from_raw_vec(self.raw_data, self.endianness)?;
        let mut data: Vec<O> = data.into_iter().map(AsPrimitive::as_).collect();
        <O as DataElement>::Transform::linear_transform_many_inline(&mut data, self.scl_slope, self.scl_inter);

        Ok(Array::from_shape_vec(IxDyn(&dim).f(), data)
            .expect("Inconsistent raw data size"))
    }
}

#[cfg(feature = "ndarray_volumes")]
impl IntoNdArray for InMemNiftiVolume {
    /// Consume the volume into an ndarray.
    fn to_ndarray<T>(self) -> Result<Array<T, IxDyn>>
    where
        T: DataElement,
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
        match self.datatype {
            NiftiType::Uint8 => self.convert_bytes_and_cast_to::<u8, T>(),
            NiftiType::Int8 => self.convert_bytes_and_cast_to::<i8, T>(),
            NiftiType::Uint16 => self.convert_bytes_and_cast_to::<u16, T>(),
            NiftiType::Int16 => self.convert_bytes_and_cast_to::<i16, T>(),
            NiftiType::Uint32 => self.convert_bytes_and_cast_to::<u32, T>(),
            NiftiType::Int32 => self.convert_bytes_and_cast_to::<i32, T>(),
            NiftiType::Uint64 => self.convert_bytes_and_cast_to::<u64, T>(),
            NiftiType::Int64 => self.convert_bytes_and_cast_to::<i64, T>(),
            NiftiType::Float32 => self.convert_bytes_and_cast_to::<f32, T>(),
            NiftiType::Float64 => self.convert_bytes_and_cast_to::<f64, T>(),
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
        T: Mul<Output = T>,
        T: Add<Output = T>,
        T: DataElement,
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
        self.get_prim(coords)
    }

    fn get_i8(&self, coords: &[u16]) -> Result<i8> {
        self.get_prim(coords)
    }

    fn get_u16(&self, coords: &[u16]) -> Result<u16> {
        self.get_prim(coords)
    }

    fn get_i16(&self, coords: &[u16]) -> Result<i16> {
        self.get_prim(coords)
    }

    fn get_u32(&self, coords: &[u16]) -> Result<u32> {
        self.get_prim(coords)
    }

    fn get_i32(&self, coords: &[u16]) -> Result<i32> {
        self.get_prim(coords)
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
