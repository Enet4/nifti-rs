//! Module holding an in-memory implementation of a NIfTI volume.

use super::shape::Dim;
use super::util::coords_to_index;
use crate::error::{NiftiError, Result};
use crate::header::NiftiHeader;
use crate::typedef::NiftiType;
use crate::util::{nb_bytes_for_data, nb_bytes_for_dim_datatype};
use crate::volume::element::DataElement;
use crate::volume::{FromSource, FromSourceOptions, NiftiVolume, RandomAccessNiftiVolume};
use byteordered::Endianness;
use flate2::bufread::GzDecoder;
use num_traits::Num;
use std::fs::File;
use std::io::{BufReader, Read};
use std::ops::{Add, Mul};
use std::path::Path;

#[cfg(feature = "ndarray_volumes")]
use super::ndarray::IntoNdArray;
#[cfg(feature = "ndarray_volumes")]
use ndarray::{Array, Ix, IxDyn, ShapeBuilder};

macro_rules! fn_convert_and_cast {
    ($fname: ident, $typ: ty, $converter: expr) => {
        #[cfg(feature = "ndarray_volumes")]
        fn $fname<O>(self) -> Result<Array<O, IxDyn>>
        where
            O: DataElement,
        {
            use crate::volume::element::LinearTransform;

            let dim: Vec<_> = self.dim().iter().map(|d| *d as Ix).collect();

            // cast the raw data buffer to the DataElement
            // corresponding to the declared datatype
            let data: Vec<_> = <$typ as DataElement>::from_raw_vec(self.raw_data, self.endianness)?;
            // cast elements to the requested output type
            let mut data: Vec<O> = data.into_iter().map($converter).collect();
            // apply slope and inter before creating the final ndarray
            <O as DataElement>::Transform::linear_transform_many_inline(
                &mut data,
                self.scl_slope,
                self.scl_inter,
            );

            Ok(Array::from_shape_vec(IxDyn(&dim).f(), data).expect("Inconsistent raw data size"))
        }
    };
}

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
    dim: Dim,
    datatype: NiftiType,
    scl_slope: f64,
    scl_inter: f64,
    raw_data: Vec<u8>,
    endianness: Endianness,
}

impl InMemNiftiVolume {
    /// Build an InMemNiftiVolume from a header and a buffer. The buffer length and the dimensions
    /// declared in the header are expected to fit.
    pub fn from_raw_data(header: &NiftiHeader, raw_data: Vec<u8>) -> Result<Self> {
        let nbytes = nb_bytes_for_data(header)?;
        if nbytes != raw_data.len() {
            return Err(NiftiError::IncompatibleLength(raw_data.len(), nbytes));
        }

        let datatype = header.data_type()?;
        Ok(InMemNiftiVolume {
            dim: Dim::new(header.get_dim())?,
            datatype,
            scl_slope: header.get_scl_slope(),
            scl_inter: header.get_scl_inter(),
            raw_data,
            endianness: header.get_endianness(),
        })
    }

    /// Build an InMemNiftiVolume from its raw set of attributes. The raw data
    /// is assumed to contain exactly enough bytes to contain the data elements
    /// of the volume in F-major order, with the byte order specified in
    /// `endianness`, as specified by the volume shape in `raw_dim` and data
    /// type in `datatype`.
    pub fn from_raw_fields(
        raw_dim: [u64; 8],
        datatype: NiftiType,
        scl_slope: f64,
        scl_inter: f64,
        raw_data: Vec<u8>,
        endianness: Endianness,
    ) -> Result<Self> {
        let dim = Dim::new(raw_dim)?;
        let nbytes = nb_bytes_for_dim_datatype(dim.as_ref(), datatype);
        if nbytes != Some(raw_data.len()) {
            return Err(NiftiError::IncompatibleLength(
                raw_data.len(),
                nbytes.unwrap_or(usize::max_value()),
            ));
        }

        Ok(InMemNiftiVolume {
            dim,
            datatype,
            scl_slope,
            scl_inter,
            raw_data,
            endianness,
        })
    }

    /// Read a NIFTI volume from a stream of data. The header and expected byte order
    /// of the volume's data must be known in advance. It it also expected that the
    /// following bytes represent the first voxels of the volume (and not part of the
    /// extensions).
    pub fn from_reader<R: Read>(source: R, header: &NiftiHeader) -> Result<Self> {
        let nb_bytes = nb_bytes_for_data(header)?;
        let mut raw_data = Vec::new();
        raw_data
            .try_reserve_exact(nb_bytes)
            .map_err(|e| NiftiError::ReserveVolume(nb_bytes, e))?;
        let nb_bytes_written = source.take(nb_bytes as u64).read_to_end(&mut raw_data)?;

        if nb_bytes_written != nb_bytes {
            return Err(NiftiError::IncompatibleLength(nb_bytes_written, nb_bytes));
        }

        let datatype = header.data_type()?;
        Ok(InMemNiftiVolume {
            dim: Dim::new(header.get_dim())?,
            datatype,
            scl_slope: header.get_scl_slope(),
            scl_inter: header.get_scl_inter(),
            raw_data,
            endianness: header.get_endianness(),
        })
    }

    /// Read a NIFTI volume from an image file. NIFTI-1 volume files usually have the
    /// extension ".img" or ".img.gz". In the latter case, the file is automatically
    /// decoded as a Gzip stream.
    pub fn from_file<P: AsRef<Path>>(path: P, header: &NiftiHeader) -> Result<Self> {
        let gz = path
            .as_ref()
            .extension()
            .map(|a| a.to_string_lossy() == "gz")
            .unwrap_or(false);
        let file = BufReader::new(File::open(path)?);
        if gz {
            InMemNiftiVolume::from_reader(GzDecoder::new(file), &header)
        } else {
            InMemNiftiVolume::from_reader(file, &header)
        }
    }

    /// Retrieve the raw data, consuming the volume.
    pub fn into_raw_data(self) -> Vec<u8> {
        self.raw_data
    }

    /// Retrieve a reference to the raw data.
    pub fn raw_data(&self) -> &[u8] {
        &self.raw_data
    }

    /// Retrieve a mutable reference to the raw data.
    pub fn raw_data_mut(&mut self) -> &mut [u8] {
        &mut self.raw_data
    }

    fn get_prim<T>(&self, coords: &[u64]) -> Result<T>
    where
        T: DataElement,
        T: Num,
        T: Copy,
        T: Mul<Output = T>,
        T: Add<Output = T>,
    {
        let index = coords_to_index(coords, self.dim())?;
        let range = &self.raw_data[index * self.datatype.size_of()..];
        self.datatype
            .read_primitive_value(range, self.endianness, self.scl_slope, self.scl_inter)
    }

    fn_convert_and_cast!(convert_and_cast_u8, u8, DataElement::from_u8);
    fn_convert_and_cast!(convert_and_cast_i8, i8, DataElement::from_i8);
    fn_convert_and_cast!(convert_and_cast_u16, u16, DataElement::from_u16);
    fn_convert_and_cast!(convert_and_cast_i16, i16, DataElement::from_i16);
    fn_convert_and_cast!(convert_and_cast_u32, u32, DataElement::from_u32);
    fn_convert_and_cast!(convert_and_cast_i32, i32, DataElement::from_i32);
    fn_convert_and_cast!(convert_and_cast_u64, u64, DataElement::from_u64);
    fn_convert_and_cast!(convert_and_cast_i64, i64, DataElement::from_i64);
    fn_convert_and_cast!(convert_and_cast_f32, f32, DataElement::from_f32);
    fn_convert_and_cast!(convert_and_cast_f64, f64, DataElement::from_f64);
}

impl FromSourceOptions for InMemNiftiVolume {
    type Options = ();
}

impl<R> FromSource<R> for InMemNiftiVolume
where
    R: Read,
{
    fn from_reader(reader: R, header: &NiftiHeader, (): Self::Options) -> Result<Self> {
        InMemNiftiVolume::from_reader(reader, header)
    }
}

#[cfg(feature = "ndarray_volumes")]
impl IntoNdArray for InMemNiftiVolume {
    /// Consume the volume into an ndarray.
    fn into_ndarray<T>(self) -> Result<Array<T, IxDyn>>
    where
        T: DataElement,
    {
        match self.datatype {
            NiftiType::Uint8 => self.convert_and_cast_u8::<T>(),
            NiftiType::Int8 => self.convert_and_cast_i8::<T>(),
            NiftiType::Uint16 => self.convert_and_cast_u16::<T>(),
            NiftiType::Int16 => self.convert_and_cast_i16::<T>(),
            NiftiType::Uint32 => self.convert_and_cast_u32::<T>(),
            NiftiType::Int32 => self.convert_and_cast_i32::<T>(),
            NiftiType::Uint64 => self.convert_and_cast_u64::<T>(),
            NiftiType::Int64 => self.convert_and_cast_i64::<T>(),
            NiftiType::Float32 => self.convert_and_cast_f32::<T>(),
            NiftiType::Float64 => self.convert_and_cast_f64::<T>(),
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
    fn into_ndarray<T>(self) -> Result<Array<T, IxDyn>>
    where
        T: DataElement,
    {
        self.clone().into_ndarray()
    }
}

impl<'a> NiftiVolume for &'a InMemNiftiVolume {
    fn dim(&self) -> &[u64] {
        (**self).dim()
    }

    fn dimensionality(&self) -> usize {
        (**self).dimensionality()
    }

    fn data_type(&self) -> NiftiType {
        (**self).data_type()
    }
}

impl NiftiVolume for InMemNiftiVolume {
    fn dim(&self) -> &[u64] {
        self.dim.as_ref()
    }

    fn dimensionality(&self) -> usize {
        self.dim.rank()
    }

    fn data_type(&self) -> NiftiType {
        self.datatype
    }
}

impl RandomAccessNiftiVolume for InMemNiftiVolume {
    fn get_f32(&self, coords: &[u64]) -> Result<f32> {
        self.get_prim(coords)
    }

    fn get_f64(&self, coords: &[u64]) -> Result<f64> {
        self.get_prim(coords)
    }

    fn get_u8(&self, coords: &[u64]) -> Result<u8> {
        self.get_prim(coords)
    }

    fn get_i8(&self, coords: &[u64]) -> Result<i8> {
        self.get_prim(coords)
    }

    fn get_u16(&self, coords: &[u64]) -> Result<u16> {
        self.get_prim(coords)
    }

    fn get_i16(&self, coords: &[u64]) -> Result<i16> {
        self.get_prim(coords)
    }

    fn get_u32(&self, coords: &[u64]) -> Result<u32> {
        self.get_prim(coords)
    }

    fn get_i32(&self, coords: &[u64]) -> Result<i32> {
        self.get_prim(coords)
    }

    fn get_u64(&self, coords: &[u64]) -> Result<u64> {
        self.get_prim(coords)
    }

    fn get_i64(&self, coords: &[u64]) -> Result<i64> {
        self.get_prim(coords)
    }
}

impl<'a> RandomAccessNiftiVolume for &'a InMemNiftiVolume {
    fn get_f32(&self, coords: &[u64]) -> Result<f32> {
        (**self).get_f32(coords)
    }

    fn get_f64(&self, coords: &[u64]) -> Result<f64> {
        (**self).get_f64(coords)
    }

    fn get_u8(&self, coords: &[u64]) -> Result<u8> {
        (**self).get_u8(coords)
    }

    fn get_i8(&self, coords: &[u64]) -> Result<i8> {
        (**self).get_i8(coords)
    }

    fn get_u16(&self, coords: &[u64]) -> Result<u16> {
        (**self).get_u16(coords)
    }

    fn get_i16(&self, coords: &[u64]) -> Result<i16> {
        (**self).get_i16(coords)
    }

    fn get_u32(&self, coords: &[u64]) -> Result<u32> {
        (**self).get_u32(coords)
    }

    fn get_i32(&self, coords: &[u64]) -> Result<i32> {
        (**self).get_i32(coords)
    }

    fn get_u64(&self, coords: &[u64]) -> Result<u64> {
        (**self).get_u64(coords)
    }

    fn get_i64(&self, coords: &[u64]) -> Result<i64> {
        (**self).get_i64(coords)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typedef::NiftiType;
    use crate::volume::shape::Dim;
    use crate::volume::Sliceable;
    use byteordered::Endianness;

    #[test]
    fn test_u8_inmem_volume() {
        let data: Vec<u8> = (0..64).map(|x| x * 2).collect();
        let vol = InMemNiftiVolume {
            dim: Dim::new([3, 4, 4, 4, 0, 0, 0, 0]).unwrap(),
            datatype: NiftiType::Uint8,
            scl_slope: 1.,
            scl_inter: -5.,
            raw_data: data,
            endianness: Endianness::Little,
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
            dim: Dim::new([3, 4, 4, 4, 0, 0, 0, 0]).unwrap(),
            datatype: NiftiType::Uint8,
            scl_slope: 1.,
            scl_inter: -5.,
            raw_data: data,
            endianness: Endianness::Little,
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

    #[test]
    fn test_false_4d() {
        let (w, h, d) = (5, 5, 5);
        let mut header = crate::header::Nifti1Header {
            dim: [4, w, h, d, 1, 1, 1, 1],
            datatype: 2,
            bitpix: 8,
            ..Default::default()
        }
        .into_nifti();
        let raw_data = vec![0; (w * h * d) as usize];
        let mut volume = InMemNiftiVolume::from_raw_data(&header, raw_data).unwrap();
        assert_eq!(header.get_dim()[0], 4);
        assert_eq!(volume.dimensionality(), 4);
        if header.get_dim()[header.get_dim()[0] as usize] == 1 {
            let mut dim = header.get_dim();
            dim[0] -= 1;
            header.set_dim(&dim).unwrap();
            volume = InMemNiftiVolume::from_raw_data(&header, volume.into_raw_data()).unwrap();
        }
        assert_eq!(volume.dimensionality(), 3);

        #[cfg(feature = "ndarray_volumes")]
        {
            use ndarray::Ix3;

            let dyn_data = volume.into_ndarray::<f32>().unwrap();
            assert_eq!(dyn_data.ndim(), 3);
            let data = dyn_data.into_dimensionality::<Ix3>().unwrap();
            assert_eq!(data.ndim(), 3); // Obvious, but it's to avoid being optimized away
        }
    }
}
