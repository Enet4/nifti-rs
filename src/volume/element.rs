//! This module defines the data element API, which enables NIfTI
//! volume API implementations to read, write and convert data
//! elements.
use crate::error::Result;
use crate::util::convert_bytes_to;
use crate::NiftiError;
use crate::NiftiType;

use bytemuck::*;
use byteordered::{ByteOrdered, Endian};
use num_complex::{Complex, Complex32, Complex64};
use rgb::*;
use std::io::Read;
use std::mem::align_of;

/// NiftiDataRescaler, a trait for rescaling data elements according to the Nifti 1.1 specification
pub trait NiftiDataRescaler<T: 'static + Copy> {
    /// Rescale a single value with the given slope and intercept.
    fn nifti_rescale(_value: T, _slope: f32, _intercept: f32) -> T;

    /// Rescale a slice of values, with the given slope and intercept.
    fn nifti_rescale_many(value: &[T], slope: f32, intercept: f32) -> Vec<T> {
        value
            .iter()
            .map(|x| Self::nifti_rescale(*x, slope, intercept))
            .collect()
    }

    /// Rescale a slice of values inline (inplace), with the given slope and intercept.
    fn nifti_rescale_many_inline(value: &mut [T], slope: f32, intercept: f32) {
        for v in value.iter_mut() {
            *v = Self::nifti_rescale(*v, slope, intercept);
        }
    }
}

impl NiftiDataRescaler<u8> for u8 {
    fn nifti_rescale(value: u8, slope: f32, intercept: f32) -> u8 {
        if slope == 0. {
            return value;
        }
        (value as f32 * slope + intercept) as u8
    }
}

impl NiftiDataRescaler<i8> for i8 {
    fn nifti_rescale(value: i8, slope: f32, intercept: f32) -> i8 {
        if slope == 0. {
            return value;
        }
        (value as f32 * slope + intercept) as i8
    }
}

impl NiftiDataRescaler<u16> for u16 {
    fn nifti_rescale(value: u16, slope: f32, intercept: f32) -> u16 {
        if slope == 0. {
            return value;
        }
        (value as f32 * slope + intercept) as u16
    }
}

impl NiftiDataRescaler<i16> for i16 {
    fn nifti_rescale(value: i16, slope: f32, intercept: f32) -> i16 {
        if slope == 0. {
            return value;
        }
        (value as f32 * slope + intercept) as i16
    }
}

impl NiftiDataRescaler<u32> for u32 {
    fn nifti_rescale(value: u32, slope: f32, intercept: f32) -> u32 {
        if slope == 0. {
            return value;
        }
        (value as f32 * slope + intercept) as u32
    }
}

impl NiftiDataRescaler<i32> for i32 {
    fn nifti_rescale(value: i32, slope: f32, intercept: f32) -> i32 {
        if slope == 0. {
            return value;
        }
        (value as f32 * slope + intercept) as i32
    }
}

impl NiftiDataRescaler<u64> for u64 {
    fn nifti_rescale(value: u64, slope: f32, intercept: f32) -> u64 {
        if slope == 0. {
            return value;
        }
        (value as f64 * slope as f64 + intercept as f64) as u64
    }
}

impl NiftiDataRescaler<i64> for i64 {
    fn nifti_rescale(value: i64, slope: f32, intercept: f32) -> i64 {
        if slope == 0. {
            return value;
        }
        (value as f64 * slope as f64 + intercept as f64) as i64
    }
}

impl NiftiDataRescaler<f32> for f32 {
    fn nifti_rescale(value: f32, slope: f32, intercept: f32) -> f32 {
        if slope == 0. {
            return value;
        }
        value * slope + intercept
    }
}

impl NiftiDataRescaler<f64> for f64 {
    fn nifti_rescale(value: f64, slope: f32, intercept: f32) -> f64 {
        if slope == 0. {
            return value;
        }
        value * slope as f64 + intercept as f64
    }
}

// Nifti 1.1 specifies that Complex valued data is scaled the same for both real and imaginary parts
impl NiftiDataRescaler<Complex32> for Complex32 {
    fn nifti_rescale(value: Complex32, slope: f32, intercept: f32) -> Complex32 {
        if slope == 0. {
            return value;
        }
        Complex32::new(value.re * slope + intercept, value.im * slope + intercept)
    }
}

// Nifti 1.1 specifies that Complex valued data is scaled the same for both real and imaginary parts
impl NiftiDataRescaler<Complex64> for Complex64 {
    fn nifti_rescale(value: Complex64, slope: f32, intercept: f32) -> Complex64 {
        if slope == 0. {
            return value;
        }
        Complex64::new(
            value.re * slope as f64 + intercept as f64,
            value.im * slope as f64 + intercept as f64,
        )
    }
}

// Nifti 1.1 specifies that RGB data must NOT be rescaled
impl NiftiDataRescaler<RGB8> for RGB8 {
    fn nifti_rescale(value: RGB8, _slope: f32, _intercept: f32) -> RGB8 {
        return value;
    }
}

// Nifti 1.1 specifies that RGB(A) data must NOT be rescaled
impl NiftiDataRescaler<RGBA8> for RGBA8 {
    fn nifti_rescale(value: RGBA8, _slope: f32, _intercept: f32) -> RGBA8 {
        return value;
    }
}

// This is some kind of implicit RGB for poor people, don't scale
impl NiftiDataRescaler<[u8; 3]> for [u8; 3] {
    fn nifti_rescale(value: [u8; 3], _slope: f32, _intercept: f32) -> [u8; 3] {
        return value;
    }
}

// This is some kind of implicit RGBA for poor people, don't scale
impl NiftiDataRescaler<[u8; 4]> for [u8; 4] {
    fn nifti_rescale(value: [u8; 4], _slope: f32, _intercept: f32) -> [u8; 4] {
        return value;
    }
}

/// A vessel to host the NiftiDataRescaler trait
#[derive(Debug)]
pub struct DataRescaler;

impl<T> NiftiDataRescaler<T> for DataRescaler
where
    T: 'static + Copy + DataElement + NiftiDataRescaler<T>,
{
    fn nifti_rescale(value: T, _slope: f32, _intercept: f32) -> T {
        T::nifti_rescale(value, _slope, _intercept)
    }
}

/// Trait type for characterizing a NIfTI data element, implemented for
/// primitive numeric types which are used by the crate to represent voxel
/// values.
pub trait DataElement: 'static + Sized + Copy
//+ AsPrimitive<u8> + AsPrimitive<f32> + AsPrimitive<f64>
{
    /// The `datatype` mapped to the type T
    const DATA_TYPE: NiftiType;

    /// Implement rescaling for the given data type
    type DataRescaler: NiftiDataRescaler<Self>;

    /// Read a single element from the given byte source.
    fn from_raw<R: Read, E>(src: R, endianness: E) -> Result<Self>
    where
        R: Read,
        E: Endian;

    /// Create a single element by converting a scalar value.
    fn from_u8(_value: u8) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a scalar value.
    fn from_i8(_value: i8) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a scalar value.
    fn from_u16(_value: u16) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a scalar value.
    fn from_i16(_value: i16) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a scalar value.
    fn from_u32(_value: u32) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a scalar value.
    fn from_i32(_value: i32) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a scalar value.
    fn from_u64(_value: u64) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a scalar value.
    fn from_i64(_value: i64) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a scalar value.
    fn from_f32(_value: f32) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a scalar value.
    fn from_f64(_value: f64) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a complex value
    fn from_complex32(_value: Complex32) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a complex value
    fn from_complex64(_value: Complex64) -> Self {
        unimplemented!()
    }

    /// Transform the given data vector into a vector of data elements.
    fn from_raw_vec<E>(vec: Vec<u8>, endianness: E) -> Result<Vec<Self>>
    where
        E: Endian,
        E: Clone,
    {
        let mut cursor: &[u8] = &vec;
        let n = align_of::<Self>();
        (0..n)
            .map(|_| Self::from_raw(&mut cursor, endianness))
            .collect()
    }

    /// Return a vector of data elements of the native type indicated in the Nifti file with runtime check
    fn from_raw_vec_validated<E>(
        vec: Vec<u8>,
        endianness: E,
        datatype: NiftiType,
    ) -> Result<Vec<Self>>
    where
        E: Endian;
}

/// Mass-implement primitive conversions from scalar types
macro_rules! fn_from_scalar {
    ($typ: ty) => {
        fn from_u8(value: u8) -> Self {
            value as $typ
        }

        fn from_i8(value: i8) -> Self {
            value as $typ
        }

        fn from_u16(value: u16) -> Self {
            value as $typ
        }

        fn from_i16(value: i16) -> Self {
            value as $typ
        }

        fn from_u32(value: u32) -> Self {
            value as $typ
        }

        fn from_i32(value: i32) -> Self {
            value as $typ
        }

        fn from_u64(value: u64) -> Self {
            value as $typ
        }

        fn from_i64(value: i64) -> Self {
            value as $typ
        }

        fn from_f32(value: f32) -> Self {
            value as $typ
        }

        fn from_f64(value: f64) -> Self {
            value as $typ
        }
    };
}

macro_rules! fn_cplx_from_scalar {
    ($typ: ty) => {
        fn from_u8(value: u8) -> Self {
            Complex::<$typ>::new(value as $typ, 0.)
        }

        fn from_i8(value: i8) -> Self {
            Complex::<$typ>::new(value as $typ, 0.)
        }

        fn from_u16(value: u16) -> Self {
            Complex::<$typ>::new(value as $typ, 0.)
        }

        fn from_i16(value: i16) -> Self {
            Complex::<$typ>::new(value as $typ, 0.)
        }

        fn from_u32(value: u32) -> Self {
            Complex::<$typ>::new(value as $typ, 0.)
        }

        fn from_i32(value: i32) -> Self {
            Complex::<$typ>::new(value as $typ, 0.)
        }

        fn from_u64(value: u64) -> Self {
            Complex::<$typ>::new(value as $typ, 0.)
        }

        fn from_i64(value: i64) -> Self {
            Complex::<$typ>::new(value as $typ, 0.)
        }

        fn from_f32(value: f32) -> Self {
            Complex::<$typ>::new(value as $typ, 0.)
        }

        fn from_f64(value: f64) -> Self {
            Complex::<$typ>::new(value as $typ, 0.)
        }
    };
}

macro_rules! fn_from_complex {
    ($typ: ty) => {
        fn from_complex32(value: Complex32) -> Self {
            Complex::<$typ>::new(value.re as $typ, value.im as $typ)
        }

        fn from_complex64(value: Complex64) -> Self {
            Complex::<$typ>::new(value.re as $typ, value.im as $typ)
        }
    };
}

impl DataElement for u8 {
    const DATA_TYPE: NiftiType = NiftiType::Uint8;
    type DataRescaler = DataRescaler;

    fn from_raw_vec<E>(vec: Vec<u8>, _: E) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        Ok(vec)
    }

    fn from_raw_vec_validated<E>(
        vec: Vec<u8>,
        endianness: E,
        datatype: NiftiType,
    ) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        if datatype == NiftiType::Uint8 {
            Self::from_raw_vec(vec, endianness)
        } else {
            Err(NiftiError::InvalidTypeConversion(datatype, "u8"))
        }
    }

    fn from_raw<R, E>(src: R, _: E) -> Result<Self>
    where
        R: Read,
        E: Endian,
    {
        ByteOrdered::native(src).read_u8().map_err(From::from)
    }

    fn_from_scalar!(u8);
}

impl DataElement for i8 {
    const DATA_TYPE: NiftiType = NiftiType::Int8;
    type DataRescaler = DataRescaler;
    fn from_raw_vec<E>(vec: Vec<u8>, _: E) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        Ok(try_cast_vec(vec).unwrap())
    }

    fn from_raw_vec_validated<E>(
        vec: Vec<u8>,
        endianness: E,
        datatype: NiftiType,
    ) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        if datatype == NiftiType::Int8 {
            Self::from_raw_vec(vec, endianness)
        } else {
            Err(NiftiError::InvalidTypeConversion(datatype, "i8"))
        }
    }

    fn from_raw<R, E>(src: R, _: E) -> Result<Self>
    where
        R: Read,
        E: Endian,
    {
        ByteOrdered::native(src).read_i8().map_err(From::from)
    }

    fn_from_scalar!(i8);
}

impl DataElement for u16 {
    const DATA_TYPE: NiftiType = NiftiType::Uint16;
    type DataRescaler = DataRescaler;
    fn from_raw_vec<E>(vec: Vec<u8>, e: E) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        Ok(convert_bytes_to(vec, e))
    }

    fn from_raw_vec_validated<E>(
        vec: Vec<u8>,
        endianness: E,
        datatype: NiftiType,
    ) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        if datatype == NiftiType::Uint16 {
            Self::from_raw_vec(vec, endianness)
        } else {
            Err(NiftiError::InvalidTypeConversion(datatype, "u16"))
        }
    }

    fn from_raw<R, E>(src: R, e: E) -> Result<Self>
    where
        R: Read,
        E: Endian,
    {
        e.read_u16(src).map_err(From::from)
    }

    fn_from_scalar!(u16);
}

impl DataElement for i16 {
    const DATA_TYPE: NiftiType = NiftiType::Int16;
    type DataRescaler = DataRescaler;
    fn from_raw_vec<E>(vec: Vec<u8>, e: E) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        Ok(convert_bytes_to(vec, e))
    }

    fn from_raw_vec_validated<E>(
        vec: Vec<u8>,
        endianness: E,
        datatype: NiftiType,
    ) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        if datatype == NiftiType::Int16 {
            Self::from_raw_vec(vec, endianness)
        } else {
            Err(NiftiError::InvalidTypeConversion(datatype, "i16"))
        }
    }

    fn from_raw<R, E>(src: R, e: E) -> Result<Self>
    where
        R: Read,
        E: Endian,
    {
        e.read_i16(src).map_err(From::from)
    }

    fn_from_scalar!(i16);
}

impl DataElement for u32 {
    const DATA_TYPE: NiftiType = NiftiType::Uint32;
    type DataRescaler = DataRescaler;
    fn from_raw_vec<E>(vec: Vec<u8>, e: E) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        Ok(convert_bytes_to(vec, e))
    }

    fn from_raw_vec_validated<E>(
        vec: Vec<u8>,
        endianness: E,
        datatype: NiftiType,
    ) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        if datatype == NiftiType::Uint32 {
            Self::from_raw_vec(vec, endianness)
        } else {
            Err(NiftiError::InvalidTypeConversion(datatype, "u32"))
        }
    }
    fn from_raw<R, E>(src: R, e: E) -> Result<Self>
    where
        R: Read,
        E: Endian,
    {
        e.read_u32(src).map_err(From::from)
    }

    fn_from_scalar!(u32);
}

impl DataElement for i32 {
    const DATA_TYPE: NiftiType = NiftiType::Int32;
    type DataRescaler = DataRescaler;
    fn from_raw_vec<E>(vec: Vec<u8>, e: E) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        Ok(convert_bytes_to(vec, e))
    }

    fn from_raw_vec_validated<E>(
        vec: Vec<u8>,
        endianness: E,
        datatype: NiftiType,
    ) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        if datatype == NiftiType::Int32 {
            Self::from_raw_vec(vec, endianness)
        } else {
            Err(NiftiError::InvalidTypeConversion(datatype, "i32"))
        }
    }

    fn from_raw<R, E>(src: R, e: E) -> Result<Self>
    where
        R: Read,
        E: Endian,
    {
        e.read_i32(src).map_err(From::from)
    }

    fn_from_scalar!(i32);
}

impl DataElement for u64 {
    const DATA_TYPE: NiftiType = NiftiType::Uint64;
    type DataRescaler = DataRescaler;
    fn from_raw_vec<E>(vec: Vec<u8>, e: E) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        Ok(convert_bytes_to(vec, e))
    }

    fn from_raw_vec_validated<E>(
        vec: Vec<u8>,
        endianness: E,
        datatype: NiftiType,
    ) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        if datatype == NiftiType::Uint64 {
            Self::from_raw_vec(vec, endianness)
        } else {
            Err(NiftiError::InvalidTypeConversion(datatype, "u64"))
        }
    }

    fn from_raw<R, E>(src: R, e: E) -> Result<Self>
    where
        R: Read,
        E: Endian,
    {
        e.read_u64(src).map_err(From::from)
    }

    fn_from_scalar!(u64);
}

impl DataElement for i64 {
    const DATA_TYPE: NiftiType = NiftiType::Int64;
    type DataRescaler = DataRescaler;

    fn from_raw_vec<E>(vec: Vec<u8>, e: E) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        Ok(convert_bytes_to(vec, e))
    }

    fn from_raw_vec_validated<E>(
        vec: Vec<u8>,
        endianness: E,
        datatype: NiftiType,
    ) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        if datatype == NiftiType::Int64 {
            Self::from_raw_vec(vec, endianness)
        } else {
            Err(NiftiError::InvalidTypeConversion(datatype, "i64"))
        }
    }

    fn from_raw<R, E>(src: R, e: E) -> Result<Self>
    where
        R: Read,
        E: Endian,
    {
        e.read_i64(src).map_err(From::from)
    }

    fn_from_scalar!(i64);
}

impl DataElement for f32 {
    const DATA_TYPE: NiftiType = NiftiType::Float32;
    type DataRescaler = DataRescaler;

    fn from_raw_vec<E>(vec: Vec<u8>, e: E) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        Ok(convert_bytes_to(vec, e))
    }

    fn from_raw_vec_validated<E>(
        vec: Vec<u8>,
        endianness: E,
        datatype: NiftiType,
    ) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        if datatype == NiftiType::Float32 {
            Self::from_raw_vec(vec, endianness)
        } else {
            Err(NiftiError::InvalidTypeConversion(datatype, "f32"))
        }
    }

    fn from_raw<R, E>(src: R, e: E) -> Result<Self>
    where
        R: Read,
        E: Endian,
    {
        e.read_f32(src).map_err(From::from)
    }

    fn_from_scalar!(f32);
}

impl DataElement for f64 {
    const DATA_TYPE: NiftiType = NiftiType::Float64;
    type DataRescaler = DataRescaler;

    fn from_raw_vec<E>(vec: Vec<u8>, e: E) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        Ok(convert_bytes_to(vec, e))
    }

    fn from_raw_vec_validated<E>(
        vec: Vec<u8>,
        endianness: E,
        datatype: NiftiType,
    ) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        if datatype == NiftiType::Float64 {
            Self::from_raw_vec(vec, endianness)
        } else {
            Err(NiftiError::InvalidTypeConversion(datatype, "f64"))
        }
    }

    fn from_raw<R, E>(src: R, e: E) -> Result<Self>
    where
        R: Read,
        E: Endian,
    {
        e.read_f64(src).map_err(From::from)
    }

    fn_from_scalar!(f64);
}

impl DataElement for Complex32 {
    const DATA_TYPE: NiftiType = NiftiType::Complex64;
    type DataRescaler = DataRescaler;

    fn from_raw_vec<E>(vec: Vec<u8>, e: E) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        Ok(convert_bytes_to::<[f32; 2], _>(vec, e)
            .into_iter()
            .map(|x| Complex32::new(x[0], x[1]))
            .collect())
    }

    fn from_raw_vec_validated<E>(
        vec: Vec<u8>,
        endianness: E,
        datatype: NiftiType,
    ) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        if datatype == NiftiType::Complex64 {
            Self::from_raw_vec(vec, endianness)
        } else {
            Err(NiftiError::InvalidTypeConversion(datatype, "Complex32"))
        }
    }

    fn from_raw<R, E>(mut src: R, e: E) -> Result<Self>
    where
        R: Read,
        E: Endian,
    {
        let real = e.read_f32(&mut src)?;
        let imag = e.read_f32(&mut src)?;

        Ok(Complex32::new(real, imag))
    }

    fn_cplx_from_scalar!(f32);
    fn_from_complex!(f32);
}

impl DataElement for Complex64 {
    const DATA_TYPE: NiftiType = NiftiType::Complex128;
    type DataRescaler = DataRescaler;

    fn from_raw_vec<E>(vec: Vec<u8>, e: E) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        Ok(convert_bytes_to::<[f64; 2], _>(vec, e)
            .into_iter()
            .map(|x| Complex64::new(x[0], x[1]))
            .collect())
    }

    fn from_raw_vec_validated<E>(
        vec: Vec<u8>,
        endianness: E,
        datatype: NiftiType,
    ) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        if datatype == NiftiType::Complex128 {
            Self::from_raw_vec(vec, endianness)
        } else {
            Err(NiftiError::InvalidTypeConversion(datatype, "Complex64"))
        }
    }

    fn from_raw<R, E>(mut src: R, e: E) -> Result<Self>
    where
        R: Read,
        E: Endian,
    {
        let real = e.read_f64(&mut src)?;
        let imag = e.read_f64(&mut src)?;

        Ok(Complex64::new(real, imag))
    }

    fn_cplx_from_scalar!(f64);
    fn_from_complex!(f64);
}

impl DataElement for RGB8 {
    const DATA_TYPE: NiftiType = NiftiType::Rgb24;
    type DataRescaler = DataRescaler;

    fn from_raw_vec<E>(vec: Vec<u8>, e: E) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        Ok(convert_bytes_to::<[u8; 3], _>(vec, e)
            .into_iter()
            .map(|x| RGB8::new(x[0], x[1], x[2]))
            .collect())
    }

    fn from_raw_vec_validated<E>(
        vec: Vec<u8>,
        endianness: E,
        datatype: NiftiType,
    ) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        if datatype == NiftiType::Rgb24 {
            Self::from_raw_vec(vec, endianness)
        } else {
            Err(NiftiError::InvalidTypeConversion(datatype, "RGB8"))
        }
    }

    fn from_raw<R, E>(mut src: R, _: E) -> Result<Self>
    where
        R: Read,
        E: Endian,
    {
        let r = ByteOrdered::native(&mut src).read_u8()?;
        let g = ByteOrdered::native(&mut src).read_u8()?;
        let b = ByteOrdered::native(&mut src).read_u8()?;

        Ok(RGB8::new(r, g, b))
    }
}

impl DataElement for [u8; 3] {
    const DATA_TYPE: NiftiType = NiftiType::Rgb24;
    type DataRescaler = DataRescaler;

    fn from_raw_vec<E>(vec: Vec<u8>, e: E) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        Ok(convert_bytes_to::<[u8; 3], _>(vec, e))
    }

    fn from_raw_vec_validated<E>(
        vec: Vec<u8>,
        endianness: E,
        datatype: NiftiType,
    ) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        if datatype == NiftiType::Rgb24 {
            Self::from_raw_vec(vec, endianness)
        } else {
            Err(NiftiError::InvalidTypeConversion(datatype, "[u8; 3]"))
        }
    }

    fn from_raw<R, E>(mut src: R, _: E) -> Result<Self>
    where
        R: Read,
        E: Endian,
    {
        let r = ByteOrdered::native(&mut src).read_u8()?;
        let g = ByteOrdered::native(&mut src).read_u8()?;
        let b = ByteOrdered::native(&mut src).read_u8()?;

        Ok([r, g, b])
    }
}

impl DataElement for RGBA8 {
    const DATA_TYPE: NiftiType = NiftiType::Rgba32;
    type DataRescaler = DataRescaler;

    fn from_raw_vec<E>(vec: Vec<u8>, e: E) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        Ok(convert_bytes_to::<[u8; 4], _>(vec, e)
            .into_iter()
            .map(|x| RGBA8::new(x[0], x[1], x[2], x[3]))
            .collect())
    }

    fn from_raw_vec_validated<E>(
        vec: Vec<u8>,
        endianness: E,
        datatype: NiftiType,
    ) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        if datatype == NiftiType::Rgba32 {
            Self::from_raw_vec(vec, endianness)
        } else {
            Err(NiftiError::InvalidTypeConversion(datatype, "RGBA8"))
        }
    }

    fn from_raw<R, E>(mut src: R, _: E) -> Result<Self>
    where
        R: Read,
        E: Endian,
    {
        let r = ByteOrdered::native(&mut src).read_u8()?;
        let g = ByteOrdered::native(&mut src).read_u8()?;
        let b = ByteOrdered::native(&mut src).read_u8()?;
        let a = ByteOrdered::native(&mut src).read_u8()?;

        Ok(RGBA8::new(r, g, b, a))
    }
}

impl DataElement for [u8; 4] {
    const DATA_TYPE: NiftiType = NiftiType::Rgba32;
    type DataRescaler = DataRescaler;

    fn from_raw_vec<E>(vec: Vec<u8>, e: E) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        Ok(convert_bytes_to::<[u8; 4], _>(vec, e))
    }

    fn from_raw_vec_validated<E>(
        vec: Vec<u8>,
        endianness: E,
        datatype: NiftiType,
    ) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        if datatype == NiftiType::Rgba32 {
            Self::from_raw_vec(vec, endianness)
        } else {
            Err(NiftiError::InvalidTypeConversion(datatype, "[u8; 4]"))
        }
    }

    fn from_raw<R, E>(mut src: R, _: E) -> Result<Self>
    where
        R: Read,
        E: Endian,
    {
        let r = ByteOrdered::native(&mut src).read_u8()?;
        let g = ByteOrdered::native(&mut src).read_u8()?;
        let b = ByteOrdered::native(&mut src).read_u8()?;
        let a = ByteOrdered::native(&mut src).read_u8()?;

        Ok([r, g, b, a])
    }
}
