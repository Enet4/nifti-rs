//! This module defines the data element API, which enables NIfTI
//! volume API implementations to read, write and convert data
//! elements.
use crate::error::Result;
use crate::util::convert_bytes_to;
use crate::NiftiError;
use crate::NiftiType;

use byteordered::{ByteOrdered, Endian};
use num_complex::{Complex, Complex32, Complex64};
use num_traits::cast::AsPrimitive;
use rgb::*;
use safe_transmute::transmute_vec;
use std::io::Read;
use std::mem::align_of;
use std::ops::{Add, Mul};
use safe_transmute::TriviallyTransmutable;

/// Interface for linear (affine) transformations to values. Multiple
/// implementations are needed because the original type `T` may not have
/// enough precision to obtain an appropriate outcome. For example,
/// transforming a `u8` is always done through `f32`, but an `f64` is instead
/// manipulated through its own type by first converting the slope and
/// intercept arguments to `f64`.
pub trait LinearTransform<T: 'static + Copy> {
    /// Linearly transform a value with the given slope and intercept.
    fn linear_transform(value: T, slope: f32, intercept: f32) -> T;

    /// Linearly transform a sequence of values with the given slope and intercept into
    /// a vector.
    fn linear_transform_many(value: &[T], slope: f32, intercept: f32) -> Vec<T> {
        value
            .iter()
            .map(|x| Self::linear_transform(*x, slope, intercept))
            .collect()
    }

    /// Linearly transform a sequence of values inline, with the given slope and intercept.
    fn linear_transform_many_inline(value: &mut [T], slope: f32, intercept: f32) {
        for v in value.iter_mut() {
            *v = Self::linear_transform(*v, slope, intercept);
        }
    }
}

pub trait NiftiDataRescaler<T: 'static + Copy> {
    fn nifti_rescale(_value: T, _slope: f32, _intercept: f32) -> T {
        unimplemented!()
    }

    fn nifti_rescale_many(value: &[T], slope: f32, intercept: f32) -> Vec<T> {
        value
            .iter()
            .map(|x| Self::nifti_rescale(*x, slope, intercept))
            .collect()
    }

    /// Linearly transform a sequence of values inline, with the given slope and intercept.
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
impl NiftiDataRescaler<NiftiRGB> for NiftiRGB {
    fn nifti_rescale(value: NiftiRGB, _slope: f32, _intercept: f32) -> NiftiRGB {
        return value;
    }
}

// Nifti 1.1 specifies that RGB(A) data must NOT be rescaled
impl NiftiDataRescaler<NiftiRGBA> for NiftiRGBA {
    fn nifti_rescale(value: NiftiRGBA, _slope: f32, _intercept: f32) -> NiftiRGBA {
        return value;
    }
}

/// A linear transformation in which the slope and intercept parameters are
/// converted to the value's type for the affine transformation. Ideal
/// for high precision or complex number types.
#[derive(Debug)]
pub struct LinearTransformViaOriginal;

impl<T> LinearTransform<T> for LinearTransformViaOriginal
where
    T: 'static + Copy + DataElement + Mul<Output = T> + Add<Output = T> + NiftiDataRescaler<T>,
{
    fn linear_transform(value: T, slope: f32, intercept: f32) -> T {
        T::nifti_rescale(value, slope, intercept)
    }
}

#[derive(Debug)]
pub struct NoTransform;
impl<T> LinearTransform<T> for NoTransform
where
    T: 'static + Copy + DataElement,
{
    fn linear_transform(value: T, _slope: f32, _intercept: f32) -> T {
        value
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

    /// For defining how this element is linearly transformed to another.
    type Transform: LinearTransform<Self>;

    /// Read a single element from the given byte source.
    fn from_raw<R: Read, E>(src: R, endianness: E) -> Result<Self>
    where
        R: Read,
        E: Endian;

    /// Create a single element by converting a scalar value.
    fn from_u8(value: u8) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a scalar value.
    fn from_i8(value: i8) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a scalar value.
    fn from_u16(value: u16) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a scalar value.
    fn from_i16(value: i16) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a scalar value.
    fn from_u32(value: u32) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a scalar value.
    fn from_i32(value: i32) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a scalar value.
    fn from_u64(value: u64) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a scalar value.
    fn from_i64(value: i64) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a scalar value.
    fn from_f32(value: f32) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a scalar value.
    fn from_f64(value: f64) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a complex value
    fn from_complex_f32(real: f32, imag: f32) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a complex value
    fn from_complex_f64(real: f64, imag: f64) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a complex value
    fn from_complex32(value: Complex32) -> Self {
        unimplemented!()
    }

    /// Create a single element by converting a complex value
    fn from_complex64(value: Complex64) -> Self {
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

        fn from_complex_f32(real: f32, imag: f32) -> Self {
            ((real * real + imag * imag).sqrt() as $typ)
        }

        fn from_complex_f64(real: f64, imag: f64) -> Self {
            ((real * real + imag * imag).sqrt() as $typ)
        }

        fn from_complex32(value: Complex32) -> Self {
            (value.norm() as $typ)
        }

        fn from_complex64(value: Complex64) -> Self {
            (value.norm() as $typ)
        }
    };
}
macro_rules! fn_from_real_scalar {
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

        fn from_complex_f32(real: f32, imag: f32) -> Self {
            Complex::<$typ>::new(real as $typ, imag as $typ)
        }

        fn from_complex_f64(real: f64, imag: f64) -> Self {
            Complex::<$typ>::new(real as $typ, imag as $typ)
        }

        fn from_complex32(value: Complex32) -> Self {
            Complex::<$typ>::new(value.re as $typ, value.im as $typ)
        }

        fn from_complex64(value: Complex64) -> Self {
            Complex::<$typ>::new(value.re as $typ, value.im as $typ)
        }
    };
}

macro_rules! fn_from_real_scalar {
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

        fn from_complex_f32(real: f32, imag: f32) -> Self {
            Complex::<$typ>::new(real as $typ, imag as $typ)
        }

        fn from_complex_f64(real: f64, imag: f64) -> Self {
            Complex::<$typ>::new(real as $typ, imag as $typ)
        }

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
    type Transform = LinearTransformViaOriginal;
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
    type Transform = LinearTransformViaOriginal;
    fn from_raw_vec<E>(vec: Vec<u8>, _: E) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        Ok(transmute_vec(vec).unwrap())
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
    type Transform = LinearTransformViaOriginal;
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
    type Transform = LinearTransformViaOriginal;
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
    type Transform = LinearTransformViaOriginal;
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
    type Transform = LinearTransformViaOriginal;
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
    type Transform = LinearTransformViaOriginal;
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
    type Transform = LinearTransformViaOriginal;
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
    type Transform = LinearTransformViaOriginal;
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
    type Transform = LinearTransformViaOriginal;
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
    type Transform = LinearTransformViaOriginal;

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

    fn_from_real_scalar!(f32);
}

impl DataElement for Complex64 {
    const DATA_TYPE: NiftiType = NiftiType::Complex128;
    type Transform = LinearTransformViaOriginal;

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

    fn_from_real_scalar!(f64);
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct NiftiRGB {
    r: u8,
    g: u8,
    b: u8,
}

impl NiftiRGB {
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        NiftiRGB { r, g, b }
    }
}

impl Into<RGB8> for NiftiRGB {
    fn into(self) -> RGB8 {
        RGB8::new(self.r, self.g, self.b)
    }
}

impl From<RGB8> for NiftiRGB {
    fn from(rgb: RGB8) -> Self {
        NiftiRGB::new(rgb.r, rgb.g, rgb.b)
    }
}

unsafe impl TriviallyTransmutable for NiftiRGB {}

impl DataElement for NiftiRGB {
    const DATA_TYPE: NiftiType = NiftiType::Rgb24;
    type Transform = NoTransform;

    fn from_raw_vec<E>(vec: Vec<u8>, e: E) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        Ok(convert_bytes_to::<[u8; 3], _>(vec, e)
            .into_iter()
            .map(|x| NiftiRGB::new(x[0], x[1], x[2]))
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

        Ok(NiftiRGB::new(r, g, b))
    }
}

impl DataElement for [u8; 3] {
    const DATA_TYPE: NiftiType = NiftiType::Rgb24;
    type Transform = NoTransform;

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

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct NiftiRGBA {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

impl NiftiRGBA {
    fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        NiftiRGBA { r, g, b, a }
    }
}

impl Into<RGBA8> for NiftiRGBA {
    fn into(self) -> RGBA8 {
        RGBA8::new(self.r, self.g, self.b, self.a)
    }
}

unsafe impl TriviallyTransmutable for NiftiRGBA {}

impl DataElement for NiftiRGBA {
    const DATA_TYPE: NiftiType = NiftiType::Rgba32;
    type Transform = NoTransform;

    fn from_raw_vec<E>(vec: Vec<u8>, e: E) -> Result<Vec<Self>>
    where
        E: Endian,
    {
        Ok(convert_bytes_to::<[u8; 4], _>(vec, e)
            .into_iter()
            .map(|x| NiftiRGBA::new(x[0], x[1], x[2], x[3]))
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

        Ok(NiftiRGBA::new(r, g, b, a))
    }
}

impl DataElement for [u8; 4] {
    const DATA_TYPE: NiftiType = NiftiType::Rgba32;
    type Transform = NoTransform;

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