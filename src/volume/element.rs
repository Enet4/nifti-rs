//! This module defines the data element API, which enables NIfTI
//! volume API implementations to read, write and convert data
//! elements.
use crate::error::Result;
use crate::util::convert_bytes_to;
use crate::NiftiType;
use byteordered::{ByteOrdered, Endian};
use num_traits::cast::AsPrimitive;
use safe_transmute::{transmute_vec};
use std::io::Read;
use std::mem::align_of;
use std::ops::{Add, Mul};
use num_complex::{Complex, Complex32, Complex64};

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

pub trait NiftiDataRescaler {
    fn nifti_rescale(&self, slope: f32, intercept: f32) -> Self;
}

impl NiftiDataRescaler for u8 {
    fn nifti_rescale(&self, slope: f32, intercept: f32) -> Self {
        if slope == 0. {
            return *self;
        }
        (*self as f32 * slope + intercept) as u8
    }
}

impl NiftiDataRescaler for i8 {
    fn nifti_rescale(&self, slope: f32, intercept: f32) -> Self {
        if slope == 0. {
            return *self;
        }
        (*self as f32 * slope + intercept) as i8
    }
}

impl NiftiDataRescaler for u16 {
    fn nifti_rescale(&self, slope: f32, intercept: f32) -> Self {
        if slope == 0. {
            return *self;
        }
        (*self as f32 * slope + intercept) as u16
    }
}

impl NiftiDataRescaler for i16 {
    fn nifti_rescale(&self, slope: f32, intercept: f32) -> Self {
        if slope == 0. {
            return *self;
        }
        (*self as f32 * slope + intercept) as i16
    }
}

impl NiftiDataRescaler for u32 {
    fn nifti_rescale(&self, slope: f32, intercept: f32) -> Self {
        if slope == 0. {
            return *self;
        }
        (*self as f32 * slope + intercept) as u32
    }
}

impl NiftiDataRescaler for i32 {
    fn nifti_rescale(&self, slope: f32, intercept: f32) -> Self {
        if slope == 0. {
            return *self;
        }
        (*self as f32 * slope + intercept) as i32
    }
}

impl NiftiDataRescaler for u64 {
    fn nifti_rescale(&self, slope: f32, intercept: f32) -> Self {
        if slope == 0. {
            return *self;
        }
        (*self as f64 * slope as f64 + intercept as f64) as u64
    }
}

impl NiftiDataRescaler for i64 {
    fn nifti_rescale(&self, slope: f32, intercept: f32) -> Self {
        if slope == 0. {
            return *self;
        }
        (*self as f64 * slope as f64 + intercept as f64) as i64
    }
}

impl NiftiDataRescaler for f32 {
    fn nifti_rescale(&self, slope: f32, intercept: f32) -> Self {
        if slope == 0. {
            return *self;
        }
        self * slope + intercept
    }
}

impl NiftiDataRescaler for f64 {
    fn nifti_rescale(&self, slope: f32, intercept: f32) -> Self {
        if slope == 0. {
            return *self;
        }
        *self * slope as f64 + intercept as f64
    }
}

impl NiftiDataRescaler for Complex32 {
    fn nifti_rescale(&self, slope: f32, intercept: f32) -> Self {
        if slope == 0. {
            return *self;
        }
        Complex32::new(self.re * slope + intercept, self.im * slope + intercept)
    }
}

impl NiftiDataRescaler for Complex64 {
    fn nifti_rescale(&self, slope: f32, intercept: f32) -> Self {
        if slope == 0. {
            return *self;
        }
        Complex64::new(self.re * slope as f64 + intercept as f64, self.im * slope as f64 + intercept as f64)
    }
}

/// A linear transformation in which the slope and intercept parameters are
/// converted to the value's type for the affine transformation. Ideal
/// for high precision or complex number types.
#[derive(Debug)]
pub struct LinearTransformViaOriginal;

impl<T> LinearTransform<T> for LinearTransformViaOriginal
where
    T: 'static + Copy + DataElement + Mul<Output = T> + Add<Output = T> + NiftiDataRescaler,
{
    fn linear_transform(value: T, slope: f32, intercept: f32) -> T {
        T::nifti_rescale(&value, slope, intercept)
    }
}

/// Trait type for characterizing a NIfTI data element, implemented for
/// primitive numeric types which are used by the crate to represent voxel
/// values.
pub trait DataElement:
    'static + Sized + Copy + AsPrimitive<u8> + AsPrimitive<f32> + AsPrimitive<f64>
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
    fn from_u8(value: u8) -> Self;

    /// Create a single element by converting a scalar value.
    fn from_i8(value: i8) -> Self;

    /// Create a single element by converting a scalar value.
    fn from_u16(value: u16) -> Self;

    /// Create a single element by converting a scalar value.
    fn from_i16(value: i16) -> Self;

    /// Create a single element by converting a scalar value.
    fn from_u32(value: u32) -> Self;

    /// Create a single element by converting a scalar value.
    fn from_i32(value: i32) -> Self;

    /// Create a single element by converting a scalar value.
    fn from_u64(value: u64) -> Self;

    /// Create a single element by converting a scalar value.
    fn from_i64(value: i64) -> Self;

    /// Create a single element by converting a scalar value.
    fn from_f32(value: f32) -> Self;

    /// Create a single element by converting a scalar value.
    fn from_f64(value: f64) -> Self;

    /// Create a single element by converting a complex value
    fn from_complex_f32(real: f32, imag: f32) -> Self;

    /// Create a single element by converting a complex value
    fn from_complex_f64(real: f64, imag: f64) -> Self;
    
    /// Create a single element by converting a complex value
    fn from_complex32(value: Complex32) -> Self;

    /// Create a single element by converting a complex value
    fn from_complex64(value: Complex64) -> Self;

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
            ((real*real + imag*imag).sqrt() as $typ)
        }

        fn from_complex_f64(real: f64, imag: f64) -> Self {
            ((real*real + imag*imag).sqrt() as $typ)
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
    }
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
        Ok(convert_bytes_to::<[f32;2],_>(vec, e).into_iter().map(|x| Complex32::new(x[0],x[1])).collect())
    }

    fn from_raw<R, E>(mut src: R, e: E) -> Result<Self>
    where
        R: Read,
        E: Endian,
    {
        let real = e.read_f32(&mut src)?;
        let imag = e.read_f32(&mut src)?;

        Ok(Complex32::new(real,imag))
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

        Ok(convert_bytes_to::<[f64;2],_>(vec, e).into_iter().map(|x| Complex64::new(x[0],x[1])).collect())
    }

    fn from_raw<R, E>(mut src: R, e: E) -> Result<Self>
    where
        R: Read,
        E: Endian,
    {
        let real = e.read_f64(&mut src)?;
        let imag = e.read_f64(&mut src)?;

        Ok(Complex64::new(real,imag))
    }

    fn_from_real_scalar!(f64);
}