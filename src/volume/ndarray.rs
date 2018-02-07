//! Interfaces and implementations specific to integration with `ndarray`
use ndarray::{Array, Axis, Ix, IxDyn};
use volume::NiftiVolume;
use std::ops::{Add, Mul};
use num_traits::AsPrimitive;
use error::Result;
use volume::element::DataElement;

/// Trait for volumes which can be converted to an ndarray.
pub trait IntoNdArray {
    /// Consume the volume into an ndarray.
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
        f64: AsPrimitive<T>;
}

impl<V> IntoNdArray for super::SliceView<V>
where
    V: NiftiVolume + IntoNdArray,
{
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
        // TODO optimize this implementation (we don't need the whole volume)
        let volume = self.volume.to_ndarray()?;
        Ok(volume.into_subview(Axis(self.axis as Ix), self.index as usize))
    }
}
