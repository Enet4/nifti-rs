//! Interfaces and implementations specific to integration with `ndarray`
use util::convert_vec_f32;
use ndarray::{Array, Axis, Ix, IxDyn, ShapeBuilder};
use volume::NiftiVolume;
use std::ops::{Add, Mul};
use num::Num;
use error::Result;

/// Trait for volumes which can be converted to an ndarray.
pub trait IntoNdArray {

    /// Consume the volume into an ndarray.
    fn to_ndarray<T>(self) -> Result<Array<T, IxDyn>>
    where
        T: From<f32>,
        T: Clone,
        T: Num,
        T: Mul<Output = T>,
        T: Add<Output = T>;
}

impl<V> IntoNdArray for super::SliceView<V>
where 
    V: NiftiVolume + IntoNdArray
{
    fn to_ndarray<T>(self) -> Result<Array<T, IxDyn>>
    where
        T: From<f32>,
        T: Clone,
        T: Num,
        T: Mul<Output = T>,
        T: Add<Output = T>
    {
        // TODO optimize this implementation (we don't need the whole volume)
        let volume = self.volume.to_ndarray()?;
        Ok(volume.into_subview(Axis(self.axis as Ix), self.index as usize))
    }
}
