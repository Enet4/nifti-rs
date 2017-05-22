//! Private utility module
use std::io::{Read, Seek, Result as IoResult};
use byteorder::{LittleEndian, BigEndian, ReadBytesExt};
use std::ops::{Add, Mul};
use num::Num;

/// A trait that is both Read and Seek.
pub trait ReadSeek: Read + Seek {}
impl<T: Read + Seek> ReadSeek for T {}

/// Enumerate for the two kinds of endianness possible by the standard.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Endianness {
    /// Little Endian
    LE,
    /// Big Endian
    BE,
}

impl Endianness {
    /// Obtain this system's endianness
    #[cfg(target_endian = "little")]
    pub fn system() -> Endianness {
        Endianness::LE
    }

    /// Obtain this system's endianness
    #[cfg(target_endian = "big")]
    pub fn system() -> Endianness {
        Endianness::BE
    }

    /// The opposite endianness: Little Endian returns Big Endian and vice versa.
    pub fn opposite(&self) -> Endianness {
        if *self == Endianness::LE {
            Endianness::BE
        } else {
            Endianness::LE
        }
    }

    /// Read a primitive value with this endianness from the given source. 
    pub fn read_i16<S>(&self, mut src: S) -> IoResult<i16>
        where S: Read
    {
        match *self {
            Endianness::LE => src.read_i16::<LittleEndian>(),
            Endianness::BE => src.read_i16::<BigEndian>(),
        }
    }

    /// Read a primitive value with this endianness from the given source. 
    pub fn read_u16<S>(&self, mut src: S) -> IoResult<u16>
        where S: Read
    {
        match *self {
            Endianness::LE => src.read_u16::<LittleEndian>(),
            Endianness::BE => src.read_u16::<BigEndian>(),
        }
    }

    /// Read a primitive value with this endianness from the given source. 
    pub fn read_i32<S>(&self, mut src: S) -> IoResult<i32>
        where S: Read
    {
        match *self {
            Endianness::LE => src.read_i32::<LittleEndian>(),
            Endianness::BE => src.read_i32::<BigEndian>(),
        }
    }

    /// Read a primitive value with this endianness from the given source. 
    pub fn read_u32<S>(&self, mut src: S) -> IoResult<u32>
        where S: Read
    {
        match *self {
            Endianness::LE => src.read_u32::<LittleEndian>(),
            Endianness::BE => src.read_u32::<BigEndian>(),
        }
    }

    /// Read a primitive value with this endianness from the given source. 
    pub fn read_i64<S>(&self, mut src: S) -> IoResult<i64>
        where S: Read
    {
        match *self {
            Endianness::LE => src.read_i64::<LittleEndian>(),
            Endianness::BE => src.read_i64::<BigEndian>(),
        }
    }

    /// Read a primitive value with this endianness from the given source. 
    pub fn read_u64<S>(&self, mut src: S) -> IoResult<u64>
        where S: Read
    {
        match *self {
            Endianness::LE => src.read_u64::<LittleEndian>(),
            Endianness::BE => src.read_u64::<BigEndian>(),
        }
    }

    /// Read a primitive value with this endianness from the given source. 
    pub fn read_f32<S>(&self, mut src: S) -> IoResult<f32>
        where S: Read
    {
        match *self {
            Endianness::LE => src.read_f32::<LittleEndian>(),
            Endianness::BE => src.read_f32::<BigEndian>(),
        }
    }

    /// Read a primitive value with this endianness from the given source. 
    pub fn read_f64<S>(&self, mut src: S) -> IoResult<f64>
        where S: Read
    {
        match *self {
            Endianness::LE => src.read_f64::<LittleEndian>(),
            Endianness::BE => src.read_f64::<BigEndian>(),
        }
    }
}

/// Defines the serialization that is opposite to system native-endian.
/// This is `BigEndian` in a Little Endian system and `LittleEndian` in a Big Endian system.
///
/// Note that this type has no value constructor. It is used purely at the
/// type level.
#[cfg(target_endian = "little")]
pub type OppositeNativeEndian = BigEndian;

/// Defines the serialization that is opposite to system native-endian.
/// This is `BigEndian` in a Little Endian system and `LittleEndian` in a Big Endian system.
///
/// Note that this type has no value constructor. It is used purely at the
/// type level.
#[cfg(target_endian = "big")]
pub type OppositeNativeEndian = LittleEndian;

/// Convert a raw volume value to the scale defined
/// by the given scale slope and intercept parameters.
pub fn raw_to_value<V, T>(value: V, slope: T, intercept: T) -> T
    where V: Into<T>,
          T: Num,
          T: Mul<Output = T>,
          T: Add<Output = T>,
{
    if slope != T::zero() {
        value.into() * slope + intercept
    } else {
        value.into()
    }
}

#[cfg(test)]
mod tests {
    use super::Endianness;
    use super::raw_to_value;

    #[test]
    fn endianness() {
        let le = Endianness::LE;
        assert_eq!(le.opposite(), Endianness::BE);
        assert_eq!(le.opposite().opposite(), Endianness::LE);
    }

    #[cfg(target_endian = "little")]
    #[test]
    fn system_endianness() {
        let le = Endianness::system();
        assert_eq!(le, Endianness::LE);
        assert_eq!(le.opposite(), Endianness::BE);
    }

    #[cfg(target_endian = "big")]
    #[test]
    fn system_endianness() {
        let le = Endianness::system();
        assert_eq!(le, Endianness::BE);
        assert_eq!(le.opposite(), Endianness::LE);
    }

    #[test]
    fn test_raw_to_value() {
        let raw: u8 = 100;
        let val: f32 = raw_to_value(raw, 2., -1024.);
        assert_eq!(val, -824.);
    }
}