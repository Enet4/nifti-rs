//! Private utility module
use std::io::{Read, Seek};

/// A trait that is both Read and Seek.
pub trait ReadSeek: Read + Seek {}
impl<T: Read + Seek> ReadSeek for T {}

// Enumerate for the two kinds of endianness possible by the standard.
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
}

#[cfg(test)]
mod tests {
    use super::Endianness;

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
}