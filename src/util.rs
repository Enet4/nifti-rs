//! Private utility module
use std::io::{Read, Seek, Result as IoResult};
use byteorder::{LittleEndian, BigEndian, ReadBytesExt};
use std::ops::{Add, Mul};
use num::Num;
use std::path::{Path, PathBuf};

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

#[cfg(feature = "ndarray_volumes")]
pub fn convert_vec_f32(a: Vec<u8>, e: Endianness) -> Vec<f32> {
    let len = a.len() / 4;
    let mut v = Vec::with_capacity(len);
    let a = a.as_slice();
    for _ in ::std::iter::repeat(()).take(len) {
        v.push(e.read_f32(a).unwrap());
    }
    v
}

pub fn is_gz_file<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().file_name()
        .map(|a| a.to_string_lossy().ends_with(".gz"))
        .unwrap_or(false)
}

/// Convert a file path to a header file (.hdr or .hdr.gz) to
/// the respective volume file with GZip compression (.img.gz).
///
/// # Panics
/// Can panic if the given file path is not a valid path to a header file.
/// If it doesn't panic in this case, the result might still not be correct.
pub fn to_img_file_gz(path: PathBuf) -> PathBuf {
    let gz = is_gz_file(&path);
    let fname = path.file_name().unwrap().to_owned();
    let fname = fname.to_string_lossy();
    let mut fname = if gz {
        fname[..fname.len() - ".hdr.gz".len()].to_owned()
    } else {
        fname[..fname.len() - ".hdr".len()].to_owned()
    };
    fname += ".img.gz";
    path.with_file_name(fname)
}

#[cfg(test)]
mod tests {
    use super::Endianness;
    use super::raw_to_value;
    use super::to_img_file_gz;
    use super::is_gz_file;
    use std::path::PathBuf;

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

    #[test]
    fn filenames() {
        assert!(!is_gz_file("/path/to/something.nii"));
        assert!(is_gz_file("/path/to/something.nii.gz"));
        assert!(!is_gz_file("volume.não"));
        assert!(is_gz_file("1.2.3.nii.gz"));
        assert!(!is_gz_file("não_é_gz.hdr"));

        assert!(!is_gz_file("/path/to/image.hdr"));
        assert_eq!(
            to_img_file_gz(PathBuf::from("/path/to/image.hdr")),
            PathBuf::from("/path/to/image.img.gz")
        );

        assert!(is_gz_file("/path/to/image.hdr.gz"));
        assert_eq!(
            to_img_file_gz(PathBuf::from("/path/to/image.hdr.gz")),
            PathBuf::from("/path/to/image.img.gz")
        );

        assert_eq!(
            to_img_file_gz(PathBuf::from("my_ct_scan.1.hdr.gz")),
            PathBuf::from("my_ct_scan.1.img.gz")
        );

        assert_eq!(
            to_img_file_gz(PathBuf::from("../you.cant.fool.me.hdr.gz")),
            PathBuf::from("../you.cant.fool.me.img.gz")
        );
    }
}