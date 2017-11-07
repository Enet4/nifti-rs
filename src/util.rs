//! Private utility module
use std::io::{Read, Result as IoResult, Seek};
use std::ops::{Add, Mul};
use std::path::{Path, PathBuf};
use asprim::AsPrim;
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use num::Num;

#[cfg(feature = "ndarray_volumes")]
use std::mem;
#[cfg(feature = "ndarray_volumes")]
use ndarray::{Array, Ix, IxDyn, ShapeBuilder};
#[cfg(feature = "ndarray_volumes")]
use safe_transmute::{guarded_transmute_pod_vec_permissive, PodTransmutable};

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
    where
        S: Read,
    {
        match *self {
            Endianness::LE => src.read_i16::<LittleEndian>(),
            Endianness::BE => src.read_i16::<BigEndian>(),
        }
    }

    /// Read a primitive value with this endianness from the given source.
    pub fn read_u16<S>(&self, mut src: S) -> IoResult<u16>
    where
        S: Read,
    {
        match *self {
            Endianness::LE => src.read_u16::<LittleEndian>(),
            Endianness::BE => src.read_u16::<BigEndian>(),
        }
    }

    /// Read a primitive value with this endianness from the given source.
    pub fn read_i32<S>(&self, mut src: S) -> IoResult<i32>
    where
        S: Read,
    {
        match *self {
            Endianness::LE => src.read_i32::<LittleEndian>(),
            Endianness::BE => src.read_i32::<BigEndian>(),
        }
    }

    /// Read a primitive value with this endianness from the given source.
    pub fn read_u32<S>(&self, mut src: S) -> IoResult<u32>
    where
        S: Read,
    {
        match *self {
            Endianness::LE => src.read_u32::<LittleEndian>(),
            Endianness::BE => src.read_u32::<BigEndian>(),
        }
    }

    /// Read a primitive value with this endianness from the given source.
    pub fn read_i64<S>(&self, mut src: S) -> IoResult<i64>
    where
        S: Read,
    {
        match *self {
            Endianness::LE => src.read_i64::<LittleEndian>(),
            Endianness::BE => src.read_i64::<BigEndian>(),
        }
    }

    /// Read a primitive value with this endianness from the given source.
    pub fn read_u64<S>(&self, mut src: S) -> IoResult<u64>
    where
        S: Read,
    {
        match *self {
            Endianness::LE => src.read_u64::<LittleEndian>(),
            Endianness::BE => src.read_u64::<BigEndian>(),
        }
    }

    /// Read a primitive value with this endianness from the given source.
    pub fn read_f32<S>(&self, mut src: S) -> IoResult<f32>
    where
        S: Read,
    {
        match *self {
            Endianness::LE => src.read_f32::<LittleEndian>(),
            Endianness::BE => src.read_f32::<BigEndian>(),
        }
    }

    /// Read a primitive value with this endianness from the given source.
    pub fn read_f64<S>(&self, mut src: S) -> IoResult<f64>
    where
        S: Read,
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
/// The linear transformation is performed over the type `T`.
pub fn raw_to_value<V, T>(value: V, slope: T, intercept: T) -> T
where
    T: AsPrim,
    V: AsPrim,
    T: Num,
    T: Mul<Output = T>,
    T: Add<Output = T>,
{
    if slope != T::zero() {
        slope * value.as_() + intercept
    } else {
        value.as_()
    }
}

/// Convert a raw volume value to the scale defined
/// by the given scale slope and intercept parameters.
/// This implementation performs the linear transformation
/// over the `f32` type, then converts it to the intended value type.
pub fn raw_to_value_via_f32<V, T>(value: V, slope: f32, intercept: f32) -> T
where
    T: AsPrim,
    V: AsPrim
{
    if slope != 0. {
        (value.as_f32() * slope + intercept).as_()
    } else {
        value.as_()
    }
}

#[cfg(feature = "ndarray_volumes")]
pub fn convert_bytes_and_cast_to<I, O>(
    raw_data: Vec<u8>,
    e: Endianness,
    dim: &Vec<Ix>,
    slope: O,
    inter: O
    ) -> Array<O, IxDyn>
    where I: AsPrim + PodTransmutable,
          O: AsPrim + Num
{
    let data: Vec<I> = convert_bytes_to(raw_data, e);
    Array::from_shape_vec(IxDyn(&dim).f(), data)
        .expect("Inconsistent raw data size")
        .mapv(|v| raw_to_value::<I, O>(v, slope, inter))
}

#[cfg(feature = "ndarray_volumes")]
pub fn convert_bytes_to<T: PodTransmutable>(
    mut a: Vec<u8>,
    e: Endianness
) -> Vec<T> {
    let nb_bytes = mem::size_of::<T>();
    if e != Endianness::system() && nb_bytes > 1 {
        // Swap endianness by block of nb_bytes
        let split_at = nb_bytes / 2;
        let split_at_m1 = split_at - 1;
        for c in a.chunks_mut(nb_bytes) {
            let (a, b) = c.split_at_mut(split_at);
            for i in 0..split_at {
                mem::swap(&mut a[i], &mut b[split_at_m1 - i]);
            }
        }
    }

    guarded_transmute_pod_vec_permissive(a)
}

pub fn is_gz_file<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref()
        .file_name()
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
    use super::{raw_to_value, raw_to_value_via_f32};
    use super::to_img_file_gz;
    use super::is_gz_file;
    use super::convert_bytes_to;
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
    fn test_convert_vec_i8() {
        assert_eq!(
            convert_bytes_to::<i8>(vec![0x01, 0x11, 0xff], Endianness::BE),
            vec![1, 17, -1]);
        assert_eq!(
            convert_bytes_to::<i8>(vec![0x01, 0x11, 0xfe], Endianness::LE),
            vec![1, 17, -2]);
    }

    #[test]
    fn test_convert_vec_u8() {
        assert_eq!(
            convert_bytes_to::<u8>(vec![0x01, 0x11, 0xff], Endianness::BE),
            vec![1, 17, 255]);
        assert_eq!(
            convert_bytes_to::<u8>(vec![0x01, 0x11, 0xfe], Endianness::LE),
            vec![1, 17, 254]);
    }

    #[test]
    fn test_convert_vec_i16() {
        assert_eq!(
            convert_bytes_to::<i16>(
                vec![0x00, 0x01, 0x01, 0x00, 0xff, 0xfe], Endianness::BE),
            vec![1, 256, -2]);
        assert_eq!(
            convert_bytes_to::<i16>(
                vec![0x01, 0x00, 0x00, 0x01, 0xfe, 0xff], Endianness::LE),
            vec![1, 256, -2]);
    }

    #[test]
    fn test_convert_vec_u16() {
        assert_eq!(
            convert_bytes_to::<u16>(
                vec![0x00, 0x01, 0x01, 0x00, 0xff, 0xfe],
                Endianness::BE),
            vec![1, 256, 65534]);
        assert_eq!(
            convert_bytes_to::<u16>(
                vec![0x01, 0x00, 0x00, 0x01, 0xfe, 0xff],
                Endianness::LE),
            vec![1, 256, 65534]);
    }

    #[test]
    fn test_convert_vec_i32() {
        assert_eq!(
            convert_bytes_to::<i32>(
                vec![0x00, 0x00, 0x00, 0x01,
                     0x01, 0x00, 0x00, 0x00,
                     0xf0, 0xf1, 0xf2, 0xf3],
                Endianness::BE),
            vec![1, 16777216, -252_579_085]);
        assert_eq!(
            convert_bytes_to::<i32>(
                vec![0x01, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x01,
                     0xf3, 0xf2, 0xf1, 0xf0],
                Endianness::LE),
            vec![1, 16777216, -252_579_085]);
    }

    #[test]
    fn test_convert_vec_u32() {
        assert_eq!(
            convert_bytes_to::<u32>(
                vec![0x00, 0x00, 0x00, 0x01,
                     0x01, 0x00, 0x00, 0x00,
                     0xf0, 0xf1, 0xf2, 0xf3],
                Endianness::BE),
            vec![1, 0x01000000, 4_042_388_211]);
        assert_eq!(
            convert_bytes_to::<u32>(
                vec![0x01, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x01,
                     0xf3, 0xf2, 0xf1, 0xf0],
                Endianness::LE),
            vec![1, 0x01000000, 4_042_388_211]);
    }

    #[test]
    fn test_convert_vec_i64() {
        let gt: i64 = 0xf0f1f2f3f4f5f6f7i64;
        assert_eq!(
            convert_bytes_to::<i64>(
                vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
                     0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7],
                Endianness::BE),
            vec![1, 0x0100000000000000, gt]);
        assert_eq!(
            convert_bytes_to::<i64>(
                vec![0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
                     0xf7, 0xf6, 0xf5, 0xf4, 0xf3, 0xf2, 0xf1, 0xf0],
                Endianness::LE),
            vec![1, 0x0100000000000000, gt]);
    }

    #[test]
    fn test_convert_vec_u64() {
        assert_eq!(
            convert_bytes_to::<u64>(
                vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
                     0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7],
                Endianness::BE),
            vec![1, 0x100000000000000, 0xf0f1f2f3f4f5f6f7]);
        assert_eq!(
            convert_bytes_to::<u64>(
                vec![0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
                     0xf7, 0xf6, 0xf5, 0xf4, 0xf3, 0xf2, 0xf1, 0xf0],
                Endianness::LE),
            vec![1, 0x100000000000000, 0xf0f1f2f3f4f5f6f7]);
    }

    #[test]
    fn test_convert_vec_f32() {
        let v: Vec<f32> = convert_bytes_to(
            vec![0x42, 0x28, 0x00, 0x00, 0x42, 0x2A, 0x00, 0x00,],
            Endianness::BE);
        assert_eq!(v, vec![42., 42.5]);

        let v: Vec<f32> = convert_bytes_to(
            vec![0x00, 0x00, 0x28, 0x42, 0x00, 0x00, 0x2A, 0x42],
            Endianness::LE);
        assert_eq!(v, vec![42., 42.5]);
    }

    #[test]
    fn test_convert_vec_f64() {
        let v: Vec<f64> = convert_bytes_to(
            vec![0x40, 0x45, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                 0x40, 0x45, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00],
            Endianness::BE);
        assert_eq!(v, vec![42.0, 42.5]);

        let v: Vec<f64> = convert_bytes_to(
            vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x45, 0x40,
                 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x45, 0x40],
            Endianness::LE);
        assert_eq!(v, vec![42.0, 42.5]);
    }

    #[test]
    fn test_raw_to_value() {
        let raw: u8 = 100;
        let val: u8 = raw_to_value_via_f32(raw, 1., 0.);
        assert_eq!(val, 100);
        let val: u8 = raw_to_value_via_f32(raw, 0., 0.);
        assert_eq!(val, 100);
        let val: f32 = raw_to_value_via_f32(raw, 2., -1024.);
        assert_ulps_eq!(val, -824.);
        let val: i32 = raw_to_value_via_f32(raw, 2., -1024.);
        assert_eq!(val, -824);
        let val: i16 = raw_to_value_via_f32(raw, 2., -1024.);
        assert_eq!(val, -824);

        let raw: f32 = 0.4;
        let val: f32 = raw_to_value(raw, 1., 0.);
        assert_ulps_eq!(val, 0.4);
        let val: f64 = raw_to_value(raw, 1., 0.);
        assert_ulps_eq!(val, 0.4_f32 as f64);
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
