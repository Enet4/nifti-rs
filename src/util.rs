//! Private utility module
use super::error::NiftiError;
use super::typedef::NiftiType;
use crate::error::Result;
use crate::NiftiHeader;
use byteordered::Endian;
use either::Either;
use flate2::bufread::GzDecoder;
use safe_transmute::{transmute_vec, TriviallyTransmutable};
use std::borrow::Cow;
use std::fs::File;
use std::io::{BufReader, Read, Result as IoResult, Seek};
use std::mem;
use std::path::{Path, PathBuf};

/// A trait that is both Read and Seek.
pub trait ReadSeek: Read + Seek {}
impl<T: Read + Seek> ReadSeek for T {}

pub fn convert_bytes_to<T, E>(mut a: Vec<u8>, e: E) -> Vec<T>
where
    T: TriviallyTransmutable,
    E: Endian,
{
    adapt_bytes_inline::<T, _>(&mut a, e);
    match transmute_vec(a) {
        Ok(v) => v,
        Err(safe_transmute::Error::IncompatibleVecTarget(e)) => e.copy(),
        Err(safe_transmute::Error::Unaligned(e)) => e.copy(),
        _ => unreachable!(),
    }
}

/// Adapt a sequence of bytes for reading contiguous values of type `T`,
/// by swapping bytes if the given endianness is not native.
pub fn adapt_bytes_inline<T, E>(a: &mut [u8], e: E)
where
    E: Endian,
{
    let nb_bytes = mem::size_of::<T>();
    if !e.is_native() && nb_bytes > 1 {
        // Swap endianness by block of nb_bytes
        let split_at = nb_bytes / 2;
        for c in a.chunks_mut(nb_bytes) {
            let (a, b) = c.split_at_mut(split_at);
            for (l, r) in a.iter_mut().zip(b.iter_mut().rev()) {
                mem::swap(l, r);
            }
        }
    }
}

/// Adapt a sequence of bytes for reading contiguous values of type `T`,
/// by swapping bytes if the given endianness is not native. If no
/// swapping is needed, the same byte slice is returned.
#[cfg_attr(not(feature = "ndarray_volumes"), allow(dead_code))]
pub fn adapt_bytes<T, E>(bytes: &[u8], e: E) -> Cow<[u8]>
where
    E: Endian,
{
    let nb_bytes = mem::size_of::<T>();
    if !e.is_native() && nb_bytes > 1 {
        // Swap endianness by block of nb_bytes
        let mut a = bytes.to_vec();
        adapt_bytes_inline::<T, E>(&mut a, e);
        a.into()
    } else {
        bytes.into()
    }
}

/// Validate a raw volume dimensions array, returning a slice of the concrete
/// dimensions.
///
/// # Error
///
/// Errors if `dim[0]` is outside the accepted rank boundaries or
/// one of the used dimensions is not positive.
pub fn validate_dim(raw_dim: &[u16; 8]) -> Result<&[u16]> {
    let ndim = validate_dimensionality(raw_dim)?;
    let o = &raw_dim[1..=ndim];
    if let Some(i) = o.iter().position(|&x| x == 0) {
        return Err(NiftiError::InconsistentDim(i as u8, raw_dim[i]));
    }
    Ok(o)
}

/// Validate a raw N-dimensional index or shape, returning its rank.
///
/// # Error
///
/// Errors if `raw_dim[0]` is outside the accepted rank boundaries: 0 or
/// larger than 7.
pub fn validate_dimensionality(raw_dim: &[u16; 8]) -> Result<usize> {
    if raw_dim[0] == 0 || raw_dim[0] > 7 {
        return Err(NiftiError::InconsistentDim(0, raw_dim[0]));
    }
    Ok(usize::from(raw_dim[0]))
}

pub fn nb_bytes_for_data(header: &NiftiHeader) -> Result<usize> {
    let resolution = nb_values_for_dims(header.dim()?);
    resolution
        .and_then(|r| r.checked_mul(usize::from(header.get_bitpix()) / 8))
        .ok_or(NiftiError::BadVolumeSize)
}

pub fn nb_values_for_dims(dim: &[u16]) -> Option<usize> {
    dim.iter()
        .cloned()
        .map(usize::from)
        .fold(Some(1), |acc, v| acc.and_then(|x| x.checked_mul(v)))
}

pub fn nb_bytes_for_dim_datatype(dim: &[u16], datatype: NiftiType) -> Option<usize> {
    let resolution = nb_values_for_dims(dim);
    resolution.and_then(|r| r.checked_mul(datatype.size_of()))
}

#[cfg(feature = "ndarray_volumes")]
pub fn is_hdr_file<P>(path: P) -> bool
where
    P: AsRef<Path>,
{
    path.as_ref()
        .file_name()
        .map(|a| {
            let s = a.to_string_lossy();
            s.ends_with(".hdr") || s.ends_with(".hdr.gz")
        })
        .unwrap_or(false)
}

pub fn is_gz_file<P>(path: P) -> bool
where
    P: AsRef<Path>,
{
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
pub fn into_img_file_gz(mut path: PathBuf) -> PathBuf {
    if is_gz_file(&path) {
        // Leave only the first extension (.hdr)
        let _ = path.set_extension("");
    }
    path.with_extension("img.gz")
}

/// A reader for a GZip encoded file.
pub type GzDecodedFile = GzDecoder<BufReader<File>>;

/// A byte reader which might be GZip encoded based on some run-time condition.
pub type MaybeGzDecoded<T> = Either<T, GzDecoder<T>>;

/// A reader for a file which might be GZip encoded based on some run-time
/// condition.
pub type MaybeGzDecodedFile = MaybeGzDecoded<BufReader<File>>;

/// Open a file for reading, which might be Gzip compressed based on whether
/// the extension ends with ".gz".
pub fn open_file_maybe_gz<P>(path: P) -> IoResult<MaybeGzDecodedFile>
where
    P: AsRef<Path>,
{
    let path = path.as_ref();
    let file = BufReader::new(File::open(path)?);
    if is_gz_file(path) {
        Ok(Either::Right(GzDecoder::new(file)))
    } else {
        Ok(Either::Left(file))
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "ndarray_volumes")]
    use super::is_hdr_file;
    use super::{into_img_file_gz, is_gz_file, nb_bytes_for_dim_datatype};
    use crate::typedef::NiftiType;
    use std::path::PathBuf;

    #[test]
    fn test_nbytes() {
        assert_eq!(
            nb_bytes_for_dim_datatype(&[2, 3, 2], NiftiType::Uint8),
            Some(12),
        );
        assert_eq!(
            nb_bytes_for_dim_datatype(&[2, 3], NiftiType::Uint8),
            Some(6),
        );
        assert_eq!(
            nb_bytes_for_dim_datatype(&[2, 3], NiftiType::Uint16),
            Some(12),
        );
        assert_eq!(
            nb_bytes_for_dim_datatype(&[0x4000, 0x4000, 0x4000, 0x4000, 0x4000], NiftiType::Uint32),
            None,
        );
    }

    #[test]
    fn filenames() {
        assert!(!is_gz_file("/path/to/something.nii"));
        assert!(is_gz_file("/path/to/something.nii.gz"));
        assert!(!is_gz_file("volume.não"));
        assert!(is_gz_file("1.2.3.nii.gz"));
        assert!(!is_gz_file("não_é_gz.hdr"));

        let path = "/path/to/image.hdr";
        #[cfg(feature = "ndarray_volumes")]
        assert!(is_hdr_file(path));
        assert!(!is_gz_file(path));
        assert_eq!(
            into_img_file_gz(PathBuf::from(path)),
            PathBuf::from("/path/to/image.img.gz")
        );

        let path = "/path/to/image.hdr.gz";
        #[cfg(feature = "ndarray_volumes")]
        assert!(is_hdr_file(path));
        assert!(is_gz_file(path));
        assert_eq!(
            into_img_file_gz(PathBuf::from(path)),
            PathBuf::from("/path/to/image.img.gz")
        );

        let path = "my_ct_scan.1.hdr.gz";
        #[cfg(feature = "ndarray_volumes")]
        assert!(is_hdr_file(path));
        assert!(is_gz_file(path));
        assert_eq!(
            into_img_file_gz(PathBuf::from(path)),
            PathBuf::from("my_ct_scan.1.img.gz")
        );

        assert_eq!(
            into_img_file_gz(PathBuf::from("../you.cant.fool.me.hdr.gz")),
            PathBuf::from("../you.cant.fool.me.img.gz")
        );
    }
}

#[cfg(feature = "ndarray_volumes")]
#[cfg(test)]
mod test_nd_array {
    use super::convert_bytes_to;
    use byteordered::Endianness;

    #[test]
    fn test_convert_vec_i8() {
        assert_eq!(
            convert_bytes_to::<i8, _>(vec![0x01, 0x11, 0xff], Endianness::Big),
            vec![1, 17, -1]
        );
        assert_eq!(
            convert_bytes_to::<i8, _>(vec![0x01, 0x11, 0xfe], Endianness::Little),
            vec![1, 17, -2]
        );
    }

    #[test]
    fn test_convert_vec_u8() {
        assert_eq!(
            convert_bytes_to::<u8, _>(vec![0x01, 0x11, 0xff], Endianness::Big),
            vec![1, 17, 255]
        );
        assert_eq!(
            convert_bytes_to::<u8, _>(vec![0x01, 0x11, 0xfe], Endianness::Little),
            vec![1, 17, 254]
        );
    }

    #[test]
    fn test_convert_vec_i16() {
        assert_eq!(
            convert_bytes_to::<i16, _>(vec![0x00, 0x01, 0x01, 0x00, 0xff, 0xfe], Endianness::Big),
            vec![1, 256, -2]
        );
        assert_eq!(
            convert_bytes_to::<i16, _>(
                vec![0x01, 0x00, 0x00, 0x01, 0xfe, 0xff],
                Endianness::Little
            ),
            vec![1, 256, -2]
        );
    }

    #[test]
    fn test_convert_vec_u16() {
        assert_eq!(
            convert_bytes_to::<u16, _>(vec![0x00, 0x01, 0x01, 0x00, 0xff, 0xfe], Endianness::Big),
            vec![1, 256, 65534]
        );
        assert_eq!(
            convert_bytes_to::<u16, _>(
                vec![0x01, 0x00, 0x00, 0x01, 0xfe, 0xff],
                Endianness::Little
            ),
            vec![1, 256, 65534]
        );
    }

    #[test]
    fn test_convert_vec_i32() {
        assert_eq!(
            convert_bytes_to::<i32, _>(
                vec![0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0xf0, 0xf1, 0xf2, 0xf3],
                Endianness::Big
            ),
            vec![1, 16777216, -252_579_085]
        );
        assert_eq!(
            convert_bytes_to::<i32, _>(
                vec![0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0xf3, 0xf2, 0xf1, 0xf0],
                Endianness::Little
            ),
            vec![1, 16777216, -252_579_085]
        );
    }

    #[test]
    fn test_convert_vec_u32() {
        assert_eq!(
            convert_bytes_to::<u32, _>(
                vec![0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0xf0, 0xf1, 0xf2, 0xf3],
                Endianness::Big
            ),
            vec![1, 0x01000000, 4_042_388_211]
        );
        assert_eq!(
            convert_bytes_to::<u32, _>(
                vec![0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0xf3, 0xf2, 0xf1, 0xf0],
                Endianness::Little
            ),
            vec![1, 0x01000000, 4_042_388_211]
        );
    }

    #[test]
    fn test_convert_vec_i64() {
        assert_eq!(
            convert_bytes_to::<i64, _>(
                vec![
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
                ],
                Endianness::Big
            ),
            vec![1, 0x0100000000000000, -1_084_818_905_618_843_913]
        );
        assert_eq!(
            convert_bytes_to::<i64, _>(
                vec![
                    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x01, 0xf7, 0xf6, 0xf5, 0xf4, 0xf3, 0xf2, 0xf1, 0xf0,
                ],
                Endianness::Little
            ),
            vec![1, 0x0100000000000000, -1_084_818_905_618_843_913]
        );
    }

    #[test]
    fn test_convert_vec_u64() {
        assert_eq!(
            convert_bytes_to::<u64, _>(
                vec![
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
                ],
                Endianness::Big
            ),
            vec![1, 0x100000000000000, 0xf0f1f2f3f4f5f6f7]
        );
        assert_eq!(
            convert_bytes_to::<u64, _>(
                vec![
                    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x01, 0xf7, 0xf6, 0xf5, 0xf4, 0xf3, 0xf2, 0xf1, 0xf0,
                ],
                Endianness::Little
            ),
            vec![1, 0x100000000000000, 0xf0f1f2f3f4f5f6f7]
        );
    }

    #[test]
    fn test_convert_vec_f32() {
        let v: Vec<f32> = convert_bytes_to(
            vec![0x42, 0x28, 0x00, 0x00, 0x42, 0x2A, 0x00, 0x00],
            Endianness::Big,
        );
        assert_eq!(v, vec![42., 42.5]);

        let v: Vec<f32> = convert_bytes_to(
            vec![0x00, 0x00, 0x28, 0x42, 0x00, 0x00, 0x2A, 0x42],
            Endianness::Little,
        );
        assert_eq!(v, vec![42., 42.5]);
    }

    #[test]
    fn test_convert_vec_f64() {
        let v: Vec<f64> = convert_bytes_to(
            vec![
                0x40, 0x45, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x45, 0x40, 0x00, 0x00, 0x00,
                0x00, 0x00,
            ],
            Endianness::Big,
        );
        assert_eq!(v, vec![42.0, 42.5]);

        let v: Vec<f64> = convert_bytes_to(
            vec![
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x45, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
                0x45, 0x40,
            ],
            Endianness::Little,
        );
        assert_eq!(v, vec![42.0, 42.5]);
    }
}
