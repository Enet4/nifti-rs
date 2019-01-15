//! Private utility module
use std::io::{Read, Seek};
use std::mem;
use std::path::{Path, PathBuf};
use byteordered::Endian;

use safe_transmute::{guarded_transmute_pod_vec_permissive, PodTransmutable};

use NiftiHeader;

/// A trait that is both Read and Seek.
pub trait ReadSeek: Read + Seek {}
impl<T: Read + Seek> ReadSeek for T {}

pub fn convert_bytes_to<T, E>(mut a: Vec<u8>, e: E) -> Vec<T>
where
    T: PodTransmutable,
    E: Endian,
{
    adapt_bytes::<T, _>(&mut a, e);
    guarded_transmute_pod_vec_permissive(a)
}

/// Adapt a sequence of bytes for reading contiguous values of type `T`,
/// by swapping bytes if the given endianness is not native.
pub fn adapt_bytes<T, E>(a: &mut [u8], e: E)
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

pub fn nb_bytes_for_data(header: &NiftiHeader) -> usize {
    let ndims = header.dim[0];
    let resolution: usize = header.dim[1..(ndims + 1) as usize]
        .iter()
        .map(|d| *d as usize)
        .product();
    resolution * header.bitpix as usize / 8
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

#[cfg(test)]
mod tests {
    use super::into_img_file_gz;
    use super::is_gz_file;
    #[cfg(feature = "ndarray_volumes")]
    use super::is_hdr_file;
    use std::path::PathBuf;

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
    use byteordered::Endianness;
    use super::convert_bytes_to;

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
            convert_bytes_to::<i16, _>(vec![0x01, 0x00, 0x00, 0x01, 0xfe, 0xff], Endianness::Little),
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
            convert_bytes_to::<u16, _>(vec![0x01, 0x00, 0x00, 0x01, 0xfe, 0xff], Endianness::Little),
            vec![1, 256, 65534]
        );
    }

    #[test]
    fn test_convert_vec_i32() {
        assert_eq!(
            convert_bytes_to::<i32, _>(
                vec![
                    0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0xf0, 0xf1, 0xf2, 0xf3
                ],
                Endianness::Big
            ),
            vec![1, 16777216, -252_579_085]
        );
        assert_eq!(
            convert_bytes_to::<i32, _>(
                vec![
                    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0xf3, 0xf2, 0xf1, 0xf0
                ],
                Endianness::Little
            ),
            vec![1, 16777216, -252_579_085]
        );
    }

    #[test]
    fn test_convert_vec_u32() {
        assert_eq!(
            convert_bytes_to::<u32, _>(
                vec![
                    0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0xf0, 0xf1, 0xf2, 0xf3
                ],
                Endianness::Big
            ),
            vec![1, 0x01000000, 4_042_388_211]
        );
        assert_eq!(
            convert_bytes_to::<u32, _>(
                vec![
                    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0xf3, 0xf2, 0xf1, 0xf0
                ],
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
