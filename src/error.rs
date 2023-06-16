//! Types for error handling go here.
use crate::typedef::NiftiType;
use quick_error::quick_error;
use std::io::Error as IOError;

quick_error! {
    /// Error type for all error variants originated by this crate.
    #[derive(Debug)]
    #[non_exhaustive]
    pub enum NiftiError {
        /// An invalid NIfTI-1 file was parsed.
        /// This is detected when reading the file's magic code,
        /// which should be either `b"ni1\0"` or `b"n+1\0`.
        InvalidFormat {
            display("Invalid NIfTI-1 file")
        }
        /// The field `dim` is in an invalid state, as a consequence of
        /// `dim[0]` or one of the elements in `1..dim[0] + 1` not being
        /// positive.
        InconsistentDim(index: u8, value: u16) {
            display("Inconsistent value `{}` in header field dim[{}] ({})", value, index, match index {
                0 if *value > 7 => "must not be higher than 7",
                _ => "must be positive"
            })
        }
        /// Attempted to read volume outside boundaries.
        OutOfBounds(coords: Vec<u16>) {
            display("Out of bounds access to volume: {:?}", &coords[..])
        }
        /// Attempted to read a volume over a volume's unexistent dimension.
        AxisOutOfBounds(axis: u16) {
            display("Out of bounds access to volume (axis {})", axis)
        }
        /// Could not retrieve a volume file based on the given header file.
        MissingVolumeFile(err: IOError) {
            source(err)
            display("Volume file not found")
        }
        /// An attempt to read a complete NIFTI-1 object from a header file
        /// was made. It can also be triggered when a NIFTI object contains
        /// the magic code "ni-1\0", even if the following bytes contain the volume.
        NoVolumeData {
            display("No volume data available")
        }
        /// An incorrect number of dimensions was provided when interacting
        /// with a volume.
        IncorrectVolumeDimensionality(expected: u16, got: u16) {
            display("Unexpected volume data dimensionality (expected {}, got {})", expected, got)
        }
        /// Inconsistent or unsupported volume size (due to one or more
        /// dimensions being too large).
        BadVolumeSize {
            display("Bad volume size")
        }
        /// This voxel data type is not supported. Sorry. :(
        UnsupportedDataType(t: NiftiType) {
            display("Unsupported data type")
        }
        /// I/O Error
        Io(err: IOError) {
            from()
            source(err)
        }
        /// Raw data buffer length and volume dimensions are incompatible
        IncompatibleLength(got: usize, expected: usize) {
            display("The buffer length ({}) and header dimensions ({} elements) are incompatible", got, expected)
        }
        /// Description length must be lower than or equal to 80 bytes
        IncorrectDescriptionLength(len: usize) {
            display("Description length ({} bytes) is greater than 80 bytes.", len)
        }
        /// Header contains a code which is not valid for the given attribute
        InvalidCode(typename: &'static str, code: i16) {
            display("invalid code `{}` for header field {}", code, typename)
        }
        /// Could not reserve enough memory for volume data
        ReserveVolume(bytes: usize, err: std::collections::TryReserveError) {
            display("Could not reserve {} bytes of memory for volume data", bytes)
            source(err)
        }
        /// Could not reserve enough memory for extended data
        ReserveExtended(bytes: usize, err: std::collections::TryReserveError) {
            display("Could not reserve {} bytes of memory for extended data", bytes)
            source(err)
        }

        InvalidTypeConversion(from: NiftiType, to: &'static str) {
            display("Invalid type conversion from {:?} to {}", from, to)
        }
    }
}

/// Alias type for results originated from this crate.
pub type Result<T> = ::std::result::Result<T, NiftiError>;
