//! Types for error handling go here.
use std::io::Error as IOError;
use typedef::NiftiType;

quick_error! {
    /// Error type for all error variants originated by this crate.
    #[derive(Debug)]
    pub enum NiftiError {
        /// An invalid NIfTI-1 file was parsed.
        /// This is detected when reading the file's magic code,
        /// which should be either `b"ni1\0"` or `b"n+1\0`.
        InvalidFormat {
            description("Invalid NIfTI-1 file")
        }
        /// The header contains a field in an invalid state.
        InconsistentHeader(name: &'static str) {
            description("Inconsistent file header")
            display("Inconsistent value of header field {}", name)
        }
        /// Attempted to read volume outside boundaries.
        OutOfBounds(coords: Vec<u16>) {
            description("Out of bounds access to volume")
        }
        /// Attempted to read a volume over a volume's unexistent dimension.
        AxisOutOfBounds(axis: u16) {
            description("Out of bounds access to volume")
        }
        /// Could not retrieve a volume file based on the given header file.
        MissingVolumeFile(err: IOError) {
            cause(err)
            description("Volume file not found")
        }
        /// An attempt to read a complete NIFTI-1 object from a header file
        /// was made. It can also be triggered when a NIFTI object contains
        /// the magic code "ni-1\0", even if the following bytes contain the volume.
        NoVolumeData {
            description("No volume data available")
        }
        /// An incorrect number of dimensions was provided when interacting
        /// with a volume.
        IncorrectVolumeDimensionality(expected: u16, got: u16) {
            description("Unexpected volume data dimensionality")
        }
        /// This voxel data type is not supported. Sorry. :(
        UnsupportedDataType(t: NiftiType) {
            description("Unsupported data type")
        }
        /// I/O Error
        Io(err: IOError) {
            from()
            cause(err)
            description(err.description())
        }
        /// Raw data and buffer length are incompatible
        IncompatibleLength {
            description("The buffer length and the header dimensions are incompatible.")
        }
        /// Description length must be lower than or equal to 80 bytes
        IncorrectDescriptionLength(len: usize) {
            description("Description length is greater than 80 bytes.")
            display("Description length ({} bytes) is greater than 80 bytes.", len)
        }
        /// Header contains a code which is not valid for the given attribute
        InvalidCode(typename: &'static str, code: i16) {
            description("invalid code")
            display("invalid code `{}` for header field {}", code, typename)
        }
    }
}

/// Alias type for results originated from this crate.
pub type Result<T> = ::std::result::Result<T, NiftiError>;
