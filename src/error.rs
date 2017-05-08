use std::io::Error as IOError;
use typedef::NiftiType;

quick_error! {
    #[derive(Debug)]
    pub enum NiftiError {
        /// Read an invalid NIfTI-1 file
        InvalidFormat {
            description("Invalid NIfTI-1 file")
        }
        /// Attempted to read volume outside boundaries.
        OutOfBounds {
            description("Out of bounds access to volume")
        }
        NoVolumeData {
            description("No volume data available")
        }
        UnsupportedDataType(t: NiftiType) {
            description("Unsupported data type")
        }
        /// I/O Error
        Io(err: IOError) {
            from()
            cause(err)
            description(err.description())
        }

    }
}

pub type Result<T> = ::std::result::Result<T, NiftiError>;
