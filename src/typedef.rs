use byteorder::ReadBytesExt;
use error::{Result, NiftiError};
use std::io::Read;
use std::ops::{Add, Mul};
use util::{Endianness, raw_to_value};

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, FromPrimitive)]
pub enum NiftiType {
    /// unsigned char.
    // NIFTI_TYPE_UINT8           2
    Uint8 = 2,
    /// signed short.
    // NIFTI_TYPE_INT16           4
    Int16 = 4,
    /// signed int.
    // NIFTI_TYPE_INT32           8
    Int32 = 8,
    /// 32 bit float.
    // NIFTI_TYPE_FLOAT32        16
    Float32 = 16,
    /// 64 bit complex = 2 32 bit floats.
    // NIFTI_TYPE_COMPLEX64      32
    Complex64 = 32,
    /// 64 bit float = double.
    // NIFTI_TYPE_FLOAT64        64
    Float64 = 64,
    /// 3 8 bit bytes.
    // NIFTI_TYPE_RGB24         128
    Rgb24 = 128,
    /// signed char.
    // NIFTI_TYPE_INT8          256
    Int8 = 256,
    /// unsigned short.
    // NIFTI_TYPE_UINT16        512
    Uint16 = 512,
    /// unsigned int.
    // NIFTI_TYPE_UINT32        768
    Uint32 = 768,
    /// signed long long.
    // NIFTI_TYPE_INT64        1024
    Int64 = 1024,
    /// unsigned long long.
    // NIFTI_TYPE_UINT64       1280
    Uint64 = 1280,
    /// 128 bit float = long double.
    // NIFTI_TYPE_FLOAT128     1536
    Float128 = 1536,
    /// 128 bit complex = 2 64 bit floats.
    // NIFTI_TYPE_COMPLEX128   1792
    Complex128 = 1792,
    /// 256 bit complex = 2 128 bit floats
    // NIFTI_TYPE_COMPLEX256   2048
    Complex256 = 2048,
    /// 4 8 bit bytes.
    // NIFTI_TYPE_RGBA32       2304
    Rgba32 = 2304,
}

impl NiftiType {
    pub fn read_primitive_value<S, T>(&self, mut source: S, endianness: Endianness, slope: f32, inter: f32) -> Result<T>
        where S: Read,
              T: From<f32>,
              T: Add<Output = T>,
              T: Mul<Output = T>
    {
       match *self {
            NiftiType::Uint8 => {
                let raw = source.read_u8()?;
                Ok(raw_to_value(raw as f32, slope, inter))
            },
            NiftiType::Uint16 => {
                let raw = endianness.read_u16(source)?;
                Ok(raw_to_value(raw as f32, slope, inter))
            },
            NiftiType::Int16 => {
                let raw = endianness.read_i16(source)?;
                Ok(raw_to_value(raw as f32, slope, inter))
            },
            NiftiType::Uint32 => {
                let raw = endianness.read_u32(source)?;
                Ok(raw_to_value(raw as f32, slope, inter))
            },
            NiftiType::Int32 => {
                let raw = endianness.read_i32(source)?;
                Ok(raw_to_value(raw as f32, slope, inter))
            },
            NiftiType::Uint64 => {
                let raw = endianness.read_u64(source)?;
                Ok(raw_to_value(raw as f32, slope, inter))
            },
            NiftiType::Int64 => {
                let raw = endianness.read_i64(source)?;
                Ok(raw_to_value(raw as f32, slope, inter))
            },
            NiftiType::Float32 => {
                let raw = endianness.read_f32(source)?;
                Ok(raw_to_value(raw, slope, inter))
            },
            NiftiType::Float64 => {
                let raw = endianness.read_f64(source)?;
                Ok(raw_to_value(raw as f32, slope, inter))

            },
            // TODO add support for more data types
            _ => Err(NiftiError::UnsupportedDataType(*self))
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, FromPrimitive)]
pub enum Unit {
    Unknown = 0,
    Meter = 1,
    Mm = 2,
    Micron = 3,
    Sec = 8,
    Msec = 16,
    Usec = 24,
    Hz = 32,
    Ppm = 40,
    Rads = 48,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, FromPrimitive)]
pub enum Intent {
    None = 0,
    Correl = 2,
    Ttest = 3,
    Ftest = 4,
    Zscore = 5,
    Chisq = 6,
    Beta = 7,
    Binom = 8,
    Gamma = 9,
    Poisson = 10,
    Normal = 11,
    FtestNonc = 12,
    ChisqNonc = 13,
    Logistic = 14,
    Laplace = 15,
    Uniform = 16,
    TtestNonc = 17,
    Weibull = 18,
    Chi = 19,
    Invgauss = 20,
    Extval = 21,
    Pval = 22,
    Logpval = 23,
    Log10pval = 24,
    /* --- these values aren't for statistics --- */
    Estimate = 1001,
    Label = 1002,
    Neuroname = 1003,
    Genmatrix = 1004,
    Symmatrix = 1005,
    Dispvect = 1006,
    Vector = 1007,
    Pointset = 1008,
    Triangle = 1009,
    Dimless = 1011,
    /* --- these values apply to GIFTI datasets --- */
    TimeSeries = 2001,
    NodeIndex = 2002,
    RgbaVector = 2003,
    Shape = 2005,
}

impl Intent {
    pub fn is_statcode(&self) -> bool {
        self.get_code() >= 2 && self.get_code() <= 24
    }

    pub fn get_code(&self) -> i16 { *self as i16 }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, FromPrimitive)]
pub enum XForm {
    /// Arbitrary coordinates (Method 1).
    Unknown = 0,
    /// Scanner-based anatomical coordinates
    ScannerAnat = 1,
    /// Coordinates aligned to another file's,
    /// or to anatomical "truth".
    AlignedAnat = 2,
    /// Coordinates aligned to Talairach-Tournoux
    /// Atlas; (0,0,0)=AC, etc.
    Talairach = 3,
    /// MNI 152 normalized coordinates.
    Mni152 = 4,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, FromPrimitive)]
pub enum SliceOrder {
    Unknown = 0,
    SeqInc = 1,
    SeqDec = 2,
    AltInc = 3,
    AltDec = 4,
    AltInc2 = 5,
    AltDec2 = 6,
}
