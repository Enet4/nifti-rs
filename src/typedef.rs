//! This module contains multiple types defined by the standard.
//! At the moment, not all of them are used internally (`NiftiType`
//! makes the exception, which also provides a safe means of
//! reading voxel values). However, primitive integer values can be
//! converted to these types and vice-versa.

use byteorder::ReadBytesExt;
use error::{NiftiError, Result};
use std::io::Read;
use std::ops::{Add, Mul};
use util::{raw_to_value, Endianness};
use num::Num;

/// Data type for representing a NIFTI value type in a volume.
/// Methods for reading values of that type from a source are also included.
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
    /// Retrieve the size of an element of this data type, in bytes.
    pub fn size_of(&self) -> usize {
        use NiftiType::*;
        match *self {
            Int8 | Uint8 => 1,
            Int16 | Uint16 => 2,
            Rgb24 => 3,
            Int32 | Uint32 | Float32 | Rgba32 => 4,
            Int64 | Uint64 | Float64 | Complex64 => 8,
            Float128 | Complex128 => 16,
            Complex256 => 32,
        }
    }
}

impl NiftiType {
    /// Read a primitive voxel value from a source.
    pub fn read_primitive_value<S, T>(
        &self,
        mut source: S,
        endianness: Endianness,
        slope: f32,
        inter: f32,
    ) -> Result<T>
    where
        S: Read,
        T: From<f32>,
        T: Num,
        T: Add<Output = T>,
        T: Mul<Output = T>,
    {
        let slope: T = slope.into();
        let inter: T = inter.into();
        match *self {
            NiftiType::Uint8 => {
                let raw = source.read_u8()?;
                Ok(raw_to_value(raw as f32, slope, inter))
            }
            NiftiType::Uint16 => {
                let raw = endianness.read_u16(source)?;
                Ok(raw_to_value(raw as f32, slope, inter))
            }
            NiftiType::Int16 => {
                let raw = endianness.read_i16(source)?;
                Ok(raw_to_value(raw as f32, slope, inter))
            }
            NiftiType::Uint32 => {
                let raw = endianness.read_u32(source)?;
                Ok(raw_to_value(raw as f32, slope, inter))
            }
            NiftiType::Int32 => {
                let raw = endianness.read_i32(source)?;
                Ok(raw_to_value(raw as f32, slope, inter))
            }
            NiftiType::Uint64 => {
                // TODO find a way to not lose precision
                let raw = endianness.read_u64(source)?;
                Ok(raw_to_value(raw as f32, slope, inter))
            }
            NiftiType::Int64 => {
                // TODO find a way to not lose precision
                let raw = endianness.read_i64(source)?;
                Ok(raw_to_value(raw as f32, slope, inter))
            }
            NiftiType::Float32 => {
                let raw = endianness.read_f32(source)?;
                Ok(raw_to_value(raw, slope, inter))
            }
            NiftiType::Float64 => {
                // TODO find a way to not lose precision
                let raw = endianness.read_f64(source)?;
                Ok(raw_to_value(raw as f32, slope, inter))
            }
            // TODO add support for more data types
            _ => Err(NiftiError::UnsupportedDataType(*self)),
        }
    }
}

/// An enum type which represents a unit type.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, FromPrimitive)]
pub enum Unit {
    /// NIFTI code for unspecified units.
    Unknown = 0,
    /* Space codes are multiples of 1. */
    /// NIFTI code for meters.
    Meter = 1,
    /// NIFTI code for millimeters.
    Mm = 2,
    /// NIFTI code for micrometers.
    Micron = 3,
    /* Time codes are multiples of 8. */
    /// NIFTI code for seconds.
    Sec = 8,
    /// NIFTI code for milliseconds.
    Msec = 16,
    /// NIFTI code for microseconds.
    Usec = 24,
    /* These units are for spectral data: */
    /// NIFTI code for Hertz.
    Hz = 32,
    /// NIFTI code for ppm.
    Ppm = 40,
    /// NIFTI code for radians per second.
    Rads = 48,
}

/// An enum type for representing a NIFTI intent code.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, FromPrimitive)]
pub enum Intent {
    /// default: no intention is indicated in the header.
    None = 0,
    /// nifti1 intent codes, to describe intended meaning of dataset contents
    Correl = 2,
    /// [C2, chap 28] Student t statistic (1 param): p1 = DOF.
    Ttest = 3,
    /// [C2, chap 27] Fisher F statistic (2 params):
    /// p1 = numerator DOF, p2 = denominator DOF.
    Ftest = 4,
    /// [C1, chap 13] Standard normal (0 params): Density = N(0,1).
    Zscore = 5,
    /// [C1, chap 18] Chi-squared (1 param): p1 = DOF.
    /// Density(x) proportional to exp(-x/2) * x^(p1/2-1).
    Chisq = 6,
    /// [C2, chap 25] Beta distribution (2 params): p1=a, p2=b.
    /// Density(x) proportional to x^(a-1) * (1-x)^(b-1).
    Beta = 7,
    /// [U, chap 3] Binomial distribution (2 params):
    /// p1 = number of trials, p2 = probability per trial.
    /// Prob(x) = (p1 choose x) * p2^x * (1-p2)^(p1-x), for x=0,1,...,p1.
    Binom = 8,
    /// [C1, chap 17] Gamma distribution (2 params):
    /// p1 = shape, p2 = scale.
    /// Density(x) proportional to x^(p1-1) * exp(-p2*x).
    Gamma = 9,
    /// [U, chap 4] Poisson distribution (1 param): p1 = mean.
    /// Prob(x) = exp(-p1) * p1^x / x! , for x=0,1,2,....
    Poisson = 10,
    /// [C1, chap 13] Normal distribution (2 params):
    /// p1 = mean, p2 = standard deviation.
    Normal = 11,
    /// [C2, chap 30] Noncentral F statistic (3 params):
    /// p1 = numerator DOF, p2 = denominator DOF,
    /// p3 = numerator noncentrality parameter.
    FtestNonc = 12,
    /// [C2, chap 29] Noncentral chi-squared statistic (2 params):
    /// p1 = DOF, p2 = noncentrality parameter.
    ChisqNonc = 13,
    /// [C2, chap 23] Logistic distribution (2 params):
    /// p1 = location, p2 = scale.
    /// Density(x) proportional to sech^2((x-p1)/(2*p2)).
    Logistic = 14,
    /// [C2, chap 24] Laplace distribution (2 params):
    /// p1 = location, p2 = scale.
    /// Density(x) proportional to exp(-abs(x-p1)/p2).
    Laplace = 15,
    /// [C2, chap 26] Uniform distribution: p1 = lower end, p2 = upper end.
    Uniform = 16,
    /// [C2, chap 31] Noncentral t statistic (2 params):
    /// p1 = DOF, p2 = noncentrality parameter.
    TtestNonc = 17,
    /// [C1, chap 21] Weibull distribution (3 params):
    /// p1 = location, p2 = scale, p3 = power.
    /// Density(x) proportional to
    /// ((x-p1)/p2)^(p3-1) * exp(-((x-p1)/p2)^p3) for x > p1.
    Weibull = 18,
    /// [C1, chap 18] Chi distribution (1 param): p1 = DOF.
    /// Density(x) proportional to x^(p1-1) * exp(-x^2/2) for x > 0.
    /// p1 = 1 = 'half normal' distribution
    /// p1 = 2 = Rayleigh distribution
    /// p1 = 3 = Maxwell-Boltzmann distribution.
    Chi = 19,
    /// [C1, chap 15] Inverse Gaussian (2 params):
    /// p1 = mu, p2 = lambda
    /// Density(x) proportional to
    /// exp(-p2*(x-p1)^2/(2*p1^2*x)) / x^3  for x > 0.
    Invgauss = 20,
    /// [C2, chap 22] Extreme value type I (2 params):
    /// p1 = location, p2 = scale
    /// cdf(x) = exp(-exp(-(x-p1)/p2)).
    Extval = 21,
    /// Data is a 'p-value' (no params).
    Pval = 22,
    /// Data is ln(p-value) (no params).
    /// To be safe, a program should compute p = exp(-abs(this_value)).
    /// The nifti_stats.c library returns this_value
    /// as positive, so that this_value = -log(p).
    Logpval = 23,
    /// Data is log10(p-value) (no params).
    /// To be safe, a program should compute p = pow(10.,-abs(this_value)).
    /// The nifti_stats.c library returns this_value
    /// as positive, so that this_value = -log10(p).
    Log10pval = 24,
    /* --- these values aren't for statistics --- */
    /// To signify that the value at each voxel is an estimate
    /// of some parameter, set intent_code = `NIFTI_INTENT_ESTIMATE`.
    /// The name of the parameter may be stored in intent_name.
    Estimate = 1001,
    /// To signify that the value at each voxel is an index into
    /// some set of labels, set intent_code = `NIFTI_INTENT_LABEL`.
    /// The filename with the labels may stored in aux_file.
    Label = 1002,
    /// To signify that the value at each voxel is an index into the
    /// NeuroNames labels set, set intent_code = `NIFTI_INTENT_NEURONAME`.
    Neuroname = 1003,
    /// To store an M x N matrix at each voxel:
    ///    - dataset must have a 5th dimension (dim[0]=5 and dim[5]>1)
    ///    - intent_code must be `NIFTI_INTENT_GENMATRIX`
    ///    - dim[5] must be M*N
    ///    - intent_p1 must be M (in float format)
    ///    - intent_p2 must be N (ditto)
    ///    - the matrix values A[i][[j] are stored in row-order:
    ///       - A[0][0] A[0][1] ... A[0][N-1]
    ///       - A[1][0] A[1][1] ... A[1][N-1]
    ///       - etc., until
    ///       - A[M-1][0] A[M-1][1] ... A[M-1][N-1]
    Genmatrix = 1004,
    /// To store an NxN symmetric matrix at each voxel:
    ///    - dataset must have a 5th dimension
    ///    - intent_code must be `NIFTI_INTENT_SYMMATRIX`
    ///    - dim[5] must be N*(N+1)/2
    ///    - intent_p1 must be N (in float format)
    ///    - the matrix values A[i][[j] are stored in row-order:
    ///       - A[0][0]
    ///       - A[1][0] A[1][1]
    ///       - A[2][0] A[2][1] A[2][2]
    ///       - etc.: row-by-row
    Symmatrix = 1005,
    /// To signify that the vector value at each voxel is to be taken
    /// as a displacement field or vector:
    ///   - dataset must have a 5th dimension
    ///   - intent_code must be `NIFTI_INTENT_DISPVECT`
    ///   - dim[5] must be the dimensionality of the displacment
    ///     vector (e.g., 3 for spatial displacement, 2 for in-plane)
    ///
    /// (specifically for displacements)
    Dispvect = 1006,
    /// (for any other type of vector)
    Vector = 1007,
    /// To signify that the vector value at each voxel is really a
    /// spatial coordinate (e.g., the vertices or nodes of a surface mesh):
    ///    - dataset must have a 5th dimension
    ///    - intent_code must be `NIFTI_INTENT_POINTSET`
    ///    - dim[0] = 5
    ///    - dim[1] = number of points
    ///    - dim[2] = dim[3] = dim[4] = 1
    ///    - dim[5] must be the dimensionality of space (e.g., 3 => 3D space).
    ///    - intent_name may describe the object these points come from
    ///      (e.g., "pial", "gray/white" , "EEG", "MEG").
    Pointset = 1008,
    /// To signify that the vector value at each voxel is really a triple
    /// of indexes (e.g., forming a triangle) from a pointset dataset:
    ///    - dataset must have a 5th dimension
    ///    - intent_code must be `NIFTI_INTENT_TRIANGLE`
    ///    - dim[0] = 5
    ///    - dim[1] = number of triangles
    ///    - dim[2] = dim[3] = dim[4] = 1
    ///    - dim[5] = 3
    ///    - datatype should be an integer type (preferably `NiftiType::Int32`)
    ///    - the data values are indexes (0,1,...) into a pointset dataset.
    Triangle = 1009,
    /// To signify that the vector value at each voxel is a quaternion:
    ///    - dataset must have a 5th dimension
    ///    - intent_code must be `NIFTI_INTENT_QUATERNION`
    ///    - dim[0] = 5
    ///    - dim[5] = 4
    ///    - datatype should be a floating point type
    Quaternion = 1010,
    /// Dimensionless value - no params - although, as in `_ESTIMATE`
    /// the name of the parameter may be stored in intent_name.
    Dimless = 1011,
    /* --- these values apply to GIFTI datasets --- */
    /// To signify that the value at each location is from a time series.
    TimeSeries = 2001,
    /// To signify that the value at each location is a node index, from
    /// a complete surface dataset.
    NodeIndex = 2002,
    /// To signify that the vector value at each location is an RGB triplet,
    /// of whatever type.
    ///    - dataset must have a 5th dimension
    ///    - dim[0] = 5
    ///    - dim[1] = number of nodes
    ///    - dim[2] = dim[3] = dim[4] = 1
    ///    - dim[5] = 3
    RgbVector = 2003,
    /// To signify that the vector value at each location is a 4 valued RGBA
    /// vector, of whatever type.
    ///   - dataset must have a 5th dimension
    ///   - dim[0] = 5
    ///   - dim[1] = number of nodes
    ///   - dim[2] = dim[3] = dim[4] = 1
    ///   - dim[5] = 4
    RgbaVector = 2004,
    /// To signify that the value at each location is a shape value, such
    /// as the curvature.
    Shape = 2005,
}

impl Intent {
    /// Check whether this intent code are used for statistics.
    pub fn is_statcode(&self) -> bool {
        *self as i16 >= 2 && *self as i16 <= 24
    }
}

/// An enum type for representing a NIFTI XForm.
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

/// An enum type for representing the slice order.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, FromPrimitive)]
pub enum SliceOrder {
    /// NIFTI_SLICE_UNKNOWN
    Unknown = 0,
    /// NIFTI_SLICE_SEQ_INC
    SeqInc = 1,
    /// NIFTI_SLICE_SEQ_DEC
    SeqDec = 2,
    /// NIFTI_SLICE_ALT_INC
    AltInc = 3,
    /// NIFTI_SLICE_ALT_DEC
    AltDec = 4,
    /// NIFTI_SLICE_ALT_INC2
    AltInc2 = 5,
    /// NIFTI_SLICE_ALT_DEC2
    AltDec2 = 6,
}
