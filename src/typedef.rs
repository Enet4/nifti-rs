#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
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

