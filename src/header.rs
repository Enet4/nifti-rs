//! This module defines the `NiftiHeader` struct, which is used
//! to provide important information about NIFTI volumes.

#[cfg(feature = "nalgebra_affine")]
use crate::affine::*;
use crate::error::{NiftiError, Result};
use crate::typedef::*;
use crate::util::{is_gz_file, validate_dim, validate_dimensionality};
use byteordered::{ByteOrdered, Endian, Endianness};
use flate2::bufread::GzDecoder;
#[cfg(feature = "nalgebra_affine")]
use nalgebra::{Matrix3, Matrix4, Quaternion, RealField, Vector3};
use num_traits::FromPrimitive;
#[cfg(feature = "nalgebra_affine")]
use num_traits::ToPrimitive;
#[cfg(feature = "nalgebra_affine")]
use simba::scalar::SubsetOf;
use std::convert::TryInto;
use std::fs::File;
use std::io::{BufReader, Read};
use std::ops::Deref;
use std::path::Path;

/// Magic code for NIFTI-1 header files (extention ".hdr[.gz]").
pub const MAGIC_CODE_NI1: &[u8; 4] = b"ni1\0";
/// Magic code for full NIFTI-1 files (extention ".nii[.gz]").
pub const MAGIC_CODE_NIP1: &[u8; 4] = b"n+1\0";
/// Magic code for NIFTI-2 header files (extention ".hdr[.gz]").
pub const MAGIC_CODE_NI2: &[u8; 8] = b"ni2\0\r\n\x1A\n";
/// Magic code for full NIFTI-1 files (extention ".nii[.gz]").
pub const MAGIC_CODE_NIP2: &[u8; 8] = b"n+2\0\r\n\x1A\n";

/// Abstraction over NIFTI version 1 and 2 headers.  This is the high-level type
/// you will likely use most often.
/// 
/// The NIFTI file format has two different versions for headers.  NIFTI-1 was
/// finalized in 2007 and is still widely used by many scanner manufacturers.
/// NIFTI-2 was created in 2011 and is widely used in many research-oriented
/// neuroimaging data sets.  NIFTI-2 is almost identical to NIFTI-1 except for
/// using larger data types (e.g. 64-bit floating point types instead of 32-bit)
/// and removal of some unused fields.  Refer to the
/// [reference documentation](https://nifti.nimh.nih.gov/pub/dist/doc/nifti2.h)
/// for a comprehensive list of changes.
/// 
/// For practical purposes, there is no difference between the header versions.
/// This `NiftiHeader` type is an `enum` storing either a `Nifti2Header` or
/// `Nifti1Header`.  Calling `from_file()` or `from_reader()` automatically
/// selects the correct variant depending on the version in the input file.
/// Getter and setter methods like `get_slice_duration()` and
/// `set_slice_duration()`, and high-level methods like `data_type()`,
/// automatically convert values to/from the appropriate underlying type
/// depending on the header version.
/// 
/// If you must work with a specific header version, you can easily convert
/// between versions with `into_nifti1()` or `into_nifti2()`.
/// 
/// # Examples
///
/// ```no_run
/// use nifti::{NiftiHeader, NiftiType};
/// # use nifti::Result;
///
/// # fn run() -> Result<()> {
/// // Read a header from a file.
/// let hdr1 = NiftiHeader::from_file("0000.hdr")?;
/// let hdr2 = NiftiHeader::from_file("0001.hdr.gz")?;
/// let hdr3 = NiftiHeader::from_file("4321.nii.gz")?;
/// 
/// // Convert a header into NIFTI-2 (does nothing if already NIFTI-2).
/// // First extracts/converts the `NiftiHeader` into a `Nifti2Header` and then
/// // puts it back into a `NiftiHeader`.
/// let hdr4 = hdr3.into_nifti2().into_nifti();
/// 
/// // Make a new header from scratch.  Defaults to NIFTI-2.
/// let mut hdr5 = NiftiHeader::default();
/// 
/// // Change the slice duration to two-and-a-half seconds.
/// hdr5.set_slice_duration(2.5);
/// assert_eq!(hdr1.get_slice_duration(), 2.5);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum NiftiHeader {
    /// The underlying header version is `NIFTI-1`.
    Nifti1Header(Nifti1Header),
    /// The underlying header version is `NIFTI-2`.
    Nifti2Header(Nifti2Header),
}
impl Default for NiftiHeader {
    fn default() -> NiftiHeader {
        Nifti2Header::default().into_nifti() // default to NIfTI-2 format
    }
}
impl NiftiHeader {
    // Getter and setter methods.
    // 
    // Signatures are for the larger types of NIFTI-2 fields.
    // When getting a smaller field from a NIFTI-1 header (e.g. intent_code) the
    // value is promoted to the larger type (e.g. i16 -> i32).
    // When setting a smaller integer field in a NIFTI-1 header, the setter
    // returns std::num::TryFromIntError if the assigned value won't fit in the
    // smaller field.
    // When setting a smaller floating point field in a NIFTI-1 header
    // (e.g. quatern_x f64 -> f32), the setter simply performs a saturating
    // cast, which may result in some loss of precision.

    /// Header size, must be 348 for a NIFTI-1 header and 540 for NIFTI-2.
    /// There is no corresponding `set_sizeof_hdr()` because you should never
    /// need to set this field manually.
    pub fn get_sizeof_hdr(&self) -> u32 {
        match *self {
            Self::Nifti1Header(ref header) => header.sizeof_hdr,
            Self::Nifti2Header(ref header) => header.sizeof_hdr,
        }
    }
    /// Get MRI slice ordering.
    pub fn get_dim_info(&self) -> u8 {
        match *self {
            Self::Nifti1Header(ref header) => header.dim_info,
            Self::Nifti2Header(ref header) => header.dim_info,
        }
    }
    /// Set MRI slice ordering.
    pub fn set_dim_info(&mut self, dim_info: u8) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.dim_info = dim_info;
            },
            Self::Nifti2Header(ref mut header) => {
                header.dim_info = dim_info;
            },
        }
    }
    /// Get data array dimensions.
    pub fn get_dim(&self) -> [u64; 8] {
        match *self {
            Self::Nifti1Header(ref header) => {
                header.dim.map(|x| x as u64)
            },
            Self::Nifti2Header(ref header) => header.dim,
        }
    }
    /// Set data array dimensions.  When the header version is NIFTI-2 returns
    /// [`NiftiError::FieldSize`] if any of the dimensions will not fit into i64
    /// as required by the format specification.  For NIFTI-1 requires
    /// dimensions to fit into `i16`.
    pub fn set_dim(&mut self, dim: &[u64; 8]) -> Result<()> {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                for (&src, dst) in dim.iter().zip(&mut header.dim) {
                    *dst = TryInto::<i16>::try_into(src)? as u16;
                }
            },
            Self::Nifti2Header(ref mut header) => {
                for (&src, dst) in dim.iter().zip(&mut header.dim) {
                    *dst = TryInto::<i64>::try_into(src)? as u64;
                }
            },
        }
        Ok(())
    }
    /// Get 1st intent parameter.
    pub fn get_intent_p1(&self) -> f64 {
        match *self {
            Self::Nifti1Header(ref header) => header.intent_p1 as f64,
            Self::Nifti2Header(ref header) => header.intent_p1,
        }
    }
    /// Set 1st intent parameter.  When the header version is NIFTI-1, precision
    /// is reduced to `f32`.
    pub fn set_intent_p1(&mut self, intent_p1: f64) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.intent_p1 = intent_p1 as f32;
            },
            Self::Nifti2Header(ref mut header) => {
                header.intent_p1 = intent_p1;
            },
        }
    }
    /// Get 2nd intent parameter.
    pub fn get_intent_p2(&self) -> f64 {
        match *self {
            Self::Nifti1Header(ref header) => header.intent_p2 as f64,
            Self::Nifti2Header(ref header) => header.intent_p2,
        }
    }
    /// Set 2nd intent parameter.  When the header version is NIFTI-1, precision
    /// is reduced to `f32`.
    pub fn set_intent_p2(&mut self, intent_p2: f64) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.intent_p2 = intent_p2 as f32;
            },
            Self::Nifti2Header(ref mut header) => {
                header.intent_p2 = intent_p2;
            },
        }
    }
    /// Get 3rd intent parameter.
    pub fn get_intent_p3(&self) -> f64 {
        match *self {
            Self::Nifti1Header(ref header) => header.intent_p3 as f64,
            Self::Nifti2Header(ref header) => header.intent_p3,
        }
    }
    /// Set 3rd intent parameter.  When the header version is NIFTI-1, precision
    /// is reduced to `f32`.
    pub fn set_intent_p3(&mut self, intent_p3: f64) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.intent_p3 = intent_p3 as f32;
            },
            Self::Nifti2Header(ref mut header) => {
                header.intent_p3 = intent_p3;
            },
        }
    }
    /// Get NIFTI_INTENT_* code.
    pub fn get_intent_code(&self) -> i32 {
        match *self {
            Self::Nifti1Header(ref header) => header.intent_code as i32,
            Self::Nifti2Header(ref header) => header.intent_code,
        }
    }
    /// Set NIFTI_INTENT_* code.  Always succeeds when the header version is
    /// NIFTI-2.  When the header version is NIFTI-1, returns
    /// [`NiftiError::FieldSize`] if the intent code will not fit into `i16`.
    pub fn set_intent_code(&mut self, intent_code: i32) -> Result<()> {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.intent_code = intent_code.try_into()?;
            },
            Self::Nifti2Header(ref mut header) => {
                header.intent_code = intent_code;
            }
        }
        Ok(())
    }
    /// Get the data type.
    pub fn get_datatype(&self) -> i16 {
        match *self {
            Self::Nifti1Header(ref header) => header.datatype,
            Self::Nifti2Header(ref header) => header.datatype,
        }
    }
    /// Set the data type.
    pub fn set_datatype(&mut self, datatype: i16) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.datatype = datatype;
            },
            Self::Nifti2Header(ref mut header) => {
                header.datatype = datatype;
            },
        }
    }
    /// Get number of bits per voxel.
    pub fn get_bitpix(&self) -> u16 {
        match *self {
            Self::Nifti1Header(ref header) => header.bitpix,
            Self::Nifti2Header(ref header) => header.bitpix,
        }
    }
    /// Set number of bits per voxel.  Returns [`NiftiError::FieldSize`] if
    /// `bitpix` will not fit into `i16`.
    pub fn set_bitpix(&mut self, bitpix: u16) -> Result<()> {
        let _: i16 = bitpix.try_into()?; // Check that bitpix can fit into i16.
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.bitpix = bitpix;
            },
            Self::Nifti2Header(ref mut header) => {
                header.bitpix = bitpix;
            },
        }
        Ok(())
    }
    /// Get first slice index.
    pub fn get_slice_start(&self) -> u64 {
        match *self {
            Self::Nifti1Header(ref header) => header.slice_start as u64,
            Self::Nifti2Header(ref header) => header.slice_start,
        }
    }
    /// Set first slice index.  Returns [`NiftiError::FieldSize`] if
    /// `slice_start` will not fit into `i16` as required by the NIFTI-1 format
    /// or `i64` as required by NIFTI-2.
    pub fn set_slice_start(&mut self, slice_start: u64) -> Result<()> {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.slice_start = TryInto::<i16>::try_into(slice_start)? as u16;
            },
            Self::Nifti2Header(ref mut header) => {
                header.slice_start = TryInto::<i64>::try_into(slice_start)? as u64;
            },
        }
        Ok(())
    }
    /// Get grid spacings.
    pub fn get_pixdim(&self) -> [f64; 8] {
        match *self {
            Self::Nifti1Header(ref header) => header.pixdim.map(|x| x as f64),
            Self::Nifti2Header(ref header) => header.pixdim,
        }
    }
    /// Set grid spacings.
    pub fn set_pixdim(&mut self, pixdim: &[f64; 8]) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.pixdim = pixdim.map(|x| x as f32);
            },
            Self::Nifti2Header(ref mut header) => {
                header.pixdim = *pixdim;
            },
        }
    }
    /// Get offset into .nii file to reach the volume.  Note that NIFTI-1 stores
    /// this offset as `f32`.  Floating point values will be rounded to the
    /// nearest integer.  Returns [`NiftiError::FieldSize`] if that integer
    /// would be negative.
    pub fn get_vox_offset(&self) -> Result<u64> {
        Ok(
            match *self {
                Self::Nifti1Header(ref header) => (header.vox_offset.round() as i64).try_into()?,
                Self::Nifti2Header(ref header) => header.vox_offset,
            }
        )
    }
    /// Set offset into .nii file to reach the volume.  Note that NIFTI-1 stores
    /// this offset as `f32` and NIFTI-2 stores it as `i64`.  Returns
    /// [`NiftiError::FieldSize`] if vox_offset will not fit into the underlying
    /// field type.
    pub fn set_vox_offset(&mut self, vox_offset: u64) -> Result<()> {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.vox_offset = TryInto::<i32>::try_into(vox_offset)? as f32;
            },
            Self::Nifti2Header(ref mut header) => {
                header.vox_offset = TryInto::<i64>::try_into(vox_offset)? as u64;
            },
        }
        Ok(())
    }
    /// Get data scaling slope.
    pub fn get_scl_slope(&self) -> f64 {
        match *self {
            Self::Nifti1Header(ref header) => header.scl_slope as f64,
            Self::Nifti2Header(ref header) => header.scl_slope,
        }
    }
    /// Set data scaling slope.  When the header version is NIFTI-1, precision
    /// is reduced to `f32`.
    pub fn set_scl_slope(&mut self, scl_slope: f64) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.scl_slope = scl_slope as f32;
            },
            Self::Nifti2Header(ref mut header) => {
                header.scl_slope = scl_slope;
            },
        }
    }
    /// Get data scaling offset (intercept).  When the header version is
    /// NIFTI-1, precision is reduced to `f32`.
    pub fn get_scl_inter(&self) -> f64 {
        match *self {
            Self::Nifti1Header(ref header) => header.scl_inter as f64,
            Self::Nifti2Header(ref header) => header.scl_inter,
        }
    }
    /// Set data scaling offset (intercept).
    pub fn set_scl_inter(&mut self, scl_inter: f64) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.scl_inter = scl_inter as f32;
            },
            Self::Nifti2Header(ref mut header) => {
                header.scl_inter = scl_inter;
            },
        }
    }
    /// Get last slice index.
    pub fn get_slice_end(&self) -> u64 {
        match *self {
            Self::Nifti1Header(ref header) => header.slice_end as u64,
            Self::Nifti2Header(ref header) => header.slice_end,
        }
    }
    /// Get last slice index.  Returns [`NiftiError::FieldSize`] if
    /// `slice_end` will not fit into `i16` as required by the NIFTI-1 format
    /// or `i64` as required by NIFTI-2.
    pub fn set_slice_end(&mut self, slice_end: u64) -> Result<()> {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.slice_end = TryInto::<i16>::try_into(slice_end)? as u16;
            },
            Self::Nifti2Header(ref mut header) => {
                header.slice_end = TryInto::<i64>::try_into(slice_end)? as u64;
            },
        }
        Ok(())
    }
    /// Get slice timing order.
    pub fn get_slice_code(&self) -> i32 {
        match *self {
            Self::Nifti1Header(ref header) => header.slice_code as i32,
            Self::Nifti2Header(ref header) => header.slice_code,
        }
    }
    /// Set slice timing order.  When header is NIFTI-1 returns
    /// [`NiftiError::FieldSize`] if `slice_code` will not fit into `i8`.
    pub fn set_slice_code(&mut self, slice_code: i32) -> Result<()> {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.slice_code = slice_code.try_into()?;
            },
            Self::Nifti2Header(ref mut header) => {
                header.slice_code = slice_code;
            },
        }
        Ok(())
    }
    /// Get units of `pixdim[1..4]`.
    pub fn get_xyzt_units(&self) -> i32 {
        match *self {
            Self::Nifti1Header(ref header) => header.xyzt_units as i32,
            Self::Nifti2Header(ref header) => header.xyzt_units,
        }
    }
    /// Set units of `pixdim[1..4]`.  When header is NIFTI-1 returns
    /// [`NiftiError::FieldSize`] if `xyzt_units` will not fit into `i8`.
    pub fn set_xyzt_units(&mut self, xyzt_units: i32) -> Result<()> {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.xyzt_units = xyzt_units.try_into()?;
            },
            Self::Nifti2Header(ref mut header) => {
                header.xyzt_units = xyzt_units;
            },
        }
        Ok(())
    }
    /// Get max display intensity.
    pub fn get_cal_max(&self) -> f64 {
        match *self {
            Self::Nifti1Header(ref header) => header.cal_max as f64,
            Self::Nifti2Header(ref header) => header.cal_max,
        }
    }
    /// Set max display intensity.  When the header version is NIFTI-1,
    /// precision is reduced to `f32`.
    pub fn set_cal_max(&mut self, cal_max: f64) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.cal_max = cal_max as f32;
            },
            Self::Nifti2Header(ref mut header) => {
                header.cal_max = cal_max;
            },
        }
    }
    /// Get min display intensity.
    pub fn get_cal_min(&self) -> f64 {
        match *self {
            Self::Nifti1Header(ref header) => header.cal_min as f64,
            Self::Nifti2Header(ref header) => header.cal_min,
        }
    }
    /// Set min display intensity.  When the header version is NIFTI-1,
    /// precision is reduced to `f32`.
    pub fn set_cal_min(&mut self, cal_min: f64) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.cal_min = cal_min as f32;
            },
            Self::Nifti2Header(ref mut header) => {
                header.cal_min = cal_min;
            },
        }
    }
    /// Get time for 1 slice (in seconds).
    pub fn get_slice_duration(&self) -> f64 {
        match *self {
            Self::Nifti1Header(ref header) => header.slice_duration as f64,
            Self::Nifti2Header(ref header) => header.slice_duration,
        }
    }
    /// Set time for 1 slice (in seconds).  When the header version is NIFTI-1,
    /// precision is reduced to `f32`.
    pub fn set_slice_duration(&mut self, slice_duration: f64) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.slice_duration = slice_duration as f32;
            },
            Self::Nifti2Header(ref mut header) => {
                header.slice_duration = slice_duration;
            },
        }
    }
    /// Get time axis shift.
    pub fn get_toffset(&self) -> f64 {
        match *self {
            Self::Nifti1Header(ref header) => header.toffset as f64,
            Self::Nifti2Header(ref header) => header.toffset,
        }
    }
    /// Set time axis shift.
    pub fn set_toffset(&mut self, toffset: f64) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.toffset = toffset as f32;
            },
            Self::Nifti2Header(ref mut header) => {
                header.toffset = toffset;
            },
        }
    }
    /// Get description.
    pub fn get_descrip(&self) -> &[u8; 80] {
        match *self {
            Self::Nifti1Header(ref header) => &header.descrip,
            Self::Nifti2Header(ref header) => &header.descrip,
        }
    }
    /// Set description (any text you like).
    pub fn set_descrip(&mut self, descrip: [u8; 80]) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.descrip = descrip;
            },
            Self::Nifti2Header(ref mut header) => {
                header.descrip = descrip;
            },
        }
    }
    /// Get auxiliary filename.
    pub fn get_aux_file(&self) -> &[u8; 24] {
        match *self {
            Self::Nifti1Header(ref header) => &header.aux_file,
            Self::Nifti2Header(ref header) => &header.aux_file,
        }
    }
    /// Set auxiliary filename.
    pub fn set_aux_file(&mut self, aux_file: [u8; 24]) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.aux_file = aux_file;
            },
            Self::Nifti2Header(ref mut header) => {
                header.aux_file = aux_file;
            },
        }
    }
    /// Get NIFTI_XFORM_* code.
    pub fn get_qform_code(&self) -> i32 {
        match *self {
            Self::Nifti1Header(ref header) => header.qform_code as i32,
            Self::Nifti2Header(ref header) => header.qform_code,
        }
    }
    /// Set NIFTI_XFORM_* code.  When header is NIFTI-1 returns
    /// [`NiftiError::FieldSize`] if `qform_code` will not fit into `i16`.
    pub fn set_qform_code(&mut self, qform_code: i32) -> Result<()> {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.qform_code = qform_code.try_into()?;
            },
            Self::Nifti2Header(ref mut header) => {
                header.qform_code = qform_code;
            },
        }
        Ok(())
    }
    /// Get NIFTI_XFORM_* code.
    pub fn get_sform_code(&self) -> i32 {
        match *self {
            Self::Nifti1Header(ref header) => header.sform_code as i32,
            Self::Nifti2Header(ref header) => header.sform_code,
        }
    }
    /// Set NIFTI_XFORM_* code.  When header is NIFTI-1 returns
    /// [`NiftiError::FieldSize`] if `sform_code` will not fit into `i16`.
    pub fn set_sform_code(&mut self, sform_code: i32) -> Result<()> {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.sform_code = sform_code.try_into()?;
            },
            Self::Nifti2Header(ref mut header) => {
                header.sform_code = sform_code;
            },
        }
        Ok(())
    }
    /// Get quaternion b param.
    pub fn get_quatern_b(&self) -> f64 {
        match *self {
            Self::Nifti1Header(ref header) => header.quatern_b as f64,
            Self::Nifti2Header(ref header) => header.quatern_b,
        }
    }
    /// Set quaternion b param.  When the header version is NIFTI-1, precision
    /// is reduced to `f32`.
    pub fn set_quatern_b(&mut self, quatern_b: f64) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.quatern_b = quatern_b as f32;
            },
            Self::Nifti2Header(ref mut header) => {
                header.quatern_b = quatern_b;
            },
        }
    }
    /// Get quaternion c param.
    pub fn get_quatern_c(&self) -> f64 {
        match *self {
            Self::Nifti1Header(ref header) => header.quatern_c as f64,
            Self::Nifti2Header(ref header) => header.quatern_c,
        }
    }
    /// Set quaternion c param.  When the header version is NIFTI-1, precision
    /// is reduced to `f32`.
    pub fn set_quatern_c(&mut self, quatern_c: f64) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.quatern_c = quatern_c as f32;
            },
            Self::Nifti2Header(ref mut header) => {
                header.quatern_c = quatern_c;
            },
        }
    }
    /// Get quaternion d param.
    pub fn get_quatern_d(&self) -> f64 {
        match *self {
            Self::Nifti1Header(ref header) => header.quatern_d as f64,
            Self::Nifti2Header(ref header) => header.quatern_d,
        }
    }
    /// Set quaternion d param.  When the header version is NIFTI-1, precision
    /// is reduced to `f32`.
    pub fn set_quatern_d(&mut self, quatern_d: f64) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.quatern_d = quatern_d as f32;
            },
            Self::Nifti2Header(ref mut header) => {
                header.quatern_d = quatern_d;
            },
        }
    }
    /// Get quaternion x param, also known as qoffset_x.
    pub fn get_quatern_x(&self) -> f64 {
        match *self {
            Self::Nifti1Header(ref header) => header.quatern_x as f64,
            Self::Nifti2Header(ref header) => header.quatern_x,
        }
    }
    /// Set quaternion x param.  When the header version is NIFTI-1, precision
    /// is reduced to `f32`.
    pub fn set_quatern_x(&mut self, quatern_x: f64) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.quatern_x = quatern_x as f32;
            },
            Self::Nifti2Header(ref mut header) => {
                header.quatern_x = quatern_x;
            },
        }
    }
    /// Get quaternion y param, also known as qoffset_y.
    pub fn get_quatern_y(&self) -> f64 {
        match *self {
            Self::Nifti1Header(ref header) => header.quatern_y as f64,
            Self::Nifti2Header(ref header) => header.quatern_y,
        }
    }
    /// Set quaternion y param.  When the header version is NIFTI-1, precision
    /// is reduced to `f32`.
    pub fn set_quatern_y(&mut self, quatern_y: f64) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.quatern_y = quatern_y as f32;
            },
            Self::Nifti2Header(ref mut header) => {
                header.quatern_y = quatern_y;
            },
        }
    }
    /// Get quaternion z param, also known as qoffset_z.
    pub fn get_quatern_z(&self) -> f64 {
        match *self {
            Self::Nifti1Header(ref header) => header.quatern_z as f64,
            Self::Nifti2Header(ref header) => header.quatern_z,
        }
    }
    /// Set quaternion z param.  When the header version is NIFTI-1, precision
    /// is reduced to `f32`.
    pub fn set_quatern_z(&mut self, quatern_z: f64) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.quatern_z = quatern_z as f32;
            },
            Self::Nifti2Header(ref mut header) => {
                header.quatern_z = quatern_z;
            },
        }
    }
    /// Get 1st row affine transform.
    pub fn get_srow_x(&self) -> [f64; 4] {
        match *self {
            Self::Nifti1Header(ref header) => header.srow_x.map(|x| x as f64),
            Self::Nifti2Header(ref header) => header.srow_x,
        }
    }
    /// Set 1st row affine transform.  When the header version is NIFTI-1,
    /// precision is reduced to `f32`.
    pub fn set_srow_x(&mut self, srow_x: &[f64; 4]) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                for (&src, dst) in srow_x.iter().zip(&mut header.srow_x) {
                    *dst = src as f32;
                }
            },
            Self::Nifti2Header(ref mut header) => {
                header.srow_x = *srow_x;
            },
        }
    }
    /// Get 2nd row affine transform.
    pub fn get_srow_y(&self) -> [f64; 4] {
        match *self {
            Self::Nifti1Header(ref header) => header.srow_y.map(|x| x as f64),
            Self::Nifti2Header(ref header) => header.srow_y,
        }
    }
    /// Set 2nd row affine transform.  When the header version is NIFTI-1,
    /// precision is reduced to `f32`.
    pub fn set_srow_y(&mut self, srow_y: &[f64; 4]) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                for (&src, dst) in srow_y.iter().zip(&mut header.srow_y) {
                    *dst = src as f32;
                }
            },
            Self::Nifti2Header(ref mut header) => {
                header.srow_y = *srow_y;
            },
        }
    }
    /// Get 3rd row affine transform.
    pub fn get_srow_z(&self) -> [f64; 4] {
        match *self {
            Self::Nifti1Header(ref header) => header.srow_z.map(|x| x as f64),
            Self::Nifti2Header(ref header) => header.srow_z,
        }
    }
    /// Set 3rd row affine transform.  When the header version is NIFTI-1,
    /// precision is reduced to `f32`.
    pub fn set_srow_z(&mut self, srow_z: &[f64; 4]) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                for (&src, dst) in srow_z.iter().zip(&mut header.srow_z) {
                    *dst = src as f32;
                }
            },
            Self::Nifti2Header(ref mut header) => {
                header.srow_z = *srow_z;
            },
        }
    }
    /// Get magic code.  This will be 4 bytes for a NIFTI-1 header and 8 bytes
    /// for a NIFTI-2 header.  There is no corresponding `set_magic()` because
    /// you should never need to set this field manually.
    pub fn get_magic(&self) -> &[u8] {
        match *self {
            Self::Nifti1Header(ref header) => &header.magic,
            Self::Nifti2Header(ref header) => &header.magic,
        }
    }
    /// Get 'name' or meaning of data.
    pub fn get_intent_name(&self) -> &[u8; 16] {
        match *self {
            Self::Nifti1Header(ref header) => &header.intent_name,
            Self::Nifti2Header(ref header) => &header.intent_name,
        }
    }
    /// Set 'name' or meaning of data.
    pub fn set_intent_name(&mut self, intent_name: [u8; 16]) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.intent_name = intent_name;
            },
            Self::Nifti2Header(ref mut header) => {
                header.intent_name = intent_name;
            },
        }
    }
    /// Get original data endianness.
    pub fn get_endianness(&self) -> Endianness {
        match *self {
            Self::Nifti1Header(ref header) => header.endianness,
            Self::Nifti2Header(ref header) => header.endianness,
        }
    }
    /// Set data endianness field in header
    /// (will not change endianness of data itself).
    pub fn set_endianness(&mut self, endianness: Endianness) {
        match *self {
            Self::Nifti1Header(ref mut header) => {
                header.endianness = endianness;
            },
            Self::Nifti2Header(ref mut header) => {
                header.endianness = endianness;
            },
        }
    }

    /// Convert this header into a NIFTI-2 header.
    /// Does nothing if the header is already NIFTI-2.
    /// Promotes NIFTI-1 fields to larger types and removes unused fields.
    /// Increases header size from NIFTI-1's 348 to NIFTI-2's 540 bytes.
    /// Moves vox_offset forward by corresponding 192 bytes.
    /// Changes NIFTI-1 magic string to NIFTI-2 magic string.
    pub fn into_nifti2(self) -> Nifti2Header {
        match self {
            Self::Nifti1Header(header) => {
                Nifti2Header {
                    // If the original header file uses the NIFTI-1 magic string
                    // for .hdr/.img then the new header should use the NIFTI-2
                    // magic string for this filetype, otherwise use the magic
                    // string for .nii.
                    magic: if &header.magic == MAGIC_CODE_NI1 {
                        *MAGIC_CODE_NI2
                    } else {
                        *MAGIC_CODE_NIP2
                    },
                    datatype: header.datatype,
                    bitpix: header.bitpix,
                    dim: header.dim.map(|x| x as u64),
                    intent_p1: header.intent_p1 as f64,
                    intent_p2: header.intent_p2 as f64,
                    intent_p3: header.intent_p3 as f64,
                    pixdim: header.pixdim.map(|x| x as f64),
                    // Voxel offset should be equal to previous voxel offset
                    // plus difference in header size = 540 - 348 = 192.
                    vox_offset: header.vox_offset.round() as u64 + 192,
                    scl_slope: header.scl_slope as f64,
                    scl_inter: header.scl_inter as f64,
                    cal_max: header.cal_max as f64,
                    cal_min: header.cal_min as f64,
                    slice_duration: header.slice_duration as f64,
                    toffset: header.toffset as f64,
                    slice_start: header.slice_start as u64,
                    slice_end: header.slice_end as u64,
                    descrip: header.descrip,
                    aux_file: header.aux_file,
                    qform_code: header.qform_code as i32,
                    sform_code: header.sform_code as i32,
                    quatern_b: header.quatern_b as f64,
                    quatern_c: header.quatern_c as f64,
                    quatern_d: header.quatern_d as f64,
                    quatern_x: header.quatern_x as f64,
                    quatern_y: header.quatern_y as f64,
                    quatern_z: header.quatern_z as f64,
                    srow_x: header.srow_x.map(|x| x as f64),
                    srow_y: header.srow_y.map(|x| x as f64),
                    srow_z: header.srow_z.map(|x| x as f64),
                    slice_code: header.slice_code as i32,
                    xyzt_units: header.xyzt_units as i32,
                    intent_code: header.intent_code as i32,
                    intent_name: header.intent_name,
                    dim_info: header.dim_info,
                    endianness: header.endianness,
                    ..Default::default()
                }
            },
            Self::Nifti2Header(header) => header,
        }
    }

    /// Convert this header into a NIFTI-1 header.
    /// Does nothing if the header is already NIFTI-1.
    /// Shrinks the NIFTI-2 header size from 540 to NIFTI-1's 348, and moves
    /// vox_offset back by a corresponding 192 bytes.
    /// Attempts to downcast NIFTI-2 fields to their smaller NIFTI-1 types.
    /// Performs saturating downcast for floating point fields.
    /// Performs fallible downcast for integer fields, returning
    /// [`NiftiError::FieldSize`] if the field won't fit into the smaller type.
    /// Initializes the unused data_type, db_name, extents, session_error,
    /// regular, glmax, and glmin fields with their default (zero) values.
    pub fn into_nifti1(self) -> Result<Nifti1Header> {
        Ok( match self {
            Self::Nifti1Header(header) => header,
            Self::Nifti2Header(header) => {
                Nifti1Header {
                    dim_info: header.dim_info,
                    dim: { // attempt to map u64 to u16 that fits in an i16
                        let mut dim: [u16; 8] = [0; 8];
                        for (&src, dst) in header.dim.iter().zip(&mut dim) {
                            *dst = TryInto::<i16>::try_into(src)? as u16;
                        }
                        dim
                    },
                    intent_p1: header.intent_p1 as f32,
                    intent_p2: header.intent_p2 as f32,
                    intent_p3: header.intent_p3 as f32,
                    intent_code: header.intent_code.try_into()?,
                    datatype: header.datatype,
                    bitpix: header.bitpix,
                    slice_start: TryInto::<i16>::try_into(header.slice_start)? as u16,
                    pixdim: header.pixdim.map(|x| x as f32),
                    vox_offset: TryInto::<i32>::try_into(header.vox_offset)? as f32,
                    scl_slope: header.scl_slope as f32,
                    scl_inter: header.scl_inter as f32,
                    slice_end: TryInto::<i16>::try_into(header.slice_end)? as u16,
                    slice_code: header.slice_code.try_into()?,
                    xyzt_units: header.xyzt_units.try_into()?,
                    cal_max: header.cal_max as f32,
                    cal_min: header.cal_min as f32,
                    slice_duration: header.slice_duration as f32,
                    toffset: header.toffset as f32,
                    descrip: header.descrip,
                    aux_file: header.aux_file,
                    qform_code: header.qform_code.try_into()?,
                    sform_code: header.sform_code.try_into()?,
                    quatern_b: header.quatern_b as f32,
                    quatern_c: header.quatern_c as f32,
                    quatern_d: header.quatern_d as f32,
                    quatern_x: header.quatern_x as f32,
                    quatern_y: header.quatern_y as f32,
                    quatern_z: header.quatern_z as f32,
                    srow_x: header.srow_x.map(|x| x as f32),
                    srow_y: header.srow_y.map(|x| x as f32),
                    srow_z: header.srow_z.map(|x| x as f32),
                    intent_name: header.intent_name,
                    // If the original header file uses the NIFTI-1 magic string
                    // for .hdr/.img then the new header should use the NIFTI-2
                    // magic string for this filetype, otherwise use the magic
                    // string for .nii.
                    magic: if &header.magic == MAGIC_CODE_NI2 {
                        *MAGIC_CODE_NI1
                    } else {
                        *MAGIC_CODE_NIP1
                    },
                    endianness: header.endianness,
                    ..Default::default()
                }
            },
        } )
    }
/// The NIFTI-1 header data type.
/// 
/// All fields are public and named after the specification's header file.
/// The type of each field was adjusted according to their use and
/// array limitations.
/// 
/// See also `NiftiHeader` for a high-level abstraction over NIFTI-1 and NIFTI-2
/// headers.
/// 
/// # Examples
///
/// ```
/// use nifti::{NiftiHeader, Nifti1Header, NiftiType};
/// let mut hdr = Nifti1Header::default();
/// hdr.cal_min = 0.;
/// hdr.cal_max = 128.;
/// hdr.datatype = 4;
/// assert_eq!(hdr.cal_min, 0.);
/// assert_eq!(hdr.cal_max, 128.);
/// let hdr: NiftiHeader = hdr.into_nifti();
/// assert_eq!(hdr.data_type().unwrap(), NiftiType::Int16);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Nifti1Header {
    /// Header size, must be 348
    pub sizeof_hdr: u32,
    /// Unused in NIFTI-1
    pub data_type: [u8; 10],
    /// Unused in NIFTI-1
    pub db_name: [u8; 18],
    /// Unused in NIFTI-1
    pub extents: i32,
    /// Unused in NIFTI-1
    pub session_error: i16,
    /// Unused in NIFTI-1
    pub regular: u8,
    /// MRI slice ordering
    pub dim_info: u8,
    /// Data array dimensions
    /// Note in the NIFTI-1 specification this is actually [i16; 8].
    pub dim: [u16; 8],
    /// 1st intent parameter
    pub intent_p1: f32,
    /// 2nd intent parameter
    pub intent_p2: f32,
    /// 3rd intent parameter
    pub intent_p3: f32,
    /// NIFTI_INTENT_* code
    pub intent_code: i16,
    /// Defines the data type!
    pub datatype: i16,
    /// Number of bits per voxel
    /// Note in the NIFTI-1 specification this is actually i16.
    pub bitpix: u16,
    /// First slice index
    /// Note in the NIFTI-1 specification this is actually i16.
    pub slice_start: u16,
    /// Grid spacings
    pub pixdim: [f32; 8],
    /// Offset into .nii file to reach the volume
    pub vox_offset: f32,
    /// Data scaling: slope
    pub scl_slope: f32,
    /// Data scaling: offset
    pub scl_inter: f32,
    /// Last slice index
    /// Note in the NIFTI-1 specification this is actually i16.
    pub slice_end: u16,
    /// Slice timing order
    pub slice_code: i8,
    /// Units of `pixdim[1..4]`
    pub xyzt_units: i8,
    /// Max display intensity
    pub cal_max: f32,
    /// Min display intensity
    pub cal_min: f32,
    /// Time for 1 slice
    pub slice_duration: f32,
    /// Time axis shift
    pub toffset: f32,
    /// Unused in NIFTI-1
    pub glmax: i32,
    /// Unused in NIFTI-1
    pub glmin: i32,

    /// Any text you like
    pub descrip: [u8; 80],
    /// Auxiliary filename
    pub aux_file: [u8; 24],
    /// NIFTI_XFORM_* code
    pub qform_code: i16,
    /// NIFTI_XFORM_* code
    pub sform_code: i16,
    /// Quaternion b param
    pub quatern_b: f32,
    /// Quaternion c param
    pub quatern_c: f32,
    /// Quaternion d param
    pub quatern_d: f32,
    /// Quaternion x shift
    pub quatern_x: f32,
    /// Quaternion y shift
    pub quatern_y: f32,
    /// Quaternion z shift
    pub quatern_z: f32,

    /// 1st row affine transform
    pub srow_x: [f32; 4],
    /// 2nd row affine transform
    pub srow_y: [f32; 4],
    /// 3rd row affine transform
    pub srow_z: [f32; 4],

    /// 'name' or meaning of data
    pub intent_name: [u8; 16],

    /// Magic code. Must be `b"ni1\0"` or `b"n+1\0"`
    pub magic: [u8; 4],

    /// Original data Endianness
    pub endianness: Endianness,
}

/// The NIFTI-2 header data type.
/// 
/// All fields are public and named after the specification's header file.
/// The type of each field was adjusted according to their use and
/// array limitations.
/// 
/// See also `NiftiHeader` for a high-level abstraction over NIFTI-1 and NIFTI-2
/// headers.
/// 
/// # Examples
///
/// ```
/// use nifti::{NiftiHeader, Nifti2Header, NiftiType};
/// let mut hdr = Nifti2Header::default();
/// hdr.cal_min = 0.;
/// hdr.cal_max = 128.;
/// hdr.datatype = 4;
/// assert_eq!(hdr.cal_min, 0.);
/// assert_eq!(hdr.cal_max, 128.);
/// let hdr: NiftiHeader = hdr.into_nifti();
/// assert_eq!(hdr.data_type().unwrap(), NiftiType::Int16);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Nifti2Header {
    /// Header size, must be 540
    pub sizeof_hdr: u32,
    /// Magic code.
    /// First 4 bytes must be `b"ni2\0"` or `b"n+2\0"`.
    /// Last 4 bytes must be `b"\r\n\032\n"` (0D 0A 1A 0A).
    pub magic: [u8; 8],
    /// Defines the data type!
    pub datatype: i16,
    /// Number of bits per voxel
    /// Note in the NIFTI-2 specification this is actually i16.
    pub bitpix: u16,
    /// Data array dimensions
    /// Note in the NIFTI-2 specification this is actually i64.
    pub dim: [u64; 8],
    /// 1st intent parameter
    pub intent_p1: f64,
    /// 2nd intent parameter
    pub intent_p2: f64,
    /// 3rd intent parameter
    pub intent_p3: f64,
    /// Grid spacings
    pub pixdim: [f64; 8],
    /// Offset into .nii file to reach the volume
    /// Note in the NIFTI-2 specification this is actually i64.
    pub vox_offset: u64,
    /// Data scaling: slope
    pub scl_slope: f64,
    /// Data scaling: offset
    pub scl_inter: f64,
    /// Max display intensity
    pub cal_max: f64,
    /// Min display intensity
    pub cal_min: f64,
    /// Time for 1 slice
    pub slice_duration: f64,
    /// Time axis shift
    pub toffset: f64,
    /// First slice index
    /// Note in the NIFTI-2 specification this is actually i64.
    pub slice_start: u64,
    /// Last slice index
    /// Note in the NIFTI-2 specification this is actually i64.
    pub slice_end: u64,
    /// Any text you like
    pub descrip: [u8; 80],
    /// Auxiliary filename
    pub aux_file: [u8; 24],
    /// NIFTI_XFORM_* code
    pub qform_code: i32,
    /// NIFTI_XFORM_* code
    pub sform_code: i32,
    /// Quaternion b param
    pub quatern_b: f64,
    /// Quaternion c param
    pub quatern_c: f64,
    /// Quaternion d param
    pub quatern_d: f64,
    /// Quaternion x shift
    pub quatern_x: f64,
    /// Quaternion y shift
    pub quatern_y: f64,
    /// Quaternion z shift
    pub quatern_z: f64,
    /// 1st row affine transform
    pub srow_x: [f64; 4],
    /// 2nd row affine transform
    pub srow_y: [f64; 4],
    /// 3rd row affine transform
    pub srow_z: [f64; 4],
    /// Slice timing order
    pub slice_code: i32,
    /// Units of `pixdim[1..4]`
    pub xyzt_units: i32,
    /// NIFTI_INTENT_* code
    pub intent_code: i32,
    /// 'name' or meaning of data
    pub intent_name: [u8; 16],
    /// MRI slice ordering
    pub dim_info: u8,

    /// Original data Endianness
    pub endianness: Endianness,
}

impl Default for Nifti1Header {
    fn default() -> Nifti1Header {
        Nifti1Header {
            sizeof_hdr: 348,
            data_type: [0; 10],
            db_name: [0; 18],
            extents: 0,
            session_error: 0,
            regular: 0,
            dim_info: 0,
            dim: [1, 0, 0, 0, 0, 0, 0, 0],
            intent_p1: 0.,
            intent_p2: 0.,
            intent_p3: 0.,
            intent_code: 0,
            datatype: 0,
            bitpix: 0,
            slice_start: 0,
            pixdim: [1.; 8],
            vox_offset: 352.,
            scl_slope: 0.,
            scl_inter: 0.,
            slice_end: 0,
            slice_code: 0,
            xyzt_units: 0,
            cal_max: 0.,
            cal_min: 0.,
            slice_duration: 0.,
            toffset: 0.,
            glmax: 0,
            glmin: 0,

            descrip: [0; 80],
            aux_file: [0; 24],
            qform_code: 1,
            sform_code: 1,
            quatern_b: 0.,
            quatern_c: 0.,
            quatern_d: 0.,
            quatern_x: 0.,
            quatern_y: 0.,
            quatern_z: 0.,

            srow_x: [1., 0., 0., 0.],
            srow_y: [0., 1., 0., 0.],
            srow_z: [0., 0., 1., 0.],

            intent_name: [0; 16],

            magic: *MAGIC_CODE_NIP1,

            endianness: Endianness::native(),
        }
    }
}

impl Default for Nifti2Header {
    fn default() -> Nifti2Header {
        Nifti2Header {
            sizeof_hdr: 540,
            magic: *MAGIC_CODE_NIP2,
            datatype: 0,
            bitpix: 0,
            dim: [1, 0, 0, 0, 0, 0, 0, 0],
            intent_p1: 0.,
            intent_p2: 0.,
            intent_p3: 0.,
            pixdim: [1.; 8],
            vox_offset: 544,
            scl_slope: 0.,
            scl_inter: 0.,
            cal_max: 0.,
            cal_min: 0.,
            slice_duration: 0.,
            toffset: 0.,
            slice_start: 0,
            slice_end: 0,
            descrip: [0; 80],
            aux_file: [0; 24],
            qform_code: 1,
            sform_code: 1,
            quatern_b: 0.,
            quatern_c: 0.,
            quatern_d: 0.,
            quatern_x: 0.,
            quatern_y: 0.,
            quatern_z: 0.,
            srow_x: [1., 0., 0., 0.],
            srow_y: [0., 1., 0., 0.],
            srow_z: [0., 0., 1., 0.],
            slice_code: 0,
            xyzt_units: 0,
            intent_code: 0,
            intent_name: [0; 16],
            dim_info: 0,

            endianness: Endianness::native(),
        }
    }
}

impl Nifti1Header {
    /// Place this `Nifti1Header` into a version-agnostic [`NiftiHeader`] enum.
    pub fn into_nifti(self) -> NiftiHeader {
        NiftiHeader::Nifti1Header(self)
    }
}

impl Nifti2Header {
    /// Place this `Nifti2Header` into a version-agnostic [`NiftiHeader`] enum.
    pub fn into_nifti(self) -> NiftiHeader {
        NiftiHeader::Nifti2Header(self)
    }
}

#[cfg(feature = "nalgebra_affine")]
impl NiftiHeader {
    /// Retrieve best of available transformations.
    ///
    /// Return the 'sform' transformation if `sform_code` has a valid value, 'qform' transformation
    /// if `qform_code` has a valid value, otherwise return a "base" transformation, constructed
    /// from the declared shape and zooms.
    ///
    /// If `sform_code` and `qform_code` both have valid values, the 'sform' affine transformation
    /// is prioritized.
    pub fn affine<T>(&self) -> Matrix4<T>
    where
        T: RealField,
        f32: SubsetOf<T>,
    {
        if self.sform_code != 0 {
            self.sform_affine::<T>()
        } else if self.qform_code != 0 {
            self.qform_affine::<T>()
        } else {
            self.base_affine::<T>()
        }
    }

    /// Retrieve affine transformation from 'sform' fields.
    pub fn sform_affine<T>(&self) -> Matrix4<T>
    where
        T: RealField,
        f32: SubsetOf<T>,
    {
        #[rustfmt::skip]
        let affine = Matrix4::new(
            self.srow_x[0], self.srow_x[1], self.srow_x[2], self.srow_x[3],
            self.srow_y[0], self.srow_y[1], self.srow_y[2], self.srow_y[3],
            self.srow_z[0], self.srow_z[1], self.srow_z[2], self.srow_z[3],
            0.0, 0.0, 0.0, 1.0,
        );
        nalgebra::convert(affine)
    }

    /// Retrieve affine transformation from qform-related fields.
    pub fn qform_affine<T>(&self) -> Matrix4<T>
    where
        T: RealField,
    {
        if self.pixdim[1] < 0.0 || self.pixdim[2] < 0.0 || self.pixdim[3] < 0.0 {
            panic!("All spacings (pixdim) should be positive");
        }
        if !self.is_pixdim_0_valid() {
            panic!("qfac (pixdim[0]) should be 1 or -1");
        }

        let quaternion = self.qform_quaternion();
        let r = quaternion_to_affine(quaternion);
        let s = Matrix3::from_diagonal(&Vector3::new(
            self.pixdim[1] as f64,
            self.pixdim[2] as f64,
            self.pixdim[3] as f64 * self.pixdim[0] as f64,
        ));
        let m = r * s;
        #[rustfmt::skip]
        let affine = Matrix4::new(
            m[0], m[3], m[6], self.quatern_x as f64,
            m[1], m[4], m[7], self.quatern_y as f64,
            m[2], m[5], m[8], self.quatern_z as f64,
            0.0, 0.0, 0.0, 1.0,
        );
        nalgebra::convert(affine)
    }

    /// Retrieve affine transformation implied by shape and zooms.
    ///
    /// Note that we get the translations from the center of the image.
    fn base_affine<T>(&self) -> Matrix4<T>
    where
        T: RealField,
    {
        let d = self.dim[0] as usize;
        let affine = shape_zoom_affine(&self.dim[1..d + 1], &self.pixdim[1..d + 1]);
        nalgebra::convert(affine)
    }

    /// Compute quaternion from b, c, d of quaternion.
    ///
    /// Fills a value by assuming this is a unit quaternion.
    fn qform_quaternion(&self) -> Quaternion<f64> {
        let xyz = Vector3::new(
            self.quatern_b as f64,
            self.quatern_c as f64,
            self.quatern_d as f64,
        );
        fill_positive(xyz)
    }

    /// Set affine transformation.
    ///
    /// Will set both affine transformations to avoid interoperability problems:
    ///
    /// * 'sform' from unmodified `affine`, with `sform_code` set to `AlignedAnat`.
    /// * 'qform' from a quaternion built from `affine`. However, the 'qform' won't be used by most
    /// nifti readers because the `qform_code` will be set to `Unknown`.
    pub fn set_affine<T>(&mut self, affine: &Matrix4<T>)
    where
        T: RealField,
        T: SubsetOf<f64>,
        T: ToPrimitive,
    {
        // Set affine into sform with default code.
        self.set_sform(affine, XForm::AlignedAnat);

        // Make qform 'unknown'.
        self.set_qform(affine, XForm::Unknown);
    }

    /// Set affine transformation in 'sform' fields.
    fn set_sform<T>(&mut self, affine: &Matrix4<T>, code: XForm)
    where
        T: RealField,
        T: ToPrimitive,
    {
        self.sform_code = code as i16;
        self.srow_x[0] = affine[0].to_f32().unwrap();
        self.srow_x[1] = affine[4].to_f32().unwrap();
        self.srow_x[2] = affine[8].to_f32().unwrap();
        self.srow_x[3] = affine[12].to_f32().unwrap();
        self.srow_y[0] = affine[1].to_f32().unwrap();
        self.srow_y[1] = affine[5].to_f32().unwrap();
        self.srow_y[2] = affine[9].to_f32().unwrap();
        self.srow_y[3] = affine[13].to_f32().unwrap();
        self.srow_z[0] = affine[2].to_f32().unwrap();
        self.srow_z[1] = affine[6].to_f32().unwrap();
        self.srow_z[2] = affine[10].to_f32().unwrap();
        self.srow_z[3] = affine[14].to_f32().unwrap();
    }

    /// Set affine transformation in qform-related fields.
    ///
    /// The qform transform only encodes translations, rotations and zooms. If there are shear
    /// components to the `affine` transform, the written qform gives the closest approximation
    /// where the rotation matrix is orthogonal. This is to allow quaternion representation. The
    /// orthogonal representation enforces orthogonal axes.
    fn set_qform<T>(&mut self, affine4: &Matrix4<T>, code: XForm)
    where
        T: RealField,
        T: SubsetOf<f64>,
        T: ToPrimitive,
    {
        let affine4: Matrix4<f64> = nalgebra::convert(*affine4);
        let (affine, translation) = affine_and_translation(&affine4);
        let aff2 = affine.component_mul(&affine);
        let spacing = (
            (aff2[0] + aff2[1] + aff2[2]).sqrt(),
            (aff2[3] + aff2[4] + aff2[5]).sqrt(),
            (aff2[6] + aff2[7] + aff2[8]).sqrt(),
        );
        #[rustfmt::skip]
        let mut r = Matrix3::new(
            affine[0] / spacing.0, affine[3] / spacing.1, affine[6] / spacing.2,
            affine[1] / spacing.0, affine[4] / spacing.1, affine[7] / spacing.2,
            affine[2] / spacing.0, affine[5] / spacing.1, affine[8] / spacing.2,
        );

        // Set qfac to make R determinant positive
        let qfac = if r.determinant() > 0.0 {
            1.0
        } else {
            r[6] *= -1.0;
            r[7] *= -1.0;
            r[8] *= -1.0;
            -1.0
        };

        // Make R orthogonal (to allow quaternion representation). The orthogonal representation
        // enforces orthogonal axes (a subtle requirement of the NIFTI format qform transform).
        // Transform below is polar decomposition, returning the closest orthogonal matrix PR, to
        // input R.
        let svd = r.svd(true, true);
        let pr = svd.u.unwrap() * svd.v_t.unwrap();
        let quaternion = affine_to_quaternion(&pr);

        self.qform_code = code as i16;
        self.pixdim[0] = qfac;
        self.pixdim[1] = spacing.0 as f32;
        self.pixdim[2] = spacing.1 as f32;
        self.pixdim[3] = spacing.2 as f32;
        self.quatern_b = quaternion[1] as f32;
        self.quatern_c = quaternion[2] as f32;
        self.quatern_d = quaternion[3] as f32;
        self.quatern_x = translation[0] as f32;
        self.quatern_y = translation[1] as f32;
        self.quatern_z = translation[2] as f32;
    }
}

fn parse_header_1<S>(input: S) -> Result<NiftiHeader>
where
    S: Read,
{
    let mut h = NiftiHeader::default();

    // try the system's native endianness first
    let mut input = ByteOrdered::native(input);

    h.sizeof_hdr = input.read_i32()?;
    input.read_exact(&mut h.data_type)?;
    input.read_exact(&mut h.db_name)?;
    h.extents = input.read_i32()?;
    h.session_error = input.read_i16()?;
    h.regular = input.read_u8()?;
    h.dim_info = input.read_u8()?;
    h.dim[0] = input.read_u16()?;

    if h.dim[0] > 7 {
        h.endianness = Endianness::native().to_opposite();

        // swap bytes read so far, continue with the opposite endianness
        h.sizeof_hdr = h.sizeof_hdr.swap_bytes();
        h.extents = h.extents.swap_bytes();
        h.session_error = h.session_error.swap_bytes();
        h.dim[0] = h.dim[0].swap_bytes();
        parse_header_2(h, input.into_opposite())
    } else {
        // all is well
        h.endianness = Endianness::native();
        parse_header_2(h, input)
    }
}

/// second part of header parsing
fn parse_header_2<S, E>(mut h: NiftiHeader, mut input: ByteOrdered<S, E>) -> Result<NiftiHeader>
where
    S: Read,
    E: Endian,
{
    for v in &mut h.dim[1..] {
        *v = input.read_u16()?;
    }
    h.intent_p1 = input.read_f32()?;
    h.intent_p2 = input.read_f32()?;
    h.intent_p3 = input.read_f32()?;
    h.intent_code = input.read_i16()?;
    h.datatype = input.read_i16()?;
    h.bitpix = input.read_i16()? as u16;
    h.slice_start = input.read_i16()? as u16;
    for v in &mut h.pixdim {
        *v = input.read_f32()?;
    }
    h.vox_offset = input.read_f32()?;
    h.scl_slope = input.read_f32()?;
    h.scl_inter = input.read_f32()?;
    h.slice_end = input.read_i16()? as u16;
    h.slice_code = input.read_i8()?;
    h.xyzt_units = input.read_i8()?;
    h.cal_max = input.read_f32()?;
    h.cal_min = input.read_f32()?;
    h.slice_duration = input.read_f32()?;
    h.toffset = input.read_f32()?;
    h.glmax = input.read_i32()?;
    h.glmin = input.read_i32()?;

    // descrip is 80-elem vec already
    input.read_exact(&mut h.descrip)?;
    input.read_exact(&mut h.aux_file)?;
    h.qform_code = input.read_i16()?;
    h.sform_code = input.read_i16()?;
    h.quatern_b = input.read_f32()?;
    h.quatern_c = input.read_f32()?;
    h.quatern_d = input.read_f32()?;
    h.quatern_x = input.read_f32()?;
    h.quatern_y = input.read_f32()?;
    h.quatern_z = input.read_f32()?;
    for v in &mut h.srow_x {
        *v = input.read_f32()?;
    }
    for v in &mut h.srow_y {
        *v = input.read_f32()?;
    }
    for v in &mut h.srow_z {
        *v = input.read_f32()?;
    }
    input.read_exact(&mut h.intent_name)?;
    input.read_exact(&mut h.magic)?;

    debug_assert_eq!(h.descrip.len(), 80);

    if &h.magic != MAGIC_CODE_NI1 && &h.magic != MAGIC_CODE_NIP1 {
        Err(NiftiError::InvalidFormat)
    } else {
        Ok(h)
    }
}
