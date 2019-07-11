//! This module defines the `NiftiHeader` struct, which is used
//! to provide important information about NIFTI-1 volumes.

#[cfg(feature = "nalgebra_affine")]
use crate::affine::*;
#[cfg(feature = "nalgebra_affine")]
use alga::general::SubsetOf;
use byteordered::{ByteOrdered, Endian, Endianness};
use crate::error::{NiftiError, Result};
use crate::typedef::*;
use crate::util::{is_gz_file, validate_dim, validate_dimensionality};
use flate2::bufread::GzDecoder;
#[cfg(feature = "nalgebra_affine")]
use nalgebra::{Matrix3, Matrix4, Quaternion, RealField, Vector3};
use num_traits::FromPrimitive;
#[cfg(feature = "nalgebra_affine")]
use num_traits::ToPrimitive;
use std::fs::File;
use std::io::{BufReader, Read};
use std::ops::Deref;
use std::path::Path;

/// Magic code for NIFTI-1 header files (extention ".hdr[.gz]").
pub const MAGIC_CODE_NI1: &'static [u8; 4] = b"ni1\0";
/// Magic code for full NIFTI-1 files (extention ".nii[.gz]").
pub const MAGIC_CODE_NIP1: &'static [u8; 4] = b"n+1\0";

/// The NIFTI-1 header data type.
/// All fields are public and named after the specification's header file.
/// The type of each field was adjusted according to their use and
/// array limitations.
///
/// # Examples
///
/// ```no_run
/// use nifti::{NiftiHeader, Endianness};
/// # use nifti::Result;
///
/// # fn run() -> Result<()> {
/// let hdr1 = NiftiHeader::from_file("0000.hdr")?;
/// let hdr2 = NiftiHeader::from_file("0001.hdr.gz")?;
/// let hdr3 = NiftiHeader::from_file("4321.nii.gz")?;
/// # Ok(())
/// # }
/// ```
///
/// Or to build one yourself:
///
/// ```
/// # use nifti::{NiftiHeader, NiftiType};
/// let mut hdr = NiftiHeader::default();
/// hdr.cal_min = 0.;
/// hdr.cal_max = 128.;
/// hdr.datatype = 4;
/// assert_eq!(hdr.cal_min, 0.);
/// assert_eq!(hdr.cal_max, 128.);
/// assert_eq!(hdr.data_type().unwrap(), NiftiType::Int16);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct NiftiHeader {
    /// Header size, must be 348
    pub sizeof_hdr: i32,
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
    pub bitpix: i16,
    /// First slice index
    pub slice_start: i16,
    /// Grid spacings
    pub pixdim: [f32; 8],
    /// Offset into .nii file to reach the volume
    pub vox_offset: f32,
    /// Data scaling: slope
    pub scl_slope: f32,
    /// Data scaling: offset
    pub scl_inter: f32,
    /// Last slice index
    pub slice_end: i16,
    /// Slice timing order
    pub slice_code: u8,
    /// Units of pixdim[1..4]
    pub xyzt_units: u8,
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
    pub descrip: Vec<u8>,
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

    /// Magic code. Must be `b"ni1\0"` or `b"ni+\0"`
    pub magic: [u8; 4],

    /// Original data Endianness
    pub endianness: Endianness,
}

impl Default for NiftiHeader {
    fn default() -> NiftiHeader {
        NiftiHeader {
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

            descrip: vec![0; 80],
            aux_file: [0; 24],
            qform_code: 0,
            sform_code: 0,
            quatern_b: 0.,
            quatern_c: 0.,
            quatern_d: 0.,
            quatern_x: 0.,
            quatern_y: 0.,
            quatern_z: 0.,

            srow_x: [1., 0., 0., 0.,],
            srow_y: [0., 1., 0., 0.,],
            srow_z: [0., 0., 1., 0.,],

            intent_name: [0; 16],

            magic: *MAGIC_CODE_NI1,

            endianness: Endianness::native(),
        }
    }
}

impl NiftiHeader {
    /// Retrieve a NIFTI header, along with its byte order, from a file in the file system.
    /// If the file's name ends with ".gz", the file is assumed to need GZip decoding.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<NiftiHeader> {
        let gz = is_gz_file(&path);
        let file = BufReader::new(File::open(path)?);
        if gz {
            NiftiHeader::from_reader(GzDecoder::new(file))
        } else {
            NiftiHeader::from_reader(file)
        }
    }

    /// Read a NIfTI-1 header, along with its byte order, from the given byte stream.
    /// It is assumed that the input is currently at the start of the
    /// NIFTI header.
    #[deprecated(since = "0.8.0", note = "use `from_reader` instead")]
    pub fn from_stream<S: Read>(input: S) -> Result<NiftiHeader> {
        Self::from_reader(input)
    }

    /// Read a NIfTI-1 header, along with its byte order, from the given byte stream.
    /// It is assumed that the input is currently at the start of the
    /// NIFTI header.
    pub fn from_reader<S>(input: S) -> Result<NiftiHeader>
    where
        S: Read,
    {
        parse_header_1(input)
    }

    /// Retrieve and validate the dimensions of the volume. Unlike how NIfTI-1
    /// stores dimensions, the returned slice does not include `dim[0]` and is
    /// clipped to the effective number of dimensions.
    /// 
    /// # Error
    /// 
    /// `NiftiError::InconsistentDim` if `dim[0]` does not represent a valid
    /// dimensionality, or any of the real dimensions are zero.
    pub fn dim(&self) -> Result<&[u16]> {
        validate_dim(&self.dim)
    }

    /// Retrieve and validate the number of dimensions of the volume. This is
    /// `dim[0]` after the necessary byte order conversions.
    /// 
    /// # Error
    /// 
    /// `NiftiError::` if `dim[0]` does not represent a valid dimensionality
    /// (it must be positive and not higher than 7).
    pub fn dimensionality(&self) -> Result<usize> {
        validate_dimensionality(&self.dim)
    }

    /// Get the data type as a validated enum.
    pub fn data_type(&self) -> Result<NiftiType> {
        FromPrimitive::from_i16(self.datatype)
            .ok_or_else(|| NiftiError::InvalidCode("datatype", self.datatype))
    }

    /// Get the spatial units type as a validated unit enum.
    pub fn xyzt_to_space(&self) -> Result<Unit> {
        let space_code = self.xyzt_units & 0o0007;
        FromPrimitive::from_u8(space_code)
            .ok_or_else(|| NiftiError::InvalidCode("xyzt units (space)", space_code as i16))
    }

    /// Get the time units type as a validated unit enum.
    pub fn xyzt_to_time(&self) -> Result<Unit> {
        let time_code = self.xyzt_units & 0o0070;
        FromPrimitive::from_u8(time_code)
            .ok_or_else(|| NiftiError::InvalidCode("xyzt units (time)", time_code as i16))
    }

    /// Get the xyzt units type as a validated pair of space and time unit enum.
    pub fn xyzt_units(&self) -> Result<(Unit, Unit)> {
        Ok((self.xyzt_to_space()?, self.xyzt_to_time()?))
    }

    /// Get the slice order as a validated enum.
    pub fn slice_order(&self) -> Result<SliceOrder> {
        FromPrimitive::from_u8(self.slice_code)
            .ok_or_else(|| NiftiError::InvalidCode("slice order", self.slice_code as i16))
    }

    /// Get the intent as a validated enum.
    pub fn intent(&self) -> Result<Intent> {
        FromPrimitive::from_i16(self.intent_code)
            .ok_or_else(|| NiftiError::InvalidCode("intent", self.intent_code))
    }

    /// Get the qform coordinate mapping method as a validated enum.
    pub fn qform(&self) -> Result<XForm> {
        FromPrimitive::from_i16(self.qform_code)
            .ok_or_else(|| NiftiError::InvalidCode("qform", self.qform_code as i16))
    }

    /// Get the sform coordinate mapping method as a validated enum.
    pub fn sform(&self) -> Result<XForm> {
        FromPrimitive::from_i16(self.sform_code)
            .ok_or_else(|| NiftiError::InvalidCode("sform", self.sform_code as i16))
    }

    /// Ensure that the current `descrip` field is valid and is exactly equal to 80 bytes.
    ///
    /// Descriptions shorter than 80 bytes will be extended with trailing zeros.
    pub fn validate_description(&mut self) -> Result<()> {
        let len = self.descrip.len();
        if len > 80 {
            Err(NiftiError::IncorrectDescriptionLength(len))
        } else {
            if len < 80 {
                self.descrip.extend((len..80).map(|_| 0));
            }
            Ok(())
        }
    }

    /// Safely set the `descrip` field using a buffer.
    pub fn set_description<D>(&mut self, description: D) -> Result<()>
    where
        D: Into<Vec<u8>>,
        D: Deref<Target = [u8]>,
    {
        let len = description.len();
        if len < 80 {
            let mut descrip = vec![0; 80];
            descrip[..len].copy_from_slice(&description);
            self.descrip = descrip;
            Ok(())
        } else if len == 80 {
            self.descrip = description.into();
            Ok(())
        } else {
            Err(NiftiError::IncorrectDescriptionLength(len))
        }
    }

    /// Safely set the `descrip` field using a  &str.
    pub fn set_description_str<T>(&mut self, description: T) -> Result<()>
    where
        T: Into<String>,
    {
        self.set_description(description.into().as_bytes())
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
    fn sform_affine<T>(&self) -> Matrix4<T>
    where
        T: RealField,
        f32: SubsetOf<T>,
    {
        let affine = Matrix4::new(
            self.srow_x[0], self.srow_x[1], self.srow_x[2], self.srow_x[3],
            self.srow_y[0], self.srow_y[1], self.srow_y[2], self.srow_y[3],
            self.srow_z[0], self.srow_z[1], self.srow_z[2], self.srow_z[3],
            0.0, 0.0, 0.0, 1.0,
        );
        nalgebra::convert(affine)
    }

    /// Retrieve affine transformation from qform-related fields.
    fn qform_affine<T>(&self) -> Matrix4<T>
    where
        T: RealField,
    {
        if self.pixdim[1] < 0.0 || self.pixdim[2] < 0.0 || self.pixdim[3] < 0.0 {
            panic!("All spacings (pixdim) should be positive");
        }
        if self.pixdim[0].abs() != 1.0 {
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
    h.bitpix = input.read_i16()?;
    h.slice_start = input.read_i16()?;
    for v in &mut h.pixdim {
        *v = input.read_f32()?;
    }
    h.vox_offset = input.read_f32()?;
    h.scl_slope = input.read_f32()?;
    h.scl_inter = input.read_f32()?;
    h.slice_end = input.read_i16()?;
    h.slice_code = input.read_u8()?;
    h.xyzt_units = input.read_u8()?;
    h.cal_max = input.read_f32()?;
    h.cal_min = input.read_f32()?;
    h.slice_duration = input.read_f32()?;
    h.toffset = input.read_f32()?;
    h.glmax = input.read_i32()?;
    h.glmin = input.read_i32()?;

    // descrip is 80-elem vec already
    input.read_exact(h.descrip.as_mut_slice())?;
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
