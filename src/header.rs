//! This module defines the `NiftiHeader` struct, which is used
//! to provide important information about NIFTI-1 volumes.

use byteorder::{ByteOrder, NativeEndian, ReadBytesExt};
use error::{NiftiError, Result};
use flate2::bufread::GzDecoder;
use num_traits::FromPrimitive;
use std::fs::File;
use std::io::{BufReader, Read};
use std::ops::Deref;
use std::path::Path;
use typedef::*;
use util::{is_gz_file, Endianness, OppositeNativeEndian};

/// Magic code for NIFTI-1 header files (extention ".hdr[.gz]").
pub const MAGIC_CODE_NI1: &'static [u8; 4] = b"ni1\0";
/// Magic code for full NIFTI-1 files (extention ".nii[.gz]").
pub const MAGIC_CODE_NIP1: &'static [u8; 4] = b"n+1\0";

/// The NIFTI-1 header data type.
/// All fields are public and named after the specification's header file.
/// The type of each field was adjusted according to their use and
/// array limitations. A builder is also available.
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
/// use nifti::NiftiHeaderBuilder;
/// # use std::error::Error;
///
/// # fn run() -> Result<(), Box<Error>> {
/// let hdr = NiftiHeaderBuilder::default()
///     .cal_min(0.)
///     .cal_max(128.)
///     .build()?;
/// assert_eq!(hdr.cal_min, 0.);
/// assert_eq!(hdr.cal_max, 128.);
/// # Ok(())
/// # }
/// # run().unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Builder)]
#[builder(derive(Debug))]
#[builder(field(public))]
#[builder(default)]
pub struct NiftiHeader {
    /// Header size, must be 348
    #[builder(default = "348")]
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
    #[builder(default = "Endianness::system()")]
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
            pixdim: [0.; 8],
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

            srow_x: [0.; 4],
            srow_y: [0.; 4],
            srow_z: [0.; 4],

            intent_name: [0; 16],

            magic: *MAGIC_CODE_NI1,

            endianness: Endianness::LE,
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
            NiftiHeader::from_stream(GzDecoder::new(file))
        } else {
            NiftiHeader::from_stream(file)
        }
    }

    /// Read a NIfTI-1 header, along with its byte order, from the given byte stream.
    /// It is assumed that the input is currently at the start of the
    /// NIFTI header.
    pub fn from_stream<S: Read>(input: S) -> Result<NiftiHeader> {
        parse_header_1(input)
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

fn parse_header_1<S: Read>(mut input: S) -> Result<NiftiHeader> {
    let mut h = NiftiHeader::default();

    // try the system's native endianness first
    type B = NativeEndian;

    h.sizeof_hdr = input.read_i32::<B>()?;
    input.read_exact(&mut h.data_type)?;
    input.read_exact(&mut h.db_name)?;
    h.extents = input.read_i32::<B>()?;
    h.session_error = input.read_i16::<B>()?;
    h.regular = input.read_u8()?;
    h.dim_info = input.read_u8()?;
    h.dim[0] = input.read_u16::<B>()?;

    if h.dim[0] > 7 {
        h.endianness = Endianness::system().opposite();

        // swap bytes read so far, continue with the opposite endianness
        h.sizeof_hdr = h.sizeof_hdr.swap_bytes();
        h.extents = h.extents.swap_bytes();
        h.session_error = h.session_error.swap_bytes();
        h.dim[0] = h.dim[0].swap_bytes();
        parse_header_2::<OppositeNativeEndian, _>(h, input)
    } else {
        // all is well
        h.endianness = Endianness::system();
        parse_header_2::<B, _>(h, input)
    }
}

/// second part of header parsing
fn parse_header_2<B: ByteOrder, S: Read>(mut h: NiftiHeader, mut input: S) -> Result<NiftiHeader> {
    for v in &mut h.dim[1..] {
        *v = input.read_u16::<B>()?;
    }
    h.intent_p1 = input.read_f32::<B>()?;
    h.intent_p2 = input.read_f32::<B>()?;
    h.intent_p3 = input.read_f32::<B>()?;
    h.intent_code = input.read_i16::<B>()?;
    h.datatype = input.read_i16::<B>()?;
    h.bitpix = input.read_i16::<B>()?;
    h.slice_start = input.read_i16::<B>()?;
    for v in &mut h.pixdim {
        *v = input.read_f32::<B>()?;
    }
    h.vox_offset = input.read_f32::<B>()?;
    h.scl_slope = input.read_f32::<B>()?;
    h.scl_inter = input.read_f32::<B>()?;
    h.slice_end = input.read_i16::<B>()?;
    h.slice_code = input.read_u8()?;
    h.xyzt_units = input.read_u8()?;
    h.cal_max = input.read_f32::<B>()?;
    h.cal_min = input.read_f32::<B>()?;
    h.slice_duration = input.read_f32::<B>()?;
    h.toffset = input.read_f32::<B>()?;
    h.glmax = input.read_i32::<B>()?;
    h.glmin = input.read_i32::<B>()?;

    // descrip is 80-elem vec already
    input.read_exact(h.descrip.as_mut_slice())?;
    input.read_exact(&mut h.aux_file)?;
    h.qform_code = input.read_i16::<B>()?;
    h.sform_code = input.read_i16::<B>()?;
    h.quatern_b = input.read_f32::<B>()?;
    h.quatern_c = input.read_f32::<B>()?;
    h.quatern_d = input.read_f32::<B>()?;
    h.quatern_x = input.read_f32::<B>()?;
    h.quatern_y = input.read_f32::<B>()?;
    h.quatern_z = input.read_f32::<B>()?;
    for v in &mut h.srow_x {
        *v = input.read_f32::<B>()?;
    }
    for v in &mut h.srow_y {
        *v = input.read_f32::<B>()?;
    }
    for v in &mut h.srow_z {
        *v = input.read_f32::<B>()?;
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
