use error::{NiftiError, Result};
use util::{ReadSeek, Endianness};
use std::fs::File;
use std::io::SeekFrom;
use std::io::{Read, BufReader, BufRead};
use std::path::Path;
use byteorder::{ByteOrder, ReadBytesExt, LittleEndian, BigEndian, NativeEndian};
use flate2::bufread::GzDecoder;

pub const MAGIC_CODE_NI1 : &'static [u8; 4] = b"ni1\0";
pub const MAGIC_CODE_NIP1 : &'static [u8; 4] = b"n+1\0";

#[derive(Debug, Clone, PartialEq)]
pub struct NiftiHeader {

    pub sizeof_hdr: i32,
    pub data_type: [u8; 10],
    pub db_name: [u8; 18],
    pub extents: i32,
    pub session_error: i16,
    pub regular: u8,
    pub dim_info: u8,
    pub dim: [i16; 8],
    pub intent_p1: f32,
    pub intent_p2: f32,
    pub intent_p3: f32,
    pub intent_code: i16,
    pub datatype: i16,
    pub bitpix: i16,
    pub slice_start: i16,
    pub pixdim: [f32; 8],
    pub vox_offset: f32,
    pub scl_slope: f32,
    pub scl_inter: f32,
    pub slice_end: i16,
    pub slice_code: u8,
    pub xyzt_units: u8,
    pub cal_max: f32,
    pub cal_min: f32,
    pub slice_duration: f32,
    pub toffset: f32,
    pub glmax: i32,
    pub glmin: i32,

    pub descrip: Vec<u8>,
    pub aux_file: [u8; 24],
    pub qform_code: i16,
    pub sform_code: i16,
    pub quatern_b: f32,
    pub quatern_c: f32,
    pub quatern_d: f32,
    pub quatern_x: f32,
    pub quatern_y: f32,
    pub quatern_z: f32,

    pub srow_x: [f32; 4],
    pub srow_y: [f32; 4],
    pub srow_z: [f32; 4],

    pub intent_name: [u8; 16],

    pub magic: [u8; 4],
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
        }
    }
}



#[derive(Debug, Default, Clone, Copy)]
struct NiftiExtender {
    extension: [u8; 4],
}

impl NiftiHeader {
    /// Retrieve a NIFTI header, along with its byte order, from a file in the file system.
    /// If the file's name ends with ".gz", the file is assumed to need GZip decoding.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<(NiftiHeader, Endianness)> {
        let gz = path.as_ref().extension()
            .map(|a| a.to_string_lossy() == "gz")
            .unwrap_or(false);
        let file = BufReader::new(File::open(path)?);
        if gz {
            NiftiHeader::from_stream(GzDecoder::new(file)?)
        } else {
            NiftiHeader::from_stream(file)
        }
    }
    
    /// Read a NIfTI-1 header, along with its byte order, from the given byte stream.
    /// It is assumed that the input is currently at the start of the
    /// NIFTI header.
    pub fn from_stream<S: Read>(input: S) -> Result<(NiftiHeader, Endianness)> {
        parse_header_1(input)
    }
}

/// Defines the serialization that is opposite to system native-endian.
/// This is `BigEndian` in a Little Endian system and `LittleEndian` in a Big Endian system.
///
/// Note that this type has no value constructor. It is used purely at the
/// type level.
#[cfg(target_endian = "little")]
type OppositeNativeEndian = BigEndian;

/// Defines the serialization that is opposite to system native-endian.
/// This is `BigEndian` in a Little Endian system and `LittleEndian` in a Big Endian system.
///
/// Note that this type has no value constructor. It is used purely at the
/// type level.
#[cfg(target_endian = "big")]
type OppositeNativeEndian = LittleEndian;

fn parse_header_1<S: Read>(mut input: S) -> Result<(NiftiHeader, Endianness)> {
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
    h.dim[0] = input.read_i16::<B>()?;

    if h.dim[0] < 0 || h.dim[0] > 7 {
        // swap bytes read so far, continue with the opposite endianness
        h.sizeof_hdr = h.sizeof_hdr.swap_bytes();
        h.extents = h.extents.swap_bytes();
        h.session_error = h.session_error.swap_bytes();
        h.dim[0] = h.dim[0].swap_bytes();
        parse_header_2::<OppositeNativeEndian, _>(h, input)
            .map(|a| (a, Endianness::system().opposite()))
    } else {
        // all is well
        parse_header_2::<B, _>(h, input)
            .map(|a| (a, Endianness::system()))
    }
}

/// second part of header parsing
fn parse_header_2<B: ByteOrder, S: Read>(mut h: NiftiHeader, mut input: S) -> Result<NiftiHeader> {
    for v in &mut h.dim[1..] {
        *v = input.read_i16::<B>()?;
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

    debug_assert_eq!(&h.magic, MAGIC_CODE_NI1);
    debug_assert_eq!(h.descrip.len(), 80);

    if &h.magic != MAGIC_CODE_NI1 && &h.magic != MAGIC_CODE_NIP1 {
        Err(NiftiError::InvalidFormat)
    } else {
        Ok(h)
    }
}

