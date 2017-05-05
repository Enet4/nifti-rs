use error::Result;
use util::{ReadSeek, Endianness};
use std::fs::File;
use std::io::SeekFrom;
use std::io::BufReader;
use std::path::Path;
use byteorder::{ByteOrder, ReadBytesExt, LittleEndian, BigEndian};

pub const MAGIC_CODE_NI1 : &'static [u8; 4] = b"ni1\0";
pub const MAGIC_CODE_NIP1 : &'static [u8; 4] = b"n+1\0";

#[derive(Debug, Clone, PartialEq)]
pub struct NIFTIHeader {

    pub sizeof_hdr: i32,
    pub data_type: [u8; 10],
    pub db_name: [u8; 10],
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

impl Default for NIFTIHeader {
    fn default() -> NIFTIHeader {
        NIFTIHeader {
            sizeof_hdr: 348,
            data_type: [0; 10],
            db_name: [0; 10],
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
            scl_slope: 1.,
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
struct NIFTIExtender {
    extension: [u8; 4],
}

impl NIFTIHeader {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<NIFTIHeader> {
        let file = File::open(path)?;
        NIFTIHeader::from_stream(file)
    }

    pub fn from_stream<S: ReadSeek>(mut input: S) -> Result<NIFTIHeader> {
        match detect_endianness_seekable(&mut input)? {
            Endianness::LE => parse_header::<LittleEndian, _>(input),
            Endianness::BE => parse_header::<BigEndian, _>(input),
        }
    }
}

/// Detect the endianness of a NIFTI object
/// It is assumed that the input is currently at the start of the
/// NIFTI header.
fn detect_endianness_seekable<S: ReadSeek>(mut input: S) -> Result<Endianness> {
    // move to dim[0]
    const OFFSET: i64 = 4 + 10 + 18 + 4 + 2 + 1 + 1;
    input.seek(SeekFrom::Current(OFFSET))?;
    let dim0 = input.read_i16::<LittleEndian>()?;

    let e = if dim0 < 1 || dim0 > 7 {
        Endianness::BE
    } else {
        Endianness::LE
    };

    // and back to the beginning
    input.seek(SeekFrom::Current(-OFFSET))?;
    Ok(e)
}

fn parse_header<B: ByteOrder, S: ReadSeek>(mut input: S) -> Result<NIFTIHeader> {
    let mut h = NIFTIHeader::default();

    let mut input = BufReader::new(input);

    h.sizeof_hdr = input.read_i32::<B>()?;
    
    // TODO

    unimplemented!()
    //Ok(h)
}

