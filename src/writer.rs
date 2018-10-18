//! Utility traits and functions to write nifti images.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::mem;
use std::ops::{Div, Sub};
use std::slice::from_raw_parts;

use byteorder::{LittleEndian, WriteBytesExt};
use flate2::Compression;
use flate2::write::GzEncoder;
use ndarray::{ArrayBase, Axis, Data, Dimension, RemoveAxis, ScalarOperand};
use num_traits::FromPrimitive;

use {NiftiHeader, util::is_gz_file};

type B = LittleEndian;

/// Specific actions and header information that depend on the ndarray type.
pub trait TypeTool<T> {

    /// Returns the `datatype` mapped to the type T
    fn to_data_type() -> i16;

    /// Calls the right `write` method. For example, if T is i16, call `write_i16`.
    fn write<W: WriteBytesExt>(bytes: &mut W, d: T);
}

impl TypeTool<u8> for u8 {
    fn to_data_type() -> i16 { 2 }
    fn write<W: WriteBytesExt>(writer: &mut W, d: u8)
    {
        writer.write_u8(d).unwrap();
    }
}
impl TypeTool<i16> for i16 {
    fn to_data_type() -> i16 { 4 }
    fn write<W: WriteBytesExt>(writer: &mut W, d: i16) {
        writer.write_i16::<B>(d).unwrap();
    }
}
impl TypeTool<i32> for i32 {
    fn to_data_type() -> i16 { 8 }
    fn write<W: WriteBytesExt>(writer: &mut W, d: i32) {
        writer.write_i32::<B>(d).unwrap();
    }
}
impl TypeTool<f32> for f32 {
    fn to_data_type() -> i16 { 16 }
    fn write<W: WriteBytesExt>(writer: &mut W, d: f32) {
        writer.write_f32::<B>(d).unwrap();
    }
}
// NIFTI_TYPE_COMPLEX64      32
impl TypeTool<f64> for f64 {
    fn to_data_type() -> i16 { 64 }
    fn write<W: WriteBytesExt>(writer: &mut W, d: f64) {
        writer.write_f64::<B>(d).unwrap();
    }
}
// NIFTI_TYPE_RGB24         128
impl TypeTool<i8> for i8 {
    fn to_data_type() -> i16 { 256 }
    fn write<W: WriteBytesExt>(writer: &mut W, d: i8) {
        writer.write_i8(d).unwrap();
    }
}
impl TypeTool<u16> for u16 {
    fn to_data_type() -> i16 { 512 }
    fn write<W: WriteBytesExt>(writer: &mut W, d: u16) {
        writer.write_u16::<B>(d).unwrap();
    }
}
impl TypeTool<u32> for u32 {
    fn to_data_type() -> i16 { 768 }
    fn write<W: WriteBytesExt>(writer: &mut W, d: u32) {
        writer.write_u32::<B>(d).unwrap();
    }
}
impl TypeTool<i64> for i64 {
    fn to_data_type() -> i16 { 1024 }
    fn write<W: WriteBytesExt>(writer: &mut W, d: i64) {
        writer.write_i64::<B>(d).unwrap();
    }
}
impl TypeTool<u64> for u64 {
    fn to_data_type() -> i16 { 1280 }
    fn write<W: WriteBytesExt>(writer: &mut W, d: u64) {
        writer.write_u64::<B>(d).unwrap();
    }
}
// NIFTI_TYPE_FLOAT128     1536
// NIFTI_TYPE_COMPLEX128   1792
// NIFTI_TYPE_COMPLEX256   2048
// NIFTI_TYPE_RGBA32       2304

/// Write a nifti file (.nii or .nii.gz).
///
/// If a `reference` is given, it will be used to fill most of the header's fields. The voxels
/// intensity will be subtracted by `scl_slope` and divided by `scl_inter`. If `reference` is not
/// given, a default `NiftiHeader` will be built and written.
///
/// In all cases, the `dim`, `datatype` and `bitpix` fields will depend only on `data`, not on the
/// header. In other words, the `datatype` defined in `reference` will be ignored.
pub fn write_nifti<A, S, D>(
    path: &str,
    data: &ArrayBase<S, D>,
    reference: Option<&NiftiHeader>)
    where S: Data<Elem=A>,
          A: TypeTool<A> + Copy + Div<Output = A> + FromPrimitive + ScalarOperand + Sub<Output = A>,
          D: Dimension + RemoveAxis
{
    let mut dim = [1; 8];
    dim[0] = data.ndim() as u16;
    for (i, s) in data.shape().iter().enumerate() {
        dim[i + 1] = *s as u16;
    }

    // If no reference header is given, use the default.
    let reference = match reference {
        Some(r) => r.clone(),
        None => NiftiHeader::default()
    };
    let header = NiftiHeader {
        dim,
        datatype: A::to_data_type(),
        bitpix: (mem::size_of::<A>() * 8) as i16,
        // All other fields are copied from reference header
        ..reference
    };

    let f = File::create(path).expect("Can't create new nifti file");
    let mut writer = BufWriter::new(f);
    if is_gz_file(&path) {
        let mut buffer = Vec::new();
        write(&mut buffer, header, data);

        let mut e = GzEncoder::new(Vec::new(), Compression::default());
        e.write_all(&buffer).unwrap();

        let buffer = e.finish().unwrap();
        writer.write_all(&buffer).unwrap();
    } else {
        write(&mut writer, header, data);
    }
}

fn write<A, S, D, W>(
    writer: &mut W,
    header: NiftiHeader,
    data: &ArrayBase<S, D>,)
    where S: Data<Elem=A>,
          A: TypeTool<A> + Copy + Div<Output = A> + FromPrimitive + ScalarOperand + Sub<Output = A>,
          D: Dimension + RemoveAxis,
          W: WriteBytesExt
{
    writer.write_i32::<B>(header.sizeof_hdr).unwrap();
    writer.write_all(&header.data_type).unwrap();
    writer.write_all(&header.db_name).unwrap();
    writer.write_i32::<B>(header.extents).unwrap();
    writer.write_i16::<B>(header.session_error).unwrap();
    writer.write_u8(header.regular).unwrap();
    writer.write_u8(header.dim_info).unwrap();
    for s in &header.dim {
        writer.write_u16::<B>(*s).unwrap();
    }
    writer.write_f32::<B>(header.intent_p1).unwrap();
    writer.write_f32::<B>(header.intent_p2).unwrap();
    writer.write_f32::<B>(header.intent_p3).unwrap();
    writer.write_i16::<B>(header.intent_code).unwrap();
    writer.write_i16::<B>(header.datatype).unwrap();
    writer.write_i16::<B>(header.bitpix).unwrap();
    writer.write_i16::<B>(header.slice_start).unwrap();
    for f in &header.pixdim {
        writer.write_f32::<B>(*f).unwrap();
    }
    writer.write_f32::<B>(header.vox_offset).unwrap();
    writer.write_f32::<B>(header.scl_slope).unwrap();
    writer.write_f32::<B>(header.scl_inter).unwrap();
    writer.write_i16::<B>(header.slice_end).unwrap();
    writer.write_u8(header.slice_code).unwrap();
    writer.write_u8(header.xyzt_units).unwrap();
    writer.write_f32::<B>(header.cal_max).unwrap();
    writer.write_f32::<B>(header.cal_min).unwrap();
    writer.write_f32::<B>(header.slice_duration).unwrap();
    writer.write_f32::<B>(header.toffset).unwrap();
    writer.write_i32::<B>(header.glmax).unwrap();
    writer.write_i32::<B>(header.glmin).unwrap();

    writer.write_all(&header.descrip).unwrap();
    writer.write_all(&header.aux_file).unwrap();
    writer.write_i16::<B>(header.qform_code).unwrap();
    writer.write_i16::<B>(header.sform_code).unwrap();
    for f in &[header.quatern_b, header.quatern_c, header.quatern_d,
               header.quatern_x, header.quatern_y, header.quatern_z] {
        writer.write_f32::<B>(*f).unwrap();
    }
    for f in header.srow_x.iter().chain(&header.srow_y).chain(&header.srow_z) {
        writer.write_f32::<B>(*f).unwrap();
    }
    writer.write_all(&header.intent_name).unwrap();
    writer.write_all(&header.magic).unwrap();

    // Empty 4 bytes after the header
    writer.write_u32::<B>(0).unwrap();

    // We finally write the data.
    // Like NiBabel, we iterate by "slice" to improve speed and use less memory

    // Need the transpose for fortran used in nifti file format.
    let data = data.t();

    let arr_len = data.subview(Axis(0), 0).len();
    let resolution: usize = header.dim[1..(header.dim[0] + 1) as usize]
        .iter().map(|d| *d as usize).product();
    let nb_bytes = resolution * header.bitpix as usize / 8;
    let dim0_size = data.shape()[0];

    // `1.0x + 0.0` would give the same results, but we avoid a lot of divisions
    let slope = if header.scl_slope == 0.0 { 1.0 } else { header.scl_slope };
    if slope != 1.0 || header.scl_inter != 0.0 {
        let slope = A::from_f32(slope).unwrap();
        let inter = A::from_f32(header.scl_inter).unwrap();
        for arr_data in data.axis_iter(Axis(0)) {
            let arr_data = arr_data.sub(inter).div(slope);
            let arr_data = arr_data.into_shape(arr_len).unwrap();
            let arr_data = arr_data.as_slice().unwrap();

            let buffer = to_bytes(&arr_data, nb_bytes / dim0_size);
            writer.write_all(&buffer).unwrap();
        }
    } else {
        for arr_data in data.axis_iter(Axis(0)) {
            // We need to own the data because of the into_shape() for `C` ordering.
            let arr_data = arr_data.to_owned();
            let arr_data = arr_data.into_shape(arr_len).unwrap();
            let arr_data = arr_data.as_slice().unwrap();

            let buffer = to_bytes(&arr_data, nb_bytes / dim0_size);
            writer.write_all(&buffer).unwrap();
        }
    }
}

fn to_bytes<T>(data: &[T], size: usize) -> &[u8] {
    unsafe { from_raw_parts::<u8>(data.as_ptr() as *const u8, size) }
}
