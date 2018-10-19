//! Utility functions to write nifti images.

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

use {NiftiHeader, volume::element::DataElement, util::is_gz_file};

type B = LittleEndian;

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
          A: Copy + DataElement + Div<Output = A> + FromPrimitive + ScalarOperand + Sub<Output = A>,
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
        datatype: A::DATA_TYPE as i16,
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
          A: Copy + DataElement + Div<Output = A> + FromPrimitive + ScalarOperand + Sub<Output = A>,
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

#[cfg(test)]
pub mod tests {
    extern crate tempfile;

    use super::*;

    use ndarray::{Array, Array2, Ix2, IxDyn, ShapeBuilder};
    use self::tempfile::tempdir;

    use {InMemNiftiObject, IntoNdArray, header::MAGIC_CODE_NIP1, object::NiftiObject};

    fn get_random_path(ext: &str) -> String {
        let dir = tempdir().unwrap();
        let path = if ext == "" {
            dir.into_path()
        } else {
            dir.into_path().join(ext)
        };
        path.to_str().unwrap().to_string()
    }

    pub fn generate_nifti_header(
        dim: [u16; 8],
        scl_slope: f32,
        scl_inter: f32,
        datatype: i16
    ) -> NiftiHeader {
        let bitpix = (mem::size_of::<f32>() * 8) as i16;
        let magic = *MAGIC_CODE_NIP1;
        NiftiHeader {
            dim, datatype, bitpix, magic, scl_slope, scl_inter, ..NiftiHeader::default()
        }
    }

    fn read_2d_image(path: &str) -> Array2<f32> {
        let nifti_object = InMemNiftiObject::from_file(path).expect("Nifti file is unreadable.");
        let volume = nifti_object.into_volume();
        let dyn_data = volume.into_ndarray().unwrap();
        dyn_data.into_dimensionality::<Ix2>().unwrap()
    }

    fn test_write_read(arr: &Array<f32, IxDyn>, path: &str) {
        let path = get_random_path(path);
        let mut dim = [1; 8];
        dim[0] = arr.ndim() as u16;
        for (i, s) in arr.shape().iter().enumerate() {
            dim[i + 1] = *s as u16;
        }
        let header = generate_nifti_header(dim, 1.0, 0.0, 16);
        write_nifti(&path, &arr, Some(&header));

        let read_nifti = read_2d_image(&path);
        assert!(read_nifti.all_close(&arr, 1e-10) == true);
    }

    fn f_order_array() -> Array<f32, IxDyn> {
        let dim = vec![4, 4];
        let vec = (0..16).map(|x| x as f32).collect();
        Array::from_shape_vec(IxDyn(&dim).f(), vec).unwrap()
    }

    fn c_order_array() -> Array<f32, IxDyn> {
        let dim = vec![4, 4];
        let vec = (0..16).map(|x| x as f32).collect();
        Array::from_shape_vec(IxDyn(&dim), vec).unwrap()
    }

    #[test]
    fn test_fortran_writing() {
        // Test .nii
        let arr = f_order_array();
        test_write_read(&arr, "test.nii");
        let mut arr = f_order_array();
        arr.invert_axis(Axis(1));
        test_write_read(&arr, "test_non_contiguous.nii");

        // Test .nii.gz
        let arr = f_order_array();
        test_write_read(&arr, "test.nii.gz");
        let mut arr = f_order_array();
        arr.invert_axis(Axis(1));
        test_write_read(&arr, "test_non_contiguous.nii.gz");
    }

    #[test]
    fn test_c_writing() {
        // Test .nii
        let arr = c_order_array();
        test_write_read(&arr, "test.nii");

        let mut arr = c_order_array();
        arr.invert_axis(Axis(1));
        test_write_read(&arr, "test_non_contiguous.nii");

        // Test .nii.gz
        let arr = c_order_array();
        test_write_read(&arr, "test.nii.gz");

        let mut arr = c_order_array();
        arr.invert_axis(Axis(1));
        test_write_read(&arr, "test_non_contiguous.nii.gz");
    }

    #[test]
    fn test_header_slope_inter() {
        use std::ops::{Add, Mul};

        let arr = f_order_array();
        let slope = 2.2;
        let inter = 101.1;

        let path = get_random_path("test_slope_inter.nii");
        let mut dim = [1; 8];
        dim[0] = arr.ndim() as u16;
        for (i, s) in arr.shape().iter().enumerate() {
            dim[i + 1] = *s as u16;
        }
        let header = generate_nifti_header(dim, slope, inter, 16);
        let transformed_data = arr.mul(slope).add(inter);
        write_nifti(&path, &transformed_data, Some(&header));

        let read_nifti = read_2d_image(&path);
        assert!(read_nifti.all_close(&transformed_data, 1e-10) == true);
    }
}
