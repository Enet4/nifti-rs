//! Utility functions to write nifti images.

use std::fs::File;
use std::io::BufWriter;
use std::ops::{Div, Sub};
use std::path::Path;

use byteorder::{LittleEndian, WriteBytesExt};
use flate2::write::GzEncoder;
use flate2::Compression;
use ndarray::{Array, ArrayBase, ArrayView, Axis, Data, Dimension, RemoveAxis, ScalarOperand};
use num_traits::FromPrimitive;
use safe_transmute::{guarded_transmute_to_bytes_pod_many, PodTransmutable};

use {util::is_gz_file, volume::element::DataElement, NiftiHeader, NiftiType, Result};

// TODO make this configurable. The Nifti standard does not specify a specific field for endianness,
// but it is encoded in `dim[0]`. "if dim[0] is outside range 1..7, then swap".
type B = LittleEndian;

/// Write a nifti file (.nii or .nii.gz) in Little Endian.
///
/// If a `reference` is given, it will be used to fill most of the header's fields. The voxels
/// intensity will be subtracted by `scl_slope` and divided by `scl_inter`. If `reference` is not
/// given, a default `NiftiHeader` will be built and written.
///
/// In all cases, the `dim`, `datatype` and `bitpix` fields will depend only on `data`, not on the
/// header. In other words, the `datatype` defined in `reference` will be ignored.
pub fn write_nifti<P, A, S, D>(
    path: P,
    data: &ArrayBase<S, D>,
    reference: Option<&NiftiHeader>,
) -> Result<()>
where
    P: AsRef<Path>,
    S: Data<Elem = A>,
    A: Copy,
    A: DataElement,
    A: Div<Output = A>,
    A: FromPrimitive,
    A: PodTransmutable,
    A: ScalarOperand,
    A: Sub<Output = A>,
    D: Dimension + RemoveAxis,
{
    let header = build_header(data, reference, A::DATA_TYPE);

    // Need the transpose for fortran ordering used in nifti file format.
    let data = data.t();

    let f = File::create(&path)?;
    let mut writer = BufWriter::new(f);
    if is_gz_file(&path) {
        let mut e = GzEncoder::new(writer, Compression::fast());
        write_header(&mut e, &header)?;
        write_data(&mut e, header, data)?;
        let _ = e.finish()?; // Must use result
    } else {
        write_header(&mut writer, &header)?;
        write_data(&mut writer, header, data)?;
    }
    Ok(())
}

/// Write a RGB nifti file (.nii or .nii.gz) in Little Endian.
///
/// If a `reference` is given, it will be used to fill most of the header's fields, except those
/// necessary to be recognized as a RGB image. `scl_slope` will be set to 1.0 and `scl_inter` to
/// 0.0.  If `reference` is not given, a default `NiftiHeader` will be built and written.
pub fn write_rgb_nifti<P, D>(
    path: P,
    data: &Array<[u8; 3], D>,
    reference: Option<&NiftiHeader>
) -> Result<()>
where
    P: AsRef<Path>,
    D: Dimension + RemoveAxis,
{
    // Copy most of the reference header, but the `bitpix`, `scl_slope` and `scl_inter` fields must
    // have those exact values.
    let mut header = build_header(data, reference, NiftiType::Rgb24);
    header.bitpix = 24;
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;

    // Need the transpose for fortran used in nifti file format.
    let data = data.t();

    let f = File::create(&path)?;
    let mut writer = BufWriter::new(f);
    if is_gz_file(&path) {
        let mut e = GzEncoder::new(writer, Compression::fast());
        write_header(&mut e, &header)?;
        write_slices(&mut e, data)?;
        let _ = e.finish().unwrap(); // Must use result
    } else {
        write_header(&mut writer, &header)?;
        write_slices(&mut writer, data)?;
    }
    Ok(())
}

fn build_header<T, D>(
    data: &ArrayBase<T, D>,
    reference: Option<&NiftiHeader>,
    datatype: NiftiType
) -> NiftiHeader
where
    T: Data,
    D: Dimension
{
    let mut dim = [1; 8];
    dim[0] = data.ndim() as u16;
    for (i, s) in data.shape().iter().enumerate() {
        dim[i + 1] = *s as u16;
    }

    // If no reference header is given, use the default.
    let reference = match reference {
        Some(r) => r.clone(),
        None => {
            let mut header = NiftiHeader::default();

            // Some programs require a non-zero spacing
            for i in 0..data.ndim() {
                if header.pixdim[i] == 0.0 {
                    header.pixdim[i] = 1.0;
                }
            }
            header
        },
    };

    NiftiHeader {
        dim,
        datatype: datatype as i16,
        bitpix: (datatype.size_of() * 8) as i16,
        // All other fields are copied from reference header
        ..reference
    }
}

fn write_header<W>(writer: &mut W, header: &NiftiHeader) -> Result<()>
where
    W: WriteBytesExt,
{
    writer.write_i32::<B>(header.sizeof_hdr)?;
    writer.write_all(&header.data_type)?;
    writer.write_all(&header.db_name)?;
    writer.write_i32::<B>(header.extents)?;
    writer.write_i16::<B>(header.session_error)?;
    writer.write_u8(header.regular)?;
    writer.write_u8(header.dim_info)?;
    for s in &header.dim {
        writer.write_u16::<B>(*s)?;
    }
    writer.write_f32::<B>(header.intent_p1)?;
    writer.write_f32::<B>(header.intent_p2)?;
    writer.write_f32::<B>(header.intent_p3)?;
    writer.write_i16::<B>(header.intent_code)?;
    writer.write_i16::<B>(header.datatype)?;
    writer.write_i16::<B>(header.bitpix)?;
    writer.write_i16::<B>(header.slice_start)?;
    for f in &header.pixdim {
        writer.write_f32::<B>(*f)?;
    }
    writer.write_f32::<B>(header.vox_offset)?;
    writer.write_f32::<B>(header.scl_slope)?;
    writer.write_f32::<B>(header.scl_inter)?;
    writer.write_i16::<B>(header.slice_end)?;
    writer.write_u8(header.slice_code)?;
    writer.write_u8(header.xyzt_units)?;
    writer.write_f32::<B>(header.cal_max)?;
    writer.write_f32::<B>(header.cal_min)?;
    writer.write_f32::<B>(header.slice_duration)?;
    writer.write_f32::<B>(header.toffset)?;
    writer.write_i32::<B>(header.glmax)?;
    writer.write_i32::<B>(header.glmin)?;

    writer.write_all(&header.descrip)?;
    writer.write_all(&header.aux_file)?;
    writer.write_i16::<B>(header.qform_code)?;
    writer.write_i16::<B>(header.sform_code)?;
    for f in &[
        header.quatern_b,
        header.quatern_c,
        header.quatern_d,
        header.quatern_x,
        header.quatern_y,
        header.quatern_z,
    ] {
        writer.write_f32::<B>(*f)?;
    }
    for f in header
        .srow_x
        .iter()
        .chain(&header.srow_y)
        .chain(&header.srow_z)
    {
        writer.write_f32::<B>(*f)?;
    }
    writer.write_all(&header.intent_name)?;
    writer.write_all(&header.magic)?;

    // Empty 4 bytes after the header
    // TODO Support writing extension data.
    writer.write_u32::<B>(0)?;

    Ok(())
}

/// Write the data in 'f' order.
///
/// Like NiBabel, we iterate by "slice" to improve speed and use less memory.
fn write_data<T, D, W>(writer: &mut W, header: NiftiHeader, data: ArrayView<T, D>) -> Result<()>
where
    T: Clone + PodTransmutable,
    T: Div<Output = T>,
    T: FromPrimitive,
    T: PodTransmutable,
    T: ScalarOperand,
    T: Sub<Output = T>,
    D: Dimension + RemoveAxis,
    W: WriteBytesExt,
{
    // `1.0x + 0.0` would give the same results, but we avoid a lot of divisions
    let slope = if header.scl_slope == 0.0 {
        1.0
    } else {
        header.scl_slope
    };
    if slope != 1.0 || header.scl_inter != 0.0 {
        let slope = T::from_f32(slope).unwrap();
        let inter = T::from_f32(header.scl_inter).unwrap();
        for arr_data in data.axis_iter(Axis(0)) {
            write_slice(writer, arr_data.sub(inter).div(slope))?;
        }
    } else {
        write_slices(writer, data)?;
    }
    Ok(())
}

fn write_slices<T, D, W>(writer: &mut W, data: ArrayView<T, D>) -> Result<()>
where
    T: Clone + PodTransmutable,
    D: Dimension + RemoveAxis,
    W: WriteBytesExt
{
    for arr_data in data.axis_iter(Axis(0)) {
        // We need to own the data because of the into_shape() for `C` ordering.
        write_slice(writer, arr_data.to_owned())?;
    }
    Ok(())
}

fn write_slice<T, D, W>(writer: &mut W, data: Array<T, D>) -> Result<()>
where
    T: Clone + PodTransmutable,
    D: Dimension,
    W: WriteBytesExt
{
    let len = data.len();
    let arr_data = data.into_shape(len).unwrap();
    let slice = arr_data.as_slice().unwrap();
    writer.write_all(guarded_transmute_to_bytes_pod_many(slice))?;
    Ok(())
}

#[cfg(test)]
pub mod tests {
    extern crate tempfile;

    use super::*;
    use std::ops::{Add, Mul};
    use std::path::PathBuf;

    use self::tempfile::tempdir;
    use ndarray::{Array, Array2, Ix2, IxDyn, ShapeBuilder};

    use {header::MAGIC_CODE_NIP1, object::NiftiObject, InMemNiftiObject, IntoNdArray};

    fn get_temporary_path(ext: &str) -> PathBuf {
        let dir = tempdir().unwrap();
        let mut path = dir.into_path();
        if !ext.is_empty() {
            path.push(ext);
        }
        path
    }

    pub fn generate_nifti_header(
        dim: [u16; 8],
        scl_slope: f32,
        scl_inter: f32,
        datatype: i16,
    ) -> NiftiHeader {
        NiftiHeader {
            dim,
            datatype,
            bitpix: (NiftiType::Float32.size_of() * 8) as i16,
            magic: *MAGIC_CODE_NIP1,
            scl_slope,
            scl_inter,
            ..NiftiHeader::default()
        }
    }

    fn read_2d_image<P>(path: P) -> Array2<f32>
    where
        P: AsRef<Path>,
    {
        let nifti_object = InMemNiftiObject::from_file(path).expect("Nifti file is unreadable.");
        let volume = nifti_object.into_volume();
        let dyn_data = volume.into_ndarray().unwrap();
        dyn_data.into_dimensionality::<Ix2>().unwrap()
    }

    fn test_write_read(arr: &Array<f32, IxDyn>, path: &str) {
        let path = get_temporary_path(path);
        let mut dim = [1; 8];
        dim[0] = arr.ndim() as u16;
        for (i, s) in arr.shape().iter().enumerate() {
            dim[i + 1] = *s as u16;
        }
        let header = generate_nifti_header(dim, 1.0, 0.0, 16);
        write_nifti(&path, &arr, Some(&header)).unwrap();

        let read_nifti = read_2d_image(path);
        assert!(read_nifti.all_close(&arr, 1e-10));
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
        let arr = f_order_array();
        let slope = 2.2;
        let inter = 101.1;

        let path = get_temporary_path("test_slope_inter.nii");
        let mut dim = [1; 8];
        dim[0] = arr.ndim() as u16;
        for (i, s) in arr.shape().iter().enumerate() {
            dim[i + 1] = *s as u16;
        }
        let header = generate_nifti_header(dim, slope, inter, 16);
        let transformed_data = arr.mul(slope).add(inter);
        write_nifti(&path, &transformed_data, Some(&header)).unwrap();

        let read_nifti = read_2d_image(path);
        assert!(read_nifti.all_close(&transformed_data, 1e-10));
    }

    #[test]
    fn test_write_3d_rgb() {
        let mut data = Array::from_elem((10, 10, 8), [255u8, 255u8, 255u8]);
        data[(5, 5, 2)] = [55, 55, 0];
        data[(5, 5, 3)] = [55, 0, 55];
        data[(5, 5, 4)] = [0, 55, 55];

        let path = get_temporary_path("rgb.nii.gz");
        write_rgb_nifti(path, &data, None).unwrap();
        // TODO Don't just test if the writing compiles and runs, we also need to read it back
    }

    #[test]
    fn test_write_4d_rgb() {
        let mut data = Array::from_elem((10, 10, 10, 2), [255u8, 255u8, 255u8]);
        for i in 0..2 {
            data[(5, 5, 2, i)] = [55, 55, 0];
            data[(5, 5, 3, i)] = [55, 0, 55];
            data[(5, 5, 4, i)] = [0, 55, 55];
        }

        let path = get_temporary_path("rgb.nii.gz");
        write_rgb_nifti(path, &data, None).unwrap();
        // TODO Don't just test if the writing compiles and runs, we also need to read it back
    }
}
