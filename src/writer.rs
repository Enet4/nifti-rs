//! Utility functions to write nifti images.

use std::fs::File;
use std::io::BufWriter;
use std::ops::{Div, Sub};
use std::path::{Path, PathBuf};

use byteorder::{LittleEndian, WriteBytesExt};
use flate2::write::GzEncoder;
use flate2::Compression;
use ndarray::{ArrayBase, ArrayView, Axis, Data, Dimension, RemoveAxis, ScalarOperand};
use num_traits::FromPrimitive;
use safe_transmute::{guarded_transmute_to_bytes_pod_many, PodTransmutable};

use {
    header::{MAGIC_CODE_NIP1, MAGIC_CODE_NI1},
    util::{is_gz_file, is_hdr_file},
    volume::element::DataElement,
    NiftiHeader, NiftiType, Result,
};

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
    let compression_level = Compression::fast();
    let is_gz = is_gz_file(&path);
    let (header, header_path, data_path) =
        prepare_header_and_paths(path, data, reference, A::DATA_TYPE);

    // Need the transpose for fortran ordering used in nifti file format.
    let data = data.t();

    let header_file = File::create(&header_path)?;
    let mut header_writer = BufWriter::new(header_file);
    if header.vox_offset > 0.0 {
        if is_gz {
            let mut e = GzEncoder::new(header_writer, compression_level);
            write_header(&mut e, &header)?;
            write_data(&mut e, &header, data)?;
            let _ = e.finish()?;
        } else {
            write_header(&mut header_writer, &header)?;
            write_data(&mut header_writer, &header, data)?;
        }
    } else {
        let data_file = File::create(&data_path)?;
        let mut data_writer = BufWriter::new(data_file);
        if is_gz {
            let mut e = GzEncoder::new(header_writer, compression_level);
            write_header(&mut e, &header)?;
            let _ = e.finish()?;

            let mut e = GzEncoder::new(data_writer, compression_level);
            write_data(&mut e, &header, data)?;
            let _ = e.finish()?;
        } else {
            write_header(&mut header_writer, &header)?;
            write_data(&mut data_writer, &header, data)?;
        }
    }

    Ok(())
}

/// Write a RGB nifti file (.nii or .nii.gz) in Little Endian.
///
/// If a `reference` is given, it will be used to fill most of the header's fields, except those
/// necessary to be recognized as a RGB image. `scl_slope` will be set to 1.0 and `scl_inter` to
/// 0.0.  If `reference` is not given, a default `NiftiHeader` will be built and written.
pub fn write_rgb_nifti<P, S, D>(
    path: P,
    data: &ArrayBase<S, D>,
    reference: Option<&NiftiHeader>,
) -> Result<()>
where
    P: AsRef<Path>,
    S: Data<Elem = [u8; 3]>,
    D: Dimension + RemoveAxis,
{
    let is_gz = is_gz_file(&path);
    let (mut header, header_path, _) =
        prepare_header_and_paths(path, data, reference, NiftiType::Rgb24);

    // The `scl_slope` and `scl_inter` fields are ignored on the Rgb24 type.
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;

    // Need the transpose for fortran used in nifti file format.
    let data = data.t();

    let f = File::create(&header_path)?;
    let mut writer = BufWriter::new(f);
    if is_gz {
        let mut e = GzEncoder::new(writer, Compression::fast());
        write_header(&mut e, &header)?;
        write_slices(&mut e, data)?;
        let _ = e.finish()?; // Must use result
    } else {
        write_header(&mut writer, &header)?;
        write_slices(&mut writer, data)?;
    }
    Ok(())
}

fn prepare_header_and_paths<P, T, D>(
    path: P,
    data: &ArrayBase<T, D>,
    reference: Option<&NiftiHeader>,
    datatype: NiftiType,
) -> (NiftiHeader, P, PathBuf)
where
    P: AsRef<Path>,
    T: Data,
    D: Dimension,
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
            header.pixdim = [1.0; 8];
            header.sform_code = 2;
            header.srow_x = [1.0, 0.0, 0.0, 0.0];
            header.srow_y = [0.0, 1.0, 0.0, 0.0];
            header.srow_z = [0.0, 0.0, 1.0, 0.0];
            header
        }
    };

    let mut header = NiftiHeader {
        dim,
        sizeof_hdr: 348,
        datatype: datatype as i16,
        bitpix: (datatype.size_of() * 8) as i16,
        vox_offset: 352.0,
        magic: *MAGIC_CODE_NIP1,
        // All other fields are copied from reference header
        ..reference
    };

    let mut path_buf = PathBuf::from(path.as_ref());
    let (header_path, data_path) = if is_hdr_file(&path) {
        header.vox_offset = 0.0;
        header.magic = *MAGIC_CODE_NI1;
        let data_path = if is_gz_file(&path) {
            let _ = path_buf.set_extension("");
            path_buf.with_extension("img.gz")
        } else {
            path_buf.with_extension("img")
        };
        (path, data_path)
    } else {
        (path, path_buf)
    };

    (header, header_path, data_path)
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
fn write_data<T, D, W>(writer: &mut W, header: &NiftiHeader, data: ArrayView<T, D>) -> Result<()>
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
        // TODO Use linear transformation like when reading. An scl_slope of 0.5 would turn all
        // voxel values to 0 if we pass an ndarray of integers.
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

fn write_slices<A, S, D, W>(writer: &mut W, data: ArrayBase<S, D>) -> Result<()>
where
    S: Data<Elem = A>,
    A: Clone + PodTransmutable,
    D: Dimension + RemoveAxis,
    W: WriteBytesExt,
{
    let mut iter = data.axis_iter(Axis(0));
    if let Some(arr_data) = iter.next() {
        // Keep slice voxels in a separate array to ensure `C` ordering even after `into_shape`.
        let mut slice = arr_data.to_owned();
        write_slice(writer, slice.view())?;
        for arr_data in iter {
            slice.assign(&arr_data);
            write_slice(writer, slice.view())?;
        }
    }
    Ok(())
}

fn write_slice<A, S, D, W>(writer: &mut W, data: ArrayBase<S, D>) -> Result<()>
where
    S: Data<Elem = A>,
    A: Clone + PodTransmutable,
    D: Dimension,
    W: WriteBytesExt,
{
    let len = data.len();
    let arr_data = data.into_shape(len).unwrap();
    let slice = arr_data.as_slice().unwrap();
    writer.write_all(guarded_transmute_to_bytes_pod_many(slice))?;
    Ok(())
}
