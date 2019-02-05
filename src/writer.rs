//! Utility functions to write nifti images.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::{Div, Sub};
use std::path::{Path, PathBuf};

use byteordered::{ByteOrdered, Endian};
use flate2::write::GzEncoder;
use flate2::Compression;
use ndarray::{ArrayBase, ArrayView, Axis, Data, Dimension, RemoveAxis, ScalarOperand};
use num_traits::FromPrimitive;
use safe_transmute::{guarded_transmute_to_bytes_pod_many, PodTransmutable};

use {
    header::{MAGIC_CODE_NI1, MAGIC_CODE_NIP1},
    util::{is_gz_file, is_hdr_file, adapt_bytes},
    volume::element::DataElement,
    NiftiHeader, NiftiType, Result,
};

/// Write a nifti file (.nii or .nii.gz) in Little Endian.
///
/// If a `reference` is given, it will be used to fill most of the header's fields. The voxels
/// intensity will be subtracted by `scl_slope` and divided by `scl_inter`. If `reference` is not
/// given, a default `NiftiHeader` will be built and written.
///
/// In all cases, the `dim`, `datatype` and `bitpix` fields will depend only on `data`, not on the
/// header. In other words, the `datatype` defined in `reference` will be ignored.
pub fn write_nifti<P, A, S, D>(
    header_path: P,
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
    let is_gz = is_gz_file(&header_path);
    let (header, data_path) =
        prepare_header_and_paths(&header_path, data, reference, A::DATA_TYPE)?;

    // Need the transpose for fortran ordering used in nifti file format.
    let data = data.t();

    let header_file = File::create(header_path)?;
    if header.vox_offset > 0.0 {
        if is_gz {
            let mut writer = ByteOrdered::runtime(
                GzEncoder::new(header_file, compression_level),
                header.endianness,
            );
            write_header(writer.as_mut(), &header)?;
            write_data(writer.as_mut(), &header, data)?;
            let _ = writer.into_inner().finish()?;
        } else {
            let mut writer = ByteOrdered::runtime(BufWriter::new(header_file), header.endianness);
            write_header(writer.as_mut(), &header)?;
            write_data(writer, &header, data)?;
        }
    } else {
        let data_file = File::create(&data_path)?;
        if is_gz {
            let mut writer = ByteOrdered::runtime(
                GzEncoder::new(header_file, compression_level),
                header.endianness,
            );
            write_header(writer.as_mut(), &header)?;
            let _ = writer.into_inner().finish()?;

            let mut writer = ByteOrdered::runtime(
                GzEncoder::new(data_file, compression_level),
                header.endianness,
            );
            write_data(writer.as_mut(), &header, data)?;
            let _ = writer.into_inner().finish()?;
        } else {
            let header_writer =
                ByteOrdered::runtime(BufWriter::new(header_file), header.endianness);
            write_header(header_writer, &header)?;
            let data_writer =
                ByteOrdered::runtime(BufWriter::new(data_file), header.endianness);
            write_data(data_writer, &header, data)?;
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
    header_path: P,
    data: &ArrayBase<S, D>,
    reference: Option<&NiftiHeader>,
) -> Result<()>
where
    P: AsRef<Path>,
    S: Data<Elem = [u8; 3]>,
    D: Dimension + RemoveAxis,
{
    let compression_level = Compression::fast();
    let is_gz = is_gz_file(&header_path);
    let (mut header, data_path) =
        prepare_header_and_paths(&header_path, data, reference, NiftiType::Rgb24)?;

    // The `scl_slope` and `scl_inter` fields are ignored on the Rgb24 type.
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;

    // Need the transpose for fortran used in nifti file format.
    let data = data.t();

    let header_file = File::create(header_path)?;
    if header.vox_offset > 0.0 {
        if is_gz {
            let mut writer = ByteOrdered::runtime(
                GzEncoder::new(header_file, compression_level),
                header.endianness,
            );
            write_header(writer.as_mut(), &header)?;
            write_slices::<_, u8, _, _, _, _>(writer.as_mut(), data)?;
            let _ = writer.into_inner().finish()?;
        } else {
            let mut writer = ByteOrdered::runtime(BufWriter::new(header_file), header.endianness);
            write_header(writer.as_mut(), &header)?;
            write_slices::<_, u8, _, _, _, _>(writer, data)?;
        }
    } else {
        let data_file = File::create(&data_path)?;
        if is_gz {
            let mut writer = ByteOrdered::runtime(
                GzEncoder::new(header_file, compression_level),
                header.endianness,
            );
            write_header(writer.as_mut(), &header)?;
            let _ = writer.into_inner().finish()?;

            let mut writer = ByteOrdered::runtime(
                GzEncoder::new(data_file, compression_level),
                header.endianness,
            );
            write_slices::<_, u8, _, _, _, _>(writer.as_mut(), data)?;
            let _ = writer.into_inner().finish()?;
        } else {
            let header_writer =
                ByteOrdered::runtime(BufWriter::new(header_file), header.endianness);
            write_header(header_writer, &header)?;

            let data_writer =
                ByteOrdered::runtime(BufWriter::new(data_file), header.endianness);
            write_slices::<_, u8, _, _, _, _>(data_writer, data)?;
        }
    }

    Ok(())
}

fn prepare_header_and_paths<P, T, D>(
    header_path: P,
    data: &ArrayBase<T, D>,
    reference: Option<&NiftiHeader>,
    datatype: NiftiType,
) -> Result<(NiftiHeader, PathBuf)>
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
            header.sform_code = 2;
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

    // The only acceptable length is 80. If different, try to set it.
    header.validate_description()?;

    let mut path_buf = PathBuf::from(header_path.as_ref());
    let data_path = if is_hdr_file(&header_path) {
        header.vox_offset = 0.0;
        header.magic = *MAGIC_CODE_NI1;
        let data_path = if is_gz_file(&header_path) {
            let _ = path_buf.set_extension("");
            path_buf.with_extension("img.gz")
        } else {
            path_buf.with_extension("img")
        };
        data_path
    } else {
        path_buf
    };

    Ok((header, data_path))
}

fn write_header<W, E>(mut writer: ByteOrdered<W, E>, header: &NiftiHeader) -> Result<()>
where
    W: Write,
    E: Endian,
{
    writer.write_i32(header.sizeof_hdr)?;
    writer.write_all(&header.data_type)?;
    writer.write_all(&header.db_name)?;
    writer.write_i32(header.extents)?;
    writer.write_i16(header.session_error)?;
    writer.write_u8(header.regular)?;
    writer.write_u8(header.dim_info)?;
    for s in &header.dim {
        writer.write_u16(*s)?;
    }
    writer.write_f32(header.intent_p1)?;
    writer.write_f32(header.intent_p2)?;
    writer.write_f32(header.intent_p3)?;
    writer.write_i16(header.intent_code)?;
    writer.write_i16(header.datatype)?;
    writer.write_i16(header.bitpix)?;
    writer.write_i16(header.slice_start)?;
    for f in &header.pixdim {
        writer.write_f32(*f)?;
    }
    writer.write_f32(header.vox_offset)?;
    writer.write_f32(header.scl_slope)?;
    writer.write_f32(header.scl_inter)?;
    writer.write_i16(header.slice_end)?;
    writer.write_u8(header.slice_code)?;
    writer.write_u8(header.xyzt_units)?;
    writer.write_f32(header.cal_max)?;
    writer.write_f32(header.cal_min)?;
    writer.write_f32(header.slice_duration)?;
    writer.write_f32(header.toffset)?;
    writer.write_i32(header.glmax)?;
    writer.write_i32(header.glmin)?;

    writer.write_all(&header.descrip)?;
    writer.write_all(&header.aux_file)?;
    writer.write_i16(header.qform_code)?;
    writer.write_i16(header.sform_code)?;
    for f in &[
        header.quatern_b,
        header.quatern_c,
        header.quatern_d,
        header.quatern_x,
        header.quatern_y,
        header.quatern_z,
    ] {
        writer.write_f32(*f)?;
    }
    for f in header
        .srow_x
        .iter()
        .chain(&header.srow_y)
        .chain(&header.srow_z)
    {
        writer.write_f32(*f)?;
    }
    writer.write_all(&header.intent_name)?;
    writer.write_all(&header.magic)?;

    // Empty 4 bytes after the header
    // TODO Support writing extension data.
    writer.write_u32(0)?;

    Ok(())
}

/// Write the data in 'f' order.
///
/// Like NiBabel, we iterate by "slice" to improve speed and use less memory.
fn write_data<T, D, W, E>(mut writer: ByteOrdered<W, E>, header: &NiftiHeader, data: ArrayView<T, D>) -> Result<()>
where
    T: Clone + PodTransmutable,
    T: Div<Output = T>,
    T: FromPrimitive,
    T: PodTransmutable,
    T: ScalarOperand,
    T: Sub<Output = T>,
    D: Dimension + RemoveAxis,
    W: Write,
    E: Endian + Copy,
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
            write_slice::<T, T, _, _, _, _>(writer.as_mut(), arr_data.sub(inter).div(slope))?;
        }
    } else {
        write_slices::<T, T, _, _, _, _>(writer, data)?;
    }
    Ok(())
}

fn write_slices<A, B, S, D, W, E>(mut writer: ByteOrdered<W, E>, data: ArrayBase<S, D>) -> Result<()>
where
    S: Data<Elem = A>,
    A: Clone + PodTransmutable,
    D: Dimension + RemoveAxis,
    W: Write,
    E: Endian + Copy,
{
    let mut iter = data.axis_iter(Axis(0));
    if let Some(arr_data) = iter.next() {
        // Keep slice voxels in a separate array to ensure `C` ordering even after `into_shape`.
        let mut slice = arr_data.to_owned();
        write_slice::<_, B, _, _, _, _>(writer.as_mut(), slice.view())?;
        for arr_data in iter {
            slice.assign(&arr_data);
            write_slice::<_, B, _, _, _, _>(writer.as_mut(), slice.view())?;
        }
    }
    Ok(())
}

fn write_slice<A, B, S, D, W, E>(writer: ByteOrdered<&mut W, E>, data: ArrayBase<S, D>) -> Result<()>
where
    S: Data<Elem = A>,
    A: Clone + PodTransmutable,
    D: Dimension,
    W: Write,
    E: Endian,
{
    let len = data.len();
    let arr_data = data.into_shape(len).unwrap();
    let slice = arr_data.as_slice().unwrap();
    let bytes = guarded_transmute_to_bytes_pod_many(slice);
    let (writer, endianness) = writer.into_parts();
    let bytes = adapt_bytes::<B, _>(&bytes, endianness);
    writer.write_all(&*bytes)?;
    Ok(())
}
