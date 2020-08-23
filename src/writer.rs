//! Utility functions to write nifti images.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use byteordered::{ByteOrdered, Endian};
use flate2::write::GzEncoder;
use flate2::Compression;
use ndarray::{ArrayBase, Axis, Data, Dimension, RemoveAxis};
use safe_transmute::{transmute_to_bytes, TriviallyTransmutable};

use crate::{
    header::{MAGIC_CODE_NI1, MAGIC_CODE_NIP1},
    util::{adapt_bytes, is_gz_file, is_hdr_file},
    volume::shape::Dim,
    DataElement, NiftiHeader, NiftiType, Result,
};

/// Options and flags which can be used to configure how a NIfTI image is written.
#[derive(Debug, Clone, PartialEq)]
pub struct WriterOptions<'a> {
    /// Where to write the output image (header and/or data).
    header_path: PathBuf,
    /// If given, it will be used to fill most of the header's fields. Otherwise, a default
    /// `NiftiHeader` will be built and written. In all cases, the `dim`, `datatype` and `bitpix`
    /// fields will depend only on `data` parameter passed to the `write` function, not on this
    /// field. This means that the `datatype` defined in `reference` will be ignored. Because of
    /// this, `scl_slope` will be set to 1.0 and `scl_inter` to 0.0.
    reference: Option<&'a NiftiHeader>,
    /// Whether to write the header and data in distinct files. (nii vs hdr+img)
    header_file: bool,
    /// Whether to compress the output to gz. Is guessed from the `header_path`, but can be
    /// overriden with the `compress` method, which will update the `header_path` to keep coherency.
    compress: bool,
    /// Compression level to use when writing to the gz file.
    compression_level: Compression,
}

impl<'a> WriterOptions<'a> {
    /// Creates a blank new set of options ready for configuration.
    pub fn new<P>(header_path: P) -> WriterOptions<'a>
    where
        P: AsRef<Path>,
    {
        let mut header_path = header_path.as_ref().to_owned();
        if header_path.extension().is_none() {
            let _ = header_path.set_extension("nii");
        }
        let header_file = is_hdr_file(&header_path);
        let compress = is_gz_file(&header_path);
        WriterOptions {
            header_path,
            reference: None,
            header_file,
            compress,
            compression_level: Compression::fast(),
        }
    }

    /// Sets the reference header to use instead of the default one.
    pub fn reference(mut self, reference: &'a NiftiHeader) -> Self {
        self.reference = Some(reference);
        self
    }

    /// Whether to write the header and data in distinct files.
    ///
    /// Will update the output path accordingly.
    pub fn header_file(mut self, header_file: bool) -> Self {
        if self.header_file != header_file {
            self.header_file = header_file;
            self.fix_header_path_extension();
        }
        self
    }

    /// Whether to compress the output to gz.
    ///
    /// Will update the output path accordingly.
    pub fn compress(mut self, compress: bool) -> Self {
        if self.compress != compress {
            self.compress = compress;
            self.fix_header_path_extension();
        }
        self
    }

    /// Fix the header path extension in case a change in `header_file` or `compress` broke it.
    fn fix_header_path_extension(&mut self) {
        let _ = self.header_path.set_extension("");
        self.header_path = match (self.header_file, self.compress) {
            (false, false) => self.header_path.with_extension("nii"),
            (false, true) => self.header_path.with_extension("nii.gz"),
            (true, false) => self.header_path.with_extension("hdr"),
            (true, true) => self.header_path.with_extension("hdr.gz"),
        };
    }

    /// Sets the compression level to use when compressing the output.
    pub fn compression_level(mut self, compression_level: Compression) -> Self {
        self.compression_level = compression_level;
        self
    }

    /// Write a nifti file (.nii or .nii.gz).
    pub fn write_nifti<A, S, D>(&self, data: &ArrayBase<S, D>) -> Result<()>
    where
        S: Data<Elem = A>,
        A: DataElement,
        A: TriviallyTransmutable,
        D: Dimension + RemoveAxis,
    {
        let (header, data_path) = self.prepare_header_and_paths(data, A::DATA_TYPE)?;

        // Need the transpose for fortran ordering used in nifti file format.
        let data = data.t();

        let header_file = File::create(&self.header_path)?;
        if header.vox_offset > 0.0 {
            if self.compress {
                let mut writer = ByteOrdered::runtime(
                    GzEncoder::new(header_file, self.compression_level),
                    header.endianness,
                );
                write_header(writer.as_mut(), &header)?;
                write_data::<_, A, _, _, _, _>(writer.as_mut(), data)?;
                let _ = writer.into_inner().finish()?;
            } else {
                let mut writer =
                    ByteOrdered::runtime(BufWriter::new(header_file), header.endianness);
                write_header(writer.as_mut(), &header)?;
                write_data::<_, A, _, _, _, _>(writer, data)?;
            }
        } else {
            let data_file = File::create(&data_path)?;
            if self.compress {
                let mut writer = ByteOrdered::runtime(
                    GzEncoder::new(header_file, self.compression_level),
                    header.endianness,
                );
                write_header(writer.as_mut(), &header)?;
                let _ = writer.into_inner().finish()?;

                let mut writer = ByteOrdered::runtime(
                    GzEncoder::new(data_file, self.compression_level),
                    header.endianness,
                );
                write_data::<_, A, _, _, _, _>(writer.as_mut(), data)?;
                let _ = writer.into_inner().finish()?;
            } else {
                let header_writer =
                    ByteOrdered::runtime(BufWriter::new(header_file), header.endianness);
                write_header(header_writer, &header)?;
                let data_writer =
                    ByteOrdered::runtime(BufWriter::new(data_file), header.endianness);
                write_data::<_, A, _, _, _, _>(data_writer, data)?;
            }
        }

        Ok(())
    }

    /// Write a RGB nifti file (.nii or .nii.gz).
    pub fn write_rgb_nifti<S, D>(&self, data: &ArrayBase<S, D>) -> Result<()>
    where
        S: Data<Elem = [u8; 3]>,
        D: Dimension + RemoveAxis,
    {
        let (header, data_path) = self.prepare_header_and_paths(data, NiftiType::Rgb24)?;

        // Need the transpose for fortran used in nifti file format.
        let data = data.t();

        let header_file = File::create(&self.header_path)?;
        if header.vox_offset > 0.0 {
            if self.compress {
                let mut writer = ByteOrdered::runtime(
                    GzEncoder::new(header_file, self.compression_level),
                    header.endianness,
                );
                write_header(writer.as_mut(), &header)?;
                write_data::<_, u8, _, _, _, _>(writer.as_mut(), data)?;
                let _ = writer.into_inner().finish()?;
            } else {
                let mut writer =
                    ByteOrdered::runtime(BufWriter::new(header_file), header.endianness);
                write_header(writer.as_mut(), &header)?;
                write_data::<_, u8, _, _, _, _>(writer, data)?;
            }
        } else {
            let data_file = File::create(&data_path)?;
            if self.compress {
                let mut writer = ByteOrdered::runtime(
                    GzEncoder::new(header_file, self.compression_level),
                    header.endianness,
                );
                write_header(writer.as_mut(), &header)?;
                let _ = writer.into_inner().finish()?;

                let mut writer = ByteOrdered::runtime(
                    GzEncoder::new(data_file, self.compression_level),
                    header.endianness,
                );
                write_data::<_, u8, _, _, _, _>(writer.as_mut(), data)?;
                let _ = writer.into_inner().finish()?;
            } else {
                let header_writer =
                    ByteOrdered::runtime(BufWriter::new(header_file), header.endianness);
                write_header(header_writer, &header)?;

                let data_writer =
                    ByteOrdered::runtime(BufWriter::new(data_file), header.endianness);
                write_data::<_, u8, _, _, _, _>(data_writer, data)?;
            }
        }

        Ok(())
    }

    fn prepare_header_and_paths<T, D>(
        &self,
        data: &ArrayBase<T, D>,
        datatype: NiftiType,
    ) -> Result<(NiftiHeader, PathBuf)>
    where
        T: Data,
        D: Dimension,
    {
        // If no reference header is given, use the default.
        let reference = match self.reference {
            Some(r) => r.clone(),
            None => {
                let mut header = NiftiHeader::default();
                header.sform_code = 2;
                header
            }
        };

        let mut header = NiftiHeader {
            dim: *Dim::from_slice(data.shape())?.raw(),
            sizeof_hdr: 348,
            datatype: datatype as i16,
            bitpix: (datatype.size_of() * 8) as i16,
            vox_offset: 352.0,
            scl_inter: 0.0,
            scl_slope: 1.0,
            magic: *MAGIC_CODE_NIP1,
            // All other fields are copied from reference header
            ..reference
        };

        // The only acceptable length is 80. If different, try to set it.
        header.validate_description()?;

        let mut path_buf = self.header_path.clone();
        let data_path = if self.header_file {
            header.vox_offset = 0.0;
            header.magic = *MAGIC_CODE_NI1;
            let data_path = if self.compress {
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
fn write_data<A, B, S, D, W, E>(mut writer: ByteOrdered<W, E>, data: ArrayBase<S, D>) -> Result<()>
where
    S: Data<Elem = A>,
    A: TriviallyTransmutable,
    D: Dimension + RemoveAxis,
    W: Write,
    E: Endian + Copy,
{
    // We always write the image by iterating on the first axis, thus
    //   3D (depth, height, width) => per axial slice
    //   4D (time, depth, height, width) => per volume
    // However, there's a problem when dim == (1, ...). We must iterate by the first non-one length
    // axis, otherwise the image will be written in the wrong ordering. See issue #63.
    let first_good_axis = data.shape().iter().position(|&d| d > 1).unwrap_or(0);
    let mut iter = data.axis_iter(Axis(first_good_axis));
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

fn write_slice<A, B, S, D, W, E>(
    writer: ByteOrdered<&mut W, E>,
    data: ArrayBase<S, D>,
) -> Result<()>
where
    S: Data<Elem = A>,
    A: Clone + TriviallyTransmutable,
    D: Dimension,
    W: Write,
    E: Endian,
{
    let len = data.len();
    let arr_data = data.into_shape(len).unwrap();
    let slice = arr_data.as_slice().unwrap();
    let bytes = transmute_to_bytes(slice);
    let (writer, endianness) = writer.into_parts();
    let bytes = adapt_bytes::<B, _>(&bytes, endianness);
    writer.write_all(&*bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_writer_path() {
        let w = WriterOptions::new("~/image");
        assert_eq!(w.header_path, PathBuf::from("~/image.nii"));

        for (p, nii, niigz) in &[
            ("~/image", "~/image.nii", "~/image.nii.gz"),
            ("~/image.nii", "~/image.nii", "~/image.nii.gz"),
            ("~/image.nii.gz", "~/image.nii", "~/image.nii.gz"),
            ("~/image.hdr", "~/image.hdr", "~/image.hdr.gz"),
            ("~/image.hdr.gz", "~/image.hdr", "~/image.hdr.gz"),
        ] {
            let nii = PathBuf::from(nii);
            let niigz = PathBuf::from(niigz);

            let w = WriterOptions::new(p).compress(true);
            assert_eq!(w.header_path, niigz);
            let w = WriterOptions::new(p).compress(true).compress(true);
            assert_eq!(w.header_path, niigz);
            let w = WriterOptions::new(p).compress(true).compress(false);
            assert_eq!(w.header_path, nii);

            let w = WriterOptions::new(p).compress(false);
            assert_eq!(w.header_path, nii);
            let w = WriterOptions::new(p).compress(false).compress(false);
            assert_eq!(w.header_path, nii);
            let w = WriterOptions::new(p).compress(false).compress(true);
            assert_eq!(w.header_path, niigz);
        }
    }
}
