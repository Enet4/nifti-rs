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
    DataElement, ExtensionSequence, NiftiHeader, NiftiType, Result,
};

#[derive(Debug, Clone, PartialEq)]
enum HeaderReference<'a> {
    None,
    FromHeader(&'a NiftiHeader),
    FromFile(&'a Path),
}

impl<'a> HeaderReference<'a> {
    fn to_header(&self) -> Result<NiftiHeader> {
        match self {
            HeaderReference::FromHeader(h) => Ok((*h).to_owned()),
            HeaderReference::FromFile(path) => NiftiHeader::from_file(path),
            HeaderReference::None => Ok(NiftiHeader {
                sform_code: 2,
                ..NiftiHeader::default()
            }),
        }
    }
}

/// Options and flags which can be used to configure how a NIfTI image is written.
#[derive(Debug, Clone, PartialEq)]
pub struct WriterOptions<'a> {
    /// Where to write the output image (header and/or data).
    path: PathBuf,
    /// If given, it will be used to fill most of the header's fields. Otherwise, a default
    /// `NiftiHeader` will be built and written. In all cases, the `dim`, `datatype` and `bitpix`
    /// fields will depend only on `data` parameter passed to the `write` function, not on this
    /// field. This means that the `datatype` defined in `header_reference` will be ignored.
    /// Because of this, `scl_slope` will be set to 1.0 and `scl_inter` to 0.0.
    header_reference: HeaderReference<'a>,
    /// Whether to write the NIfTI file pair. (nii vs hdr+img)
    write_header_file: bool,
    /// The volume will be compressed if `path` ends with ".gz", but it can be overriden with the
    /// `compress` method. If enabled, the volume will be compressed using the specified compression
    /// level. Default to `Compression::fast()`.
    compression: Option<Compression>,
    /// The header file will only be compressed if the caller specifically asked for a path ending
    /// with "hdr.gz". Otherwise, only the volume will be compressed (if requested).
    force_header_compression: bool,

    /// Optional ExtensionSequence
    extension_sequence: Option<ExtensionSequence>,
}

impl<'a> WriterOptions<'a> {
    /// Creates a blank new set of options ready for configuration.
    pub fn new<P>(path: P) -> WriterOptions<'a>
    where
        P: AsRef<Path>,
    {
        let mut path = path.as_ref().to_owned();
        if path.extension().is_none() {
            let _ = path.set_extension("nii");
        }
        let write_header_file = is_hdr_file(&path);
        let compression = if is_gz_file(&path) {
            Some(Compression::fast())
        } else {
            None
        };
        WriterOptions {
            path,
            header_reference: HeaderReference::None,
            write_header_file,
            compression,
            force_header_compression: write_header_file && compression.is_some(),
            extension_sequence: None,
        }
    }

    /// Sets the reference header.
    pub fn reference_header(mut self, header: &'a NiftiHeader) -> Self {
        self.header_reference = HeaderReference::FromHeader(header);
        self
    }

    /// Loads a reference header from a Nifti file.
    pub fn reference_file<P: 'a>(mut self, path: &'a P) -> Self
    where
        P: AsRef<Path>,
    {
        self.header_reference = HeaderReference::FromFile(path.as_ref());
        self
    }

    /// Whether to write the header and data in distinct files.
    ///
    /// Will update the output path accordingly.
    pub fn write_header_file(mut self, write_header_file: bool) -> Self {
        if self.write_header_file != write_header_file {
            self.write_header_file = write_header_file;
        }
        self
    }

    /// Whether to compress the output to gz.
    ///
    /// Will update the output path accordingly.
    pub fn compress(mut self, compress: bool) -> Self {
        if self.compression.is_none() && compress {
            self.compression = Some(Compression::fast());
        } else if self.compression.is_some() && !compress {
            self.compression = None;
        }
        self
    }

    /// Sets the compression level to use when compressing the output.
    pub fn compression_level(mut self, compression_level: Compression) -> Self {
        self.compression = Some(compression_level);
        self
    }

    /// Sets an extension sequence for the writer
    pub fn with_extensions(mut self, extension_sequence: ExtensionSequence) -> Self {
        self.extension_sequence = Some(extension_sequence);
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
        let header = self.prepare_header(data, A::DATA_TYPE)?;
        let (header_path, data_path) = self.output_paths();

        // Need the transpose for fortran ordering used in nifti file format.
        let data = data.t();

        let header_file = File::create(header_path)?;
        if header.vox_offset > 0.0 {
            if let Some(compression_level) = self.compression {
                let mut writer = ByteOrdered::runtime(
                    GzEncoder::new(header_file, compression_level),
                    header.endianness,
                );
                write_header(writer.as_mut(), &header)?;
                write_extensions(writer.as_mut(), self.extension_sequence.as_ref())?;
                write_data::<_, A, _, _, _, _>(writer.as_mut(), data)?;
                let _ = writer.into_inner().finish()?;
            } else {
                let mut writer =
                    ByteOrdered::runtime(BufWriter::new(header_file), header.endianness);
                write_header(writer.as_mut(), &header)?;
                write_extensions(writer.as_mut(), self.extension_sequence.as_ref())?;
                write_data::<_, A, _, _, _, _>(writer, data)?;
            }
        } else {
            let data_file = File::create(&data_path)?;
            if let Some(compression_level) = self.compression {
                let mut writer = ByteOrdered::runtime(
                    GzEncoder::new(header_file, compression_level),
                    header.endianness,
                );
                write_header(writer.as_mut(), &header)?;
                write_extensions(writer.as_mut(), self.extension_sequence.as_ref())?;
                let _ = writer.into_inner().finish()?;

                let mut writer = ByteOrdered::runtime(
                    GzEncoder::new(data_file, compression_level),
                    header.endianness,
                );
                write_data::<_, A, _, _, _, _>(writer.as_mut(), data)?;
                let _ = writer.into_inner().finish()?;
            } else {
                let mut header_writer =
                    ByteOrdered::runtime(BufWriter::new(header_file), header.endianness);
                write_header(header_writer.as_mut(), &header)?;
                write_extensions(header_writer.as_mut(), self.extension_sequence.as_ref())?;
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
        let header = self.prepare_header(data, NiftiType::Rgb24)?;
        let (header_path, data_path) = self.output_paths();

        // Need the transpose for fortran used in nifti file format.
        let data = data.t();

        let header_file = File::create(header_path)?;
        if header.vox_offset > 0.0 {
            if let Some(compression_level) = self.compression {
                let mut writer = ByteOrdered::runtime(
                    GzEncoder::new(header_file, compression_level),
                    header.endianness,
                );
                write_header(writer.as_mut(), &header)?;
                write_extensions(writer.as_mut(), self.extension_sequence.as_ref())?;
                write_data::<_, u8, _, _, _, _>(writer.as_mut(), data)?;
                let _ = writer.into_inner().finish()?;
            } else {
                let mut writer =
                    ByteOrdered::runtime(BufWriter::new(header_file), header.endianness);
                write_header(writer.as_mut(), &header)?;
                write_extensions(writer.as_mut(), self.extension_sequence.as_ref())?;
                write_data::<_, u8, _, _, _, _>(writer, data)?;
            }
        } else {
            let data_file = File::create(&data_path)?;
            if let Some(compression_level) = self.compression {
                let mut writer = ByteOrdered::runtime(
                    GzEncoder::new(header_file, compression_level),
                    header.endianness,
                );
                write_header(writer.as_mut(), &header)?;
                write_extensions(writer.as_mut(), self.extension_sequence.as_ref())?;
                let _ = writer.into_inner().finish()?;

                let mut writer = ByteOrdered::runtime(
                    GzEncoder::new(data_file, compression_level),
                    header.endianness,
                );
                write_data::<_, u8, _, _, _, _>(writer.as_mut(), data)?;
                let _ = writer.into_inner().finish()?;
            } else {
                let mut header_writer =
                    ByteOrdered::runtime(BufWriter::new(header_file), header.endianness);
                write_header(header_writer.as_mut(), &header)?;
                write_extensions(header_writer.as_mut(), self.extension_sequence.as_ref())?;
                let data_writer =
                    ByteOrdered::runtime(BufWriter::new(data_file), header.endianness);
                write_data::<_, u8, _, _, _, _>(data_writer, data)?;
            }
        }

        Ok(())
    }

    fn prepare_header<T, D>(
        &self,
        data: &ArrayBase<T, D>,
        datatype: NiftiType,
    ) -> Result<NiftiHeader>
    where
        T: Data,
        D: Dimension,
    {
        let mut vox_offset: f32 = 352.0;

        if let Some(extension_sequence) = self.extension_sequence.as_ref() {
            vox_offset += extension_sequence.bytes_on_disk() as f32;
        }

        let mut header = NiftiHeader {
            dim: *Dim::from_slice(data.shape())?.raw(),
            sizeof_hdr: 348,
            datatype: datatype as i16,
            bitpix: (datatype.size_of() * 8) as i16,
            vox_offset,
            scl_inter: 0.0,
            scl_slope: 1.0,
            magic: *MAGIC_CODE_NIP1,
            // All other fields are copied from the requested reference header
            ..self.header_reference.to_header()?
        };

        if self.write_header_file {
            header.vox_offset = 0.0;
            header.magic = *MAGIC_CODE_NI1;
        }

        // The only acceptable length is 80. If different, try to set it.
        header.validate_description()?;

        Ok(header)
    }

    /// Fix the header path extension in case a change in `write_header_file` or `compression`
    /// broke it.
    fn output_paths(&self) -> (PathBuf, PathBuf) {
        let mut path = self.path.clone();
        let _ = path.set_extension("");
        match (self.write_header_file, self.compression.is_some()) {
            (false, false) => (path.with_extension("nii"), path.with_extension("nii")),
            (false, true) => (path.with_extension("nii.gz"), path.with_extension("nii.gz")),
            (true, false) => (path.with_extension("hdr"), path.with_extension("img")),
            (true, true) => {
                if self.force_header_compression {
                    (path.with_extension("hdr.gz"), path.with_extension("img.gz"))
                } else {
                    (path.with_extension("hdr"), path.with_extension("img.gz"))
                }
            }
        }
    }
}

fn write_extensions<W, E>(
    mut writer: ByteOrdered<W, E>,
    extensions: Option<&ExtensionSequence>,
) -> Result<()>
where
    W: Write,
    E: Endian,
{
    let extensions = match extensions {
        Some(extensions) => extensions,
        None => {
            writer.write_u32(0)?;
            return Ok(());
        }
    };

    if extensions.is_empty() {
        // Write an extender code of 4 zeros, which for NIFTI means that there are no extensions
        writer.write_u32(0)?;
        return Ok(());
    }

    writer.write_all(extensions.extender().as_bytes())?;
    for extension in extensions.iter() {
        writer.write_i32(extension.size())?;
        writer.write_i32(extension.code())?;
        writer.write_all(extension.data())?;
    }
    Ok(())
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
