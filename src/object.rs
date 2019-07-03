//! Module for handling and retrieving complete NIFTI-1 objects.

use byteordered::ByteOrdered;
use error::NiftiError;
use error::Result;
use extension::{Extender, ExtensionSequence};
use flate2::bufread::GzDecoder;
use header::NiftiHeader;
use header::MAGIC_CODE_NI1;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;
use util::{into_img_file_gz, is_gz_file, open_file_maybe_gz};
use volume::inmem::InMemNiftiVolume;
use volume::streamed::StreamedNiftiVolume;
use volume::{FromSource, FromSourceOptions, NiftiVolume};

/// Trait type for all possible implementations of
/// owning NIFTI-1 objects. Objects contain a NIFTI header,
/// a volume, and a possibly empty extension sequence.
pub trait NiftiObject {
    /// The concrete type of the volume.
    type Volume: NiftiVolume;

    /// Obtain a reference to the NIFTI header.
    fn header(&self) -> &NiftiHeader;

    /// Obtain a mutable reference to the NIFTI header.
    fn header_mut(&mut self) -> &mut NiftiHeader;

    /// Obtain a reference to the object's extensions.
    fn extensions(&self) -> &ExtensionSequence;

    /// Obtain a reference to the object's volume.
    fn volume(&self) -> &Self::Volume;

    /// Move the volume out of the object, discarding the
    /// header and extensions.
    fn into_volume(self) -> Self::Volume;
}

/// Generic data type for a NIfTI object.
#[derive(Debug, PartialEq, Clone)]
pub struct GenericNiftiObject<V> {
    header: NiftiHeader,
    extensions: ExtensionSequence,
    volume: V,
}

impl<V> NiftiObject for GenericNiftiObject<V>
where
    V: NiftiVolume,
{
    type Volume = V;

    fn header(&self) -> &NiftiHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut NiftiHeader {
        &mut self.header
    }

    fn extensions(&self) -> &ExtensionSequence {
        &self.extensions
    }

    fn volume(&self) -> &Self::Volume {
        &self.volume
    }

    fn into_volume(self) -> Self::Volume {
        self.volume
    }
}

/// A NIfTI object containing an in-memory volume.
pub type InMemNiftiObject = GenericNiftiObject<InMemNiftiVolume>;

impl InMemNiftiObject {
    /// Retrieve the full contents of a NIFTI object.
    /// The given file system path is used as reference.
    /// If the file only contains the header, this method will
    /// look for the corresponding file with the extension ".img",
    /// or ".img.gz" if the former wasn't found.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use nifti::{NiftiObject, InMemNiftiObject};
    ///
    /// let obj = InMemNiftiObject::from_file("minimal.nii.gz")?;
    /// # Ok::<(), nifti::NiftiError>(())
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let gz = is_gz_file(&path);

        let file = BufReader::new(File::open(&path)?);
        if gz {
            Self::from_file_2(path, GzDecoder::new(file), Default::default())
        } else {
            Self::from_file_2(path, file, Default::default())
        }
    }

    /// Retrieve a NIFTI object as separate header and volume files.
    /// This method is useful when file names are not conventional for a
    /// NIFTI file pair.
    pub fn from_file_pair<P, Q>(hdr_path: P, vol_path: Q) -> Result<Self>
    where
        P: AsRef<Path>,
        Q: AsRef<Path>,
    {
        let gz = is_gz_file(&hdr_path);

        let file = BufReader::new(File::open(&hdr_path)?);
        if gz {
            Self::from_file_pair_2(GzDecoder::new(file), vol_path, Default::default())
        } else {
            Self::from_file_pair_2(file, vol_path, Default::default())
        }
    }

    /// Retrieve an in-memory NIfTI object from a stream of data.
    #[deprecated(note = "use `from_reader` instead")]
    pub fn new_from_stream<R: Read>(&self, source: R) -> Result<Self> {
        Self::from_reader(source)
    }
}

/// A NIfTI object containing a [streamed volume].
///
/// [streamed volume]: ../volume/streamed/index.html
pub type StreamedNiftiObject<R> = GenericNiftiObject<StreamedNiftiVolume<R>>;

impl StreamedNiftiObject<Box<dyn Read>> {
    /// Retrieve the NIfTI object and prepare the volume for streamed reading.
    /// The given file system path is used as reference.
    /// If the file only contains the header, this method will
    /// look for the corresponding file with the extension ".img",
    /// or ".img.gz" if the former wasn't found.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use nifti::{NiftiObject, StreamedNiftiObject};
    ///
    /// let obj = StreamedNiftiObject::from_file("minimal.nii.gz")?;
    ///
    /// let volume = obj.into_volume();
    /// for slice in volume {
    ///     let slice = slice?;
    ///     // manipulate slice here
    /// }
    /// # Ok::<(), nifti::NiftiError>(())
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let gz = is_gz_file(&path);

        let file = BufReader::new(File::open(&path)?);
        if gz {
            Self::from_file_2_erased(path, GzDecoder::new(file), None)
        } else {
            Self::from_file_2_erased(path, file, None)
        }
    }

    /// Retrieve the NIfTI object and prepare the volume for streamed reading,
    /// using `slice_rank` as the dimensionality of each slice.
    /// The given file system path is used as reference.
    /// If the file only contains the header, this method will
    /// look for the corresponding file with the extension ".img",
    /// or ".img.gz" if the former wasn't found.
    pub fn from_file_rank<P: AsRef<Path>>(path: P, slice_rank: u16) -> Result<Self> {
        let gz = is_gz_file(&path);

        let file = BufReader::new(File::open(&path)?);
        if gz {
            Self::from_file_2_erased(path, GzDecoder::new(file), Some(slice_rank))
        } else {
            Self::from_file_2_erased(path, file, Some(slice_rank))
        }
    }

    /// Retrieve a NIfTI object as separate header and volume files, for
    /// streamed volume reading. This method is useful when file names are not
    /// conventional for a NIfTI file pair.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use nifti::{NiftiObject, StreamedNiftiObject};
    ///
    /// let obj = StreamedNiftiObject::from_file_pair("abc.hdr", "abc.img.gz")?;
    ///
    /// let volume = obj.into_volume();
    /// for slice in volume {
    ///     let slice = slice?;
    ///     // manipulate slice here
    /// }
    /// # Ok::<(), nifti::NiftiError>(())
    /// ```
    pub fn from_file_pair<P, Q>(hdr_path: P, vol_path: Q) -> Result<Self>
    where
        P: AsRef<Path>,
        Q: AsRef<Path>,
    {
        let gz = is_gz_file(&hdr_path);

        let file = BufReader::new(File::open(&hdr_path)?);
        if gz {
            Self::from_file_pair_2_erased(GzDecoder::new(file), vol_path, Default::default())
        } else {
            Self::from_file_pair_2_erased(file, vol_path, Default::default())
        }
    }

    /// Retrieve a NIfTI object as separate header and volume files, for
    /// streamed volume reading, using `slice_rank` as the dimensionality of
    /// each slice. This method is useful when file names are not conventional
    /// for a NIfTI file pair.
    pub fn from_file_pair_rank<P, Q>(hdr_path: P, vol_path: Q, slice_rank: u16) -> Result<Self>
    where
        P: AsRef<Path>,
        Q: AsRef<Path>,
    {
        let gz = is_gz_file(&hdr_path);

        let file = BufReader::new(File::open(&hdr_path)?);
        if gz {
            Self::from_file_pair_2_erased::<_, _>(GzDecoder::new(file), vol_path, Some(slice_rank))
        } else {
            Self::from_file_pair_2_erased::<_, _>(file, vol_path, Some(slice_rank))
        }
    }
}

impl<V> GenericNiftiObject<V> {
    /// Construct a NIfTI object from a data reader, first by fetching the
    /// header, the extensions, and then the volume.
    ///
    /// # Errors
    ///
    /// - `NiftiError::NoVolumeData` if the source only contains (or claims to contain)
    /// a header.
    pub fn from_reader<R>(mut source: R) -> Result<Self>
    where
        R: Read,
        V: FromSource<R>,
    {
        let header = NiftiHeader::from_reader(&mut source)?;
        if &header.magic == MAGIC_CODE_NI1 {
            return Err(NiftiError::NoVolumeData);
        }
        let extender = Extender::from_reader(&mut source)?;

        let (volume, extensions) = GenericNiftiObject::from_reader_with_extensions(
            source,
            &header,
            extender,
            Default::default(),
        )?;

        Ok(GenericNiftiObject {
            header,
            extensions,
            volume,
        })
    }

    /// Read a NIFTI volume, and extensions, from a data reader. The header,
    /// extender code and expected byte order of the volume's data must be
    /// known in advance.
    fn from_reader_with_extensions<R>(
        mut source: R,
        header: &NiftiHeader,
        extender: Extender,
        options: <V as FromSourceOptions>::Options,
    ) -> Result<(V, ExtensionSequence)>
    where
        R: Read,
        V: FromSource<R>,
    {
        // fetch extensions
        let len = header.vox_offset as usize;
        let len = if len < 352 { 0 } else { len - 352 };

        let ext = {
            let source = ByteOrdered::runtime(&mut source, header.endianness);
            ExtensionSequence::from_reader(extender, source, len)?
        };

        // fetch volume (rest of file)
        Ok((V::from_reader(source, &header, options)?, ext))
    }

    fn from_file_2<P, R>(
        path: P,
        mut stream: R,
        options: <V as FromSourceOptions>::Options,
    ) -> Result<Self>
    where
        P: AsRef<Path>,
        R: Read,
        V: FromSource<R>,
        V: FromSource<BufReader<File>>,
        V: FromSource<GzDecoder<BufReader<File>>>,
    {
        let header = NiftiHeader::from_reader(&mut stream)?;
        let (volume, ext) = if &header.magic == MAGIC_CODE_NI1 {
            // extensions and volume are in another file

            // extender is optional
            let extender = Extender::from_reader_optional(&mut stream)?.unwrap_or_default();

            // look for corresponding img file
            let img_path = path.as_ref().to_path_buf();
            let mut img_path_gz = into_img_file_gz(img_path);

            Self::from_file_with_extensions(&img_path_gz, &header, extender, options.clone())
                .or_else(|e| {
                    match e {
                        NiftiError::Io(ref io_e) if io_e.kind() == io::ErrorKind::NotFound => {
                            // try .img file instead (remove .gz extension)
                            let has_ext = img_path_gz.set_extension("");
                            debug_assert!(has_ext);
                            Self::from_file_with_extensions(img_path_gz, &header, extender, options)
                        }
                        e => Err(e),
                    }
                })
                .map_err(|e| {
                    if let NiftiError::Io(io_e) = e {
                        NiftiError::MissingVolumeFile(io_e)
                    } else {
                        e
                    }
                })?
        } else {
            // extensions and volume are in the same source

            let extender = Extender::from_reader(&mut stream)?;
            let len = header.vox_offset as usize;
            let len = if len < 352 { 0 } else { len - 352 };

            let ext = {
                let mut stream = ByteOrdered::runtime(&mut stream, header.endianness);
                ExtensionSequence::from_reader(extender, stream, len)?
            };

            let volume = FromSource::from_reader(stream, &header, options)?;

            (volume, ext)
        };

        Ok(GenericNiftiObject {
            header,
            extensions: ext,
            volume,
        })
    }

    fn from_file_2_erased<P: AsRef<Path>, S: 'static>(
        path: P,
        mut stream: S,
        options: <V as FromSourceOptions>::Options,
    ) -> Result<Self>
    where
        S: Read,
        V: FromSource<Box<dyn Read>>,
    {
        let header = NiftiHeader::from_reader(&mut stream)?;
        let (volume, ext) = if &header.magic == MAGIC_CODE_NI1 {
            // extensions and volume are in another file

            // extender is optional
            let extender = Extender::from_reader_optional(&mut stream)?.unwrap_or_default();

            // look for corresponding img file
            let img_path = path.as_ref().to_path_buf();
            let mut img_path_gz = into_img_file_gz(img_path);

            Self::from_file_with_extensions_erased(&img_path_gz, &header, extender, options.clone())
                .or_else(|e| {
                    match e {
                        NiftiError::Io(ref io_e) if io_e.kind() == io::ErrorKind::NotFound => {
                            // try .img file instead (remove .gz extension)
                            let has_ext = img_path_gz.set_extension("");
                            debug_assert!(has_ext);
                            Self::from_file_with_extensions_erased(
                                img_path_gz,
                                &header,
                                extender,
                                options,
                            )
                        }
                        e => Err(e),
                    }
                })
                .map_err(|e| {
                    if let NiftiError::Io(io_e) = e {
                        NiftiError::MissingVolumeFile(io_e)
                    } else {
                        e
                    }
                })?
        } else {
            // extensions and volume are in the same source

            let extender = Extender::from_reader(&mut stream)?;
            let len = header.vox_offset as usize;
            let len = if len < 352 { 0 } else { len - 352 };

            let ext = {
                let mut stream = ByteOrdered::runtime(&mut stream, header.endianness);
                ExtensionSequence::from_reader(extender, stream, len)?
            };

            let stream = Box::from(stream);
            let volume = FromSource::<Box<dyn Read>>::from_reader(stream, &header, options)?;

            (volume, ext)
        };

        Ok(GenericNiftiObject {
            header,
            extensions: ext,
            volume,
        })
    }

    fn from_file_pair_2<S, Q>(
        mut hdr_stream: S,
        vol_path: Q,
        options: <V as FromSourceOptions>::Options,
    ) -> Result<Self>
    where
        S: Read,
        Q: AsRef<Path>,
        V: FromSource<BufReader<File>>,
        V: FromSource<GzDecoder<BufReader<File>>>,
    {
        let header = NiftiHeader::from_reader(&mut hdr_stream)?;
        let extender = Extender::from_reader_optional(hdr_stream)?.unwrap_or_default();
        let (volume, extensions) =
            Self::from_file_with_extensions(vol_path, &header, extender, options)?;

        Ok(GenericNiftiObject {
            header,
            extensions,
            volume,
        })
    }

    fn from_file_pair_2_erased<S, Q>(
        mut hdr_stream: S,
        vol_path: Q,
        options: <V as FromSourceOptions>::Options,
    ) -> Result<Self>
    where
        S: Read,
        Q: AsRef<Path>,
        V: FromSource<Box<dyn Read>>,
    {
        let header = NiftiHeader::from_reader(&mut hdr_stream)?;
        let extender = Extender::from_reader_optional(hdr_stream)?.unwrap_or_default();
        let (volume, extensions) =
            Self::from_file_with_extensions_erased(vol_path, &header, extender, options)?;

        Ok(GenericNiftiObject {
            header,
            extensions,
            volume,
        })
    }

    /// Read a NIFTI volume, along with the extensions, from an image file. NIFTI-1 volume
    /// files usually have the extension ".img" or ".img.gz". In the latter case, the file
    /// is automatically decoded as a Gzip stream.
    fn from_file_with_extensions<P>(
        path: P,
        header: &NiftiHeader,
        extender: Extender,
        options: <V as FromSourceOptions>::Options,
    ) -> Result<(V, ExtensionSequence)>
    where
        P: AsRef<Path>,
        V: FromSource<BufReader<File>>,
        V: FromSource<GzDecoder<BufReader<File>>>,
    {
        let path = path.as_ref();
        let gz = is_gz_file(path);
        let stream = BufReader::new(File::open(path)?);

        if gz {
            Self::from_reader_with_extensions(GzDecoder::new(stream), &header, extender, options)
        } else {
            Self::from_reader_with_extensions(stream, &header, extender, options)
        }
    }

    /// Read a NIFTI volume, along with the extensions, from an image file. NIFTI-1 volume
    /// files usually have the extension ".img" or ".img.gz". In the latter case, the file
    /// is automatically decoded as a Gzip stream.
    fn from_file_with_extensions_erased<P>(
        path: P,
        header: &NiftiHeader,
        extender: Extender,
        options: <V as FromSourceOptions>::Options,
    ) -> Result<(V, ExtensionSequence)>
    where
        P: AsRef<Path>,
        V: FromSource<Box<dyn Read>>,
    {
        let reader = open_file_maybe_gz(path)?;
        Self::from_reader_with_extensions(reader, &header, extender, options)
    }
}
