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
use util::{into_img_file_gz, is_gz_file};
use volume::NiftiVolume;
use volume::inmem::InMemNiftiVolume;
use volume::streamed::StreamedNiftiVolume;

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
    /// use nifti::InMemNiftiObject;
    /// # use nifti::error::Result;
    ///
    /// # fn run() -> Result<()> {
    /// let obj = InMemNiftiObject::from_file("minimal.nii.gz")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let gz = is_gz_file(&path);

        let file = BufReader::new(File::open(&path)?);
        if gz {
            Self::from_file_2(path, GzDecoder::new(file))
        } else {
            Self::from_file_2(path, file)
        }
    }

    fn from_file_2<P: AsRef<Path>, S>(path: P, mut stream: S) -> Result<Self>
    where
        S: Read,
    {
        let header = NiftiHeader::from_stream(&mut stream)?;
        let (volume, ext) = if &header.magic == MAGIC_CODE_NI1 {
            // extensions and volume are in another file

            // extender is optional
            let extender = Extender::from_stream_optional(&mut stream)?.unwrap_or_default();

            // look for corresponding img file
            let img_path = path.as_ref().to_path_buf();
            let mut img_path_gz = into_img_file_gz(img_path);

            Self::from_file_with_extensions(&img_path_gz, &header, extender)
                .or_else(|e| {
                    match e {
                        NiftiError::Io(ref io_e) if io_e.kind() == io::ErrorKind::NotFound => {
                            // try .img file instead (remove .gz extension)
                            let has_ext = img_path_gz.set_extension("");
                            debug_assert!(has_ext);
                            Self::from_file_with_extensions(
                                img_path_gz,
                                &header,
                                extender,
                            )
                        }
                        e => Err(e),
                    }
                }).map_err(|e| {
                    if let NiftiError::Io(io_e) = e {
                        NiftiError::MissingVolumeFile(io_e)
                    } else {
                        e
                    }
                })?
        } else {
            // extensions and volume are in the same source

            let extender = Extender::from_stream(&mut stream)?;
            let len = header.vox_offset as usize;
            let len = if len < 352 { 0 } else { len - 352 };

            let ext = {
                let mut stream = ByteOrdered::runtime(&mut stream, header.endianness);
                ExtensionSequence::from_stream(extender, stream, len)?
            };

            let volume = InMemNiftiVolume::from_stream(stream, &header)?;

            (volume, ext)
        };

        Ok(InMemNiftiObject {
            header,
            extensions: ext,
            volume,
        })
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
            Self::from_file_pair_2(GzDecoder::new(file), vol_path)
        } else {
            Self::from_file_pair_2(file, vol_path)
        }
    }

    fn from_file_pair_2<S, Q>(mut hdr_stream: S, vol_path: Q) -> Result<InMemNiftiObject>
    where
        S: Read,
        Q: AsRef<Path>,
    {
        let header = NiftiHeader::from_stream(&mut hdr_stream)?;
        let extender = Extender::from_stream_optional(hdr_stream)?.unwrap_or_default();
        let (volume, extensions) =
            Self::from_file_with_extensions(vol_path, &header, extender)?;

        Ok(InMemNiftiObject {
            header,
            extensions,
            volume,
        })
    }

    /// Retrieve an in-memory NIfTI object from a stream of data.
    #[deprecated(note = "use `from_reader` instead")]
    pub fn new_from_stream<R: Read>(&self, source: R) -> Result<Self> {
        Self::from_reader(source)
    }

    /// Retrieve an in-memory NIfTI object from a stream of data.
    ///
    /// # Errors
    ///
    /// - `NiftiError::NoVolumeData` if the source only contains (or claims to contain)
    /// a header.
    pub fn from_reader<R: Read>(mut source: R) -> Result<Self> {
        let header = NiftiHeader::from_stream(&mut source)?;
        if &header.magic == MAGIC_CODE_NI1 {
            return Err(NiftiError::NoVolumeData);
        }
        let len = header.vox_offset as usize;
        let len = if len < 352 { 0 } else { len - 352 };
        let extender = Extender::from_stream(&mut source)?;
        
        let ext = {
            let source = ByteOrdered::runtime(&mut source, header.endianness);
            ExtensionSequence::from_stream(extender, source, len)?
        };

        let volume = InMemNiftiVolume::from_stream(source, &header)?;

        Ok(InMemNiftiObject {
            header,
            extensions: ext,
            volume,
        })
    }

    /// Read a NIFTI volume, and extensions, from a stream of data. The header,
    /// extender code and expected byte order of the volume's data must be
    /// known in advance.
    fn from_stream_with_extensions<R>(
        mut source: R,
        header: &NiftiHeader,
        extender: Extender,
    ) -> Result<(InMemNiftiVolume, ExtensionSequence)>
    where
        R: Read,
    {
        // fetch extensions
        let len = header.vox_offset as usize;
        let len = if len < 352 { 0 } else { len - 352 };

        let ext = {
            let source = ByteOrdered::runtime(&mut source, header.endianness);
            ExtensionSequence::from_stream::<_, _>(extender, source, len)?
        };

        // fetch volume (rest of file)
        Ok((InMemNiftiVolume::from_stream(source, &header)?, ext))
    }

    /// Read a NIFTI volume, along with the extensions, from an image file. NIFTI-1 volume
    /// files usually have the extension ".img" or ".img.gz". In the latter case, the file
    /// is automatically decoded as a Gzip stream.
    fn from_file_with_extensions<P>(
        path: P,
        header: &NiftiHeader,
        extender: Extender,
    ) -> Result<(InMemNiftiVolume, ExtensionSequence)>
    where
        P: AsRef<Path>,
    {
        let gz = path
            .as_ref()
            .extension()
            .map(|a| a.to_string_lossy() == "gz")
            .unwrap_or(false);
        let stream = BufReader::new(File::open(path)?);

        if gz {
            Self::from_stream_with_extensions(GzDecoder::new(stream), &header, extender)
        } else {
            Self::from_stream_with_extensions(stream, &header, extender)
        }
    }
}

/// A NIfTI object containing an in-memory volume.
pub type StreamedNiftiObject<R> = GenericNiftiObject<StreamedNiftiVolume<R>>;

impl<R> StreamedNiftiObject<R>
where
    R: Read,
{
    /// Retrieve an in-memory NIfTI object from a stream of data.
    ///
    /// # Errors
    ///
    /// - `NiftiError::NoVolumeData` if the source only contains (or claims to contain)
    /// a header.
    pub fn new_from_stream(&self, mut source: R) -> Result<Self> {
        let header = NiftiHeader::from_stream(&mut source)?;
        if &header.magic == MAGIC_CODE_NI1 {
            return Err(NiftiError::NoVolumeData);
        }
        let len = header.vox_offset as usize;
        let len = if len < 352 { 0 } else { len - 352 };
        let extender = Extender::from_stream(&mut source)?;
        
        let ext = {
            let source = ByteOrdered::runtime(&mut source, header.endianness);
            ExtensionSequence::from_stream(extender, source, len)?
        };

        let volume = StreamedNiftiVolume::from_reader(source, &header)?;

        Ok(StreamedNiftiObject {
            header,
            extensions: ext,
            volume,
        })
    }

    /// Read a NIFTI volume, and extensions, from a stream of data. The header,
    /// extender code and expected byte order of the volume's data must be
    /// known in advance.
    fn from_stream_with_extensions(
        mut source: R,
        header: &NiftiHeader,
        extender: Extender,
    ) -> Result<(StreamedNiftiVolume<R>, ExtensionSequence)>
    where
        R: Read,
    {
        // fetch extensions
        let len = header.vox_offset as usize;
        let len = if len < 352 { 0 } else { len - 352 };

        let ext = {
            let source = ByteOrdered::runtime(&mut source, header.endianness);
            ExtensionSequence::from_stream::<_, _>(extender, source, len)?
        };

        // fetch volume (rest of file)
        Ok((StreamedNiftiVolume::from_reader(source, &header)?, ext))
    }
}

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
    /// # use nifti::error::Result;
    ///
    /// # fn run() -> Result<()> {
    /// let obj = StreamedNiftiObject::from_file("minimal.nii.gz")?;
    /// 
    /// let volume = obj.into_volume();
    /// for slice in volume {
    ///     // manipulate slice here
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let gz = is_gz_file(&path);

        let file = BufReader::new(File::open(&path)?);
        if gz {
            Self::from_file_2(path, GzDecoder::new(file))
        } else {
            Self::from_file_2(path, file)
        }
    }

    fn from_file_2<P: AsRef<Path>, S: 'static>(path: P, mut stream: S) -> Result<Self>
    where
        S: Read,
    {
        let header = NiftiHeader::from_stream(&mut stream)?;
        let (volume, ext) = if &header.magic == MAGIC_CODE_NI1 {
            // extensions and volume are in another file

            // extender is optional
            let extender = Extender::from_stream_optional(&mut stream)?.unwrap_or_default();

            // look for corresponding img file
            let img_path = path.as_ref().to_path_buf();
            let mut img_path_gz = into_img_file_gz(img_path);

            Self::from_file_with_extensions(&img_path_gz, &header, extender)
                .or_else(|e| {
                    match e {
                        NiftiError::Io(ref io_e) if io_e.kind() == io::ErrorKind::NotFound => {
                            // try .img file instead (remove .gz extension)
                            let has_ext = img_path_gz.set_extension("");
                            debug_assert!(has_ext);
                            Self::from_file_with_extensions(
                                img_path_gz,
                                &header,
                                extender,
                            )
                        }
                        e => Err(e),
                    }
                }).map_err(|e| {
                    if let NiftiError::Io(io_e) = e {
                        NiftiError::MissingVolumeFile(io_e)
                    } else {
                        e
                    }
                })?
        } else {
            // extensions and volume are in the same source

            let extender = Extender::from_stream(&mut stream)?;
            let len = header.vox_offset as usize;
            let len = if len < 352 { 0 } else { len - 352 };

            let ext = {
                let mut stream = ByteOrdered::runtime(&mut stream, header.endianness);
                ExtensionSequence::from_stream(extender, stream, len)?
            };

            let stream = Box::from(stream);
            let volume = StreamedNiftiVolume::<Box<dyn Read>>::from_reader(stream, &header)?;

            (volume, ext)
        };

        Ok(StreamedNiftiObject {
            header,
            extensions: ext,
            volume,
        })
    }

    /// Retrieve a NIfTI object as separate header and volume files, for
    /// streamed volume reading. This method is useful when file names are not
    /// conventional for a NIfTI file pair.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use nifti::{NiftiObject, StreamedNiftiObject};
    /// # use nifti::error::Result;
    ///
    /// # fn run() -> Result<()> {
    /// let obj = StreamedNiftiObject::from_file_pair("abc.hdr", "abc.img.gz")?;
    /// 
    /// let volume = obj.into_volume();
    /// for slice in volume {
    ///     // manipulate slice here
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_file_pair<P, Q>(hdr_path: P, vol_path: Q) -> Result<Self>
    where
        P: AsRef<Path>,
        Q: AsRef<Path>,
    {
        let gz = is_gz_file(&hdr_path);

        let file = BufReader::new(File::open(&hdr_path)?);
        if gz {
            Self::from_file_pair_2(GzDecoder::new(file), vol_path)
        } else {
            Self::from_file_pair_2(file, vol_path)
        }
    }

    fn from_file_pair_2<S, Q>(mut hdr_stream: S, vol_path: Q) -> Result<Self>
    where
        S: Read,
        Q: AsRef<Path>,
    {
        let header = NiftiHeader::from_stream(&mut hdr_stream)?;
        let extender = Extender::from_stream_optional(hdr_stream)?.unwrap_or_default();
        let (volume, extensions) =
            Self::from_file_with_extensions(vol_path, &header, extender)?;

        Ok(StreamedNiftiObject {
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
    ) -> Result<(StreamedNiftiVolume<Box<dyn Read>>, ExtensionSequence)>
    where
        P: AsRef<Path>,
    {
        let gz = path
            .as_ref()
            .extension()
            .map(|a| a.to_string_lossy() == "gz")
            .unwrap_or(false);
        let stream = BufReader::new(File::open(path)?);

        if gz {
            Self::from_stream_with_extensions(Box::from(GzDecoder::new(stream)), &header, extender)
        } else {
            Self::from_stream_with_extensions(Box::from(stream), &header, extender)
        }
    }
}
