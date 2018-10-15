//! Module for handling and retrieving complete NIFTI-1 objects.

use std::fs::File;
use std::path::Path;
use std::io::{self, BufReader, Read};

use error::NiftiError;
use extension::{Extender, ExtensionSequence};
use header::NiftiHeader;
use header::MAGIC_CODE_NI1;
use volume::NiftiVolume;
use volume::inmem::InMemNiftiVolume;
use util::{is_gz_file, into_img_file_gz, Endianness};
use error::Result;
use byteorder::{BigEndian, LittleEndian};
use flate2::bufread::GzDecoder;

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

/// Data type for a NIFTI object that is fully contained in memory.
#[derive(Debug, PartialEq, Clone)]
pub struct InMemNiftiObject {
    header: NiftiHeader,
    extensions: ExtensionSequence,
    volume: InMemNiftiVolume,
}

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
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<InMemNiftiObject> {
        let gz = is_gz_file(&path);

        let file = BufReader::new(File::open(&path)?);
        if gz {
            Self::from_file_2(path, GzDecoder::new(file))
        } else {
            Self::from_file_2(path, file)
        }
    }

    fn from_file_2<P: AsRef<Path>, S>(path: P, mut stream: S) -> Result<InMemNiftiObject>
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

            InMemNiftiVolume::from_file_with_extensions(&img_path_gz, &header, extender)
                .or_else(|e| {
                    match e {
                        NiftiError::Io(ref io_e) if io_e.kind() == io::ErrorKind::NotFound => {
                            // try .img file instead (remove .gz extension)
                            let has_ext = img_path_gz.set_extension("");
                            debug_assert!(has_ext);
                            InMemNiftiVolume::from_file_with_extensions(
                                img_path_gz,
                                &header,
                                extender,
                            )
                        }
                        e => Err(e),
                    }
                })
                .map_err(|e| if let NiftiError::Io(io_e) = e {
                    NiftiError::MissingVolumeFile(io_e)
                } else {
                    e
                })?
        } else {
            // extensions and volume are in the same source

            let extender = Extender::from_stream(&mut stream)?;
            let len = header.vox_offset as usize;
            let len = if len < 352 { 0 } else { len - 352 };

            let ext = match header.endianness {
                Endianness::LE => {
                    ExtensionSequence::from_stream::<LittleEndian, _>(extender, &mut stream, len)
                }
                Endianness::BE => {
                    ExtensionSequence::from_stream::<BigEndian, _>(extender, &mut stream, len)
                }
            }?;

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
    pub fn from_file_pair<P, Q>(hdr_path: P, vol_path: Q) -> Result<InMemNiftiObject>
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
            InMemNiftiVolume::from_file_with_extensions(vol_path, &header, extender)?;

        Ok(InMemNiftiObject {
            header,
            extensions,
            volume,
        })
    }


    /// Retrieve a NIFTI object from a stream of data.
    ///
    /// # Errors
    ///
    /// - `NiftiError::NoVolumeData` if the source only contains (or claims to contain)
    /// a header.
    pub fn new_from_stream<R: Read>(&self, mut source: R) -> Result<InMemNiftiObject> {
        let header = NiftiHeader::from_stream(&mut source)?;
        if &header.magic == MAGIC_CODE_NI1 {
            return Err(NiftiError::NoVolumeData);
        }
        let len = header.vox_offset as usize;
        let len = if len < 352 { 0 } else { len - 352 };
        let extender = Extender::from_stream(&mut source)?;
        let ext = match header.endianness {
            Endianness::LE => {
                ExtensionSequence::from_stream::<LittleEndian, _>(extender, &mut source, len)
            }
            Endianness::BE => {
                ExtensionSequence::from_stream::<BigEndian, _>(extender, &mut source, len)
            }
        }?;

        let volume = InMemNiftiVolume::from_stream(source, &header)?;

        Ok(InMemNiftiObject {
            header,
            extensions: ext,
            volume,
        })
    }
}

impl NiftiObject for InMemNiftiObject {
    type Volume = InMemNiftiVolume;

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
