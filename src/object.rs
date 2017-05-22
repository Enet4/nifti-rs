use std::fs::File;
use std::path::Path;
use std::io::{BufReader, Read};

use error::NiftiError;
use extension::{ExtensionSequence};
use header::NiftiHeader;
use header::MAGIC_CODE_NI1;
use volume::{NiftiVolume, InMemNiftiVolume};
use util::{ReadSeek, Endianness};
use error::Result;
use byteorder::{BigEndian, LittleEndian};
use flate2::bufread::GzDecoder;

/// Trait type for all possible implementations of
/// owning NIFTI-1 objects.
pub trait NiftiObject {
    type Volume: ?Sized + NiftiVolume;

    fn header(&self) -> &NiftiHeader;
    fn extensions(&self) -> &ExtensionSequence;
    fn volume(&self) -> &Self::Volume;

    fn into_volume(self) -> Self::Volume;
}

#[derive(Debug)]
pub struct InMemNiftiObject {
    header: NiftiHeader,
    extensions: ExtensionSequence,
    volume: InMemNiftiVolume,
}

impl InMemNiftiObject {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<InMemNiftiObject> {
        let gz = path.as_ref().extension()
            .map(|a| a.to_string_lossy().ends_with(".gz"))
            .unwrap_or(false);
        
        let file = BufReader::new(File::open(&path)?);
        if gz {
            let file = GzDecoder::new(file)?;
            Self::from_file_2(path, file, gz)
        } else {
            Self::from_file_2(path, file, gz)
        }
    }

    fn from_file_2<P: AsRef<Path>, S>(path: P, mut stream: S, gz: bool) -> Result<InMemNiftiObject>
        where S: Read
    {
        let (header, endianness) = NiftiHeader::from_stream(&mut stream)?;

        let mut eof = header.vox_offset as usize;
        if eof > 352 {
            eof = 352;
        }

        let ext = match endianness {
            Endianness::LE => ExtensionSequence::from_stream::<LittleEndian, _>(&mut stream, eof),
            Endianness::BE => ExtensionSequence::from_stream::<BigEndian, _>(&mut stream, eof),
        }?;

        let volume = if &header.magic == MAGIC_CODE_NI1 {
            // look for corresponding img file
            let mut img_path = path.as_ref().to_path_buf();
            img_path.set_extension(if gz {
                    ".img.gz"
                } else {
                    ".img"
                });
            
            InMemNiftiVolume::from_file(img_path, &header, endianness)
        } else {
            InMemNiftiVolume::from_stream(stream, &header, endianness)
        }?;
        
        Ok(InMemNiftiObject {
            header,
            extensions: ext,
            volume
        })
    }

    pub fn new_from_stream<R: ReadSeek>(&self, mut source: R) -> Result<InMemNiftiObject> {
        let (header, endianness) = NiftiHeader::from_stream(&mut source)?;
        if &header.magic == MAGIC_CODE_NI1 {
            return Err(NiftiError::NoVolumeData);
        }
        let mut eof = header.vox_offset as usize;
        if eof > 352 {
            eof = 352;
        }
        let ext = match endianness {
            Endianness::LE => ExtensionSequence::from_stream::<LittleEndian, _>(&mut source, eof),
            Endianness::BE => ExtensionSequence::from_stream::<BigEndian, _>(&mut source, eof),
        }?;

        let volume = InMemNiftiVolume::from_stream(source, &header, endianness)?;

        Ok(InMemNiftiObject {
            header,
            extensions: ext,
            volume
        })
    }
}

impl NiftiObject for InMemNiftiObject {
    type Volume = InMemNiftiVolume;

    fn header(&self) -> &NiftiHeader {
        &self.header
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