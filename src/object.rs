use std::fs::File;
use std::path::{Path, PathBuf};
use std::io::{BufRead, BufReader, Read};

use error::NiftiError;
use extension::NiftiExtension;
use header::NiftiHeader;
use header::MAGIC_CODE_NI1;
use volume::{NiftiVolume, InMemNiftiVolume};
use util::ReadSeek;
use error::Result;
use flate2::bufread::GzDecoder;

pub trait NiftiObject {
    type Volume: ?Sized + NiftiVolume;

    fn header(&self) -> &NiftiHeader;
    fn extensions(&self) -> Option<&[NiftiExtension]>;
    fn volume(&self) -> &Self::Volume;
}

#[derive(Debug)]
pub struct InMemNiftiObject {
    header: NiftiHeader,
    extensions: (), // TODO
    volume: InMemNiftiVolume,
}

impl InMemNiftiObject {
    pub fn new_from_file<P: AsRef<Path>>(&self, path: P) -> Result<InMemNiftiObject> {
        let gz = path.as_ref().extension()
            .map(|a| a.to_string_lossy().ends_with(".gz"))
            .unwrap_or(false);
        
        let file = BufReader::new(File::open(&path)?);
        if gz {
            let file = GzDecoder::new(file)?;
            self.new_from_file_2(path, file, gz)
        } else {
            self.new_from_file_2(path, file, gz)
        }
    }

    fn new_from_file_2<P: AsRef<Path>, S>(&self, path: P, stream: S, gz: bool) -> Result<InMemNiftiObject>
        where S: Read
    {
        let (header, endianness) = NiftiHeader::from_stream(stream)?;
        let vol_stream = if &header.magic == MAGIC_CODE_NI1 {
            // look for corresponding img file
            let img_path = path.as_ref().to_path_buf()
                .set_extension(if gz {
                    ".img.gz"
                } else {
                    ".img"
                });
            
            unimplemented!()
        } else {

        };
        unimplemented!()
    }

    pub fn new_from_stream<R: ReadSeek>(&self, mut source: R) -> Result<InMemNiftiObject> {
        let (header, endianness) = NiftiHeader::from_stream(&mut source)?;
        if &header.magic == MAGIC_CODE_NI1 {
            return Err(NiftiError::NoVolumeData);
        }
        unimplemented!()
    }
}

impl NiftiObject for InMemNiftiObject {
    type Volume = InMemNiftiVolume;

    fn header(&self) -> &NiftiHeader {
        &self.header
    }

    fn extensions(&self) -> Option<&[NiftiExtension]> {
        unimplemented!()
    }

    fn volume(&self) -> &Self::Volume {
        &self.volume
    }
}