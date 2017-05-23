use std::fs::File;
use std::path::Path;
use std::io::{BufReader, Read};

use error::NiftiError;
use extension::{ExtensionSequence};
use header::NiftiHeader;
use header::MAGIC_CODE_NI1;
use volume::{NiftiVolume, InMemNiftiVolume};
use util::Endianness;
use error::Result;
use byteorder::{BigEndian, LittleEndian};
use flate2::bufread::GzDecoder;

/// Trait type for all possible implementations of
/// owning NIFTI-1 objects.
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
    /// or ".img.gz" if the header is also gzip-encoded.
    /// 
    /// # Example
    /// 
    /// ```no_run
    /// use nifti::InMemNiftiObject;
    /// # use nifti::error::Result;
    ///
    /// # fn run() -> Result<()> {
    /// let obj = InMemNiftiObject::from_file("minimal.gii.gz")?;
    /// # Ok(())
    /// # }
    /// ```
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

    /// Retrieve a NIFTI object as separate header and volume files.
    /// This method is useful when file names are not conventional for a
    /// NIFTI file pair.
    pub fn from_file_pair<P1, P2>(hdr_path: P1, vol_path: P2) -> Result<InMemNiftiObject>
        where P1: AsRef<Path>,
              P2: AsRef<Path>
    {
        let (header, endianness) = NiftiHeader::from_file(hdr_path)?;
        let mut eof = header.vox_offset as usize;
        if eof > 352 {
            eof = 352;
        }
        
        let mut stream = BufReader::new(File::open(vol_path)?);
        let ext = match endianness {
            Endianness::LE => ExtensionSequence::from_stream::<LittleEndian, _>(&mut stream, eof),
            Endianness::BE => ExtensionSequence::from_stream::<BigEndian, _>(&mut stream, eof),
        }?;
        
        let volume = InMemNiftiVolume::from_stream(stream, &header, endianness)?;

        Ok(InMemNiftiObject {
            header,
            extensions: ext,
            volume,
        })
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

    /// Retrieve a NIFTI object from a stream of data.
    /// 
    /// # Errors
    /// 
    /// - [`NiftiError::NoVolumeData`] if the source only contains (or claims to contain)
    /// a header.
    pub fn new_from_stream<R: Read>(&self, mut source: R) -> Result<InMemNiftiObject> {
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