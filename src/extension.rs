use std::io::{Read, ErrorKind as IoErrorKind};
use error::{NiftiError, Result};
use byteorder::{ByteOrder, ReadBytesExt};

#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub struct Extender {
    extension: [u8; 4],
}

impl Extender {

    /// Fetch the extender code from the given source, while expecting it to exist.
    pub fn from_stream<S: Read>(mut source: S) -> Result<Self> {
        let mut extension = [0u8; 4];
        source.read_exact(&mut extension)?;
        Ok(Extender { extension })
    }

    /// Fetch the extender code from the given source, while 
    /// being possible to not be available.
    /// Returns `None` if the source reaches EoF prematurely.
    /// Any other I/O error is delegated to a `NiftiError`.
    pub fn from_stream_optional<S: Read>(mut source: S) -> Result<Option<Self>> {
        let mut extension = [0u8; 4];
        match source.read_exact(&mut extension) {
            Ok(_) => Ok(Some(Extender { extension })),
            Err(ref e) if e.kind() == IoErrorKind::UnexpectedEof => {
                Ok(None)
            }
            Err(e) => Err(NiftiError::from(e))
        }
    }

    /// Whether extensions should exist after this extender code.
    pub fn has_extensions(&self) -> bool {
        self.extension[0] != 0
    }

}

#[derive(Debug, PartialEq, Clone)]
pub struct Extension {
    esize: i32,
    ecode: i32,
    edata: Vec<u8>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ExtensionSequence {
    extender: Extender,
    extensions: Vec<Extension>,
}

impl IntoIterator for ExtensionSequence {
    type Item = Extension;
    type IntoIter = ::std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.extensions.into_iter()
    }
}

impl<'a> IntoIterator for &'a ExtensionSequence {
    type Item = &'a Extension;
    type IntoIter = ::std::slice::Iter<'a, Extension>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl ExtensionSequence {

    /// Read a sequence of extensions from a source, up until `len` bytes.
    pub fn from_stream<B: ByteOrder, S: Read>(extender: Extender, mut source: S, len: usize) -> Result<Self> {
        let mut extensions = Vec::new();
        if extender.has_extensions() {
            let mut offset = 0;
            while offset < len {
                let esize = source.read_i32::<B>()?;
                let ecode = source.read_i32::<B>()?;
                let data_size = esize as usize - 8;
                let mut edata = vec![0u8; data_size];
                source.read_exact(&mut edata)?;
                extensions.push(Extension {
                    esize,
                    ecode,
                    edata
                });
                offset += esize as usize;
            }
        }

        Ok(ExtensionSequence {
            extender,
            extensions
        })
    }

    pub fn iter(&self) -> ::std::slice::Iter<Extension> {
        self.extensions.iter()
    }
}

