use std::io::Read;
use error::Result;
use byteorder::{ByteOrder, ReadBytesExt};

#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub struct Extender {
    extension: [u8; 4],
}

impl Extender {

    pub fn from_stream<S: Read>(mut source: S) -> Result<Self> {
        let mut extension = [0u8; 4];
        source.read_exact(&mut extension)?;
        Ok(Extender { extension })
    }

    pub fn from_stream_optional<S: Read>(mut source: S) -> Result<Option<Self>> {
        let mut extension = [0u8; 4];
        match source.read_exact(&mut extension) {
            Ok(_) => Ok(Some(Extender { extension })),
            Err(_) => {
                Ok(None) // TODO send other errors properly 
            }
        }
    
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

    pub fn from_stream<B: ByteOrder, S: Read>(extender: Extender, mut source: S, end: usize) -> Result<Self> {
        let mut extensions = Vec::new();
        let mut offset = 352;
        if extender.extension[0] != 0 {
            while offset < end {
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

