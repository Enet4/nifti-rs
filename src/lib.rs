extern crate byteorder;
extern crate acc_reader;
extern crate seek_bufread;

pub mod header;
pub mod volume;
pub mod error;
mod util;

use std::fs::File;
use std::path::Path;
use util::ReadSeek;


#[derive(Debug)]
pub struct NIFTIObject<R: ReadSeek> {
    header: header::NIFTIHeader,
    volume: volume::NIFTIVolume<R>,
}

impl NIFTIObject<File> {
    pub fn from_file<P: AsRef<Path>>(&self, path: P) -> NIFTIObject<File> {
        unimplemented!()
    }
}

impl<R: ReadSeek> NIFTIObject<R> {
    pub fn from_stream(&self, source: R) -> NIFTIObject<R> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
