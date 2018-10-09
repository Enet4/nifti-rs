//! An application for reading NIFTI-1 file meta-data.

extern crate nifti;

use std::env;
use nifti::NiftiHeader;

fn main() {
    let mut args = env::args().skip(1);
    let filename = args.next().expect("Path to NIFTI file is required");
    let header = NiftiHeader::from_file(filename)
        .expect("Failed to read NIFTI file");
    println!("{:#?}", &header);
}
