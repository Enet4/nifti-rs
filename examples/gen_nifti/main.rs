//! An application for writing a NIFTI file from scratch

extern crate nifti;

use std::env;

use crate::nifti::{
    Extension, ExtensionSequence, Extender, writer::WriterOptions
};

fn main() {
    let mut args = env::args().skip(1);
    let filename = args.next().expect("Path to NIFTI file is required");

    // generate some test data 256x256 float32
    let data = ndarray::Array3::<f32>::zeros((256, 256, 1));

    let extension1 = Extension::new(8 + 4, 3, vec![0, 0, 0, 0]);

    let extension2 = Extension::from_str(6, "Hello World!");

    let extension_sequence = ExtensionSequence::new(Extender::from([1u8; 4]), vec![extension1, extension2]);

    WriterOptions::new(&filename)
    .extension_sequence(extension_sequence)
    .write_nifti(&data)
    .unwrap();
}
