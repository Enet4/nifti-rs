//! An application for writing a NIFTI file from scratch

extern crate nifti;

use std::env;

use crate::nifti::{
    header::{MAGIC_CODE_NI1, MAGIC_CODE_NIP1},
    volume::shape::Dim,
    DataElement, NiftiHeader, NiftiType, Result, Extension, ExtensionSequence, Extender
};

fn main() {
    let mut args = env::args().skip(1);
    let filename = args.next().expect("Path to NIFTI file is required");

    // generate some test data 256x256 float32
    let data = ndarray::Array3::<f32>::zeros((256, 256, 1));
    let datatype = NiftiType::Float32;

    let extension1 = Extension::new(8 + 4, 0, vec![0, 0, 0, 0]);

    let extension2 = Extension::new(8 + 4, 0, vec![0, 0, 0, 0]);

    let extension_sequence = ExtensionSequence::new(Extender::from([0u8; 4]), vec![extension1, extension2]);

    let header = NiftiHeader{
        dim: *Dim::from_slice(data.shape()).expect("Don't know why this wouldn't work").raw(),
        sizeof_hdr: 348,
        datatype: datatype as i16,
        bitpix: (datatype.size_of() * 8) as i16,
        vox_offset: 352.0 + extension_sequence.bytes_on_disk() as f32,
        scl_inter: 0.0,
        scl_slope: 1.0,
        magic: *MAGIC_CODE_NIP1,
        ..Default::default()
     };

    println!("{:#?}", &header);
}
