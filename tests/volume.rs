extern crate nifti;
extern crate flate2;
#[macro_use] extern crate pretty_assertions;

use nifti::{NiftiHeader, InMemNiftiVolume, NiftiVolume, Endianness};

#[test]
fn minimal_img_gz() {
    let minimal_hdr = NiftiHeader {
        sizeof_hdr: 348,
        dim: [3, 64, 64, 10, 0, 0, 0, 0],
        datatype: 2,
        bitpix: 8,
        pixdim: [0., 3., 3., 3., 0., 0., 0., 0.],
        vox_offset: 0.,
        scl_slope: 0.,
        scl_inter: 0.,
        magic: *b"ni1\0",
        ..Default::default()
    };

    const FILE_NAME: &str = "resources/minimal.img.gz";
    let volume = InMemNiftiVolume::from_file(FILE_NAME, &minimal_hdr, Endianness::BE).unwrap();

    assert_eq!(volume.dim(), [64, 64, 10].as_ref());
    // TODO
}
