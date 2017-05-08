extern crate nifti;
#[macro_use] extern crate pretty_assertions;

use nifti::NiftiHeader;
use nifti::Endianness;

#[test]
fn minimal_hdr() {
    let MINIMAL_HEADER = NiftiHeader {
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

    const FILE_NAME: &str = "resources/minimal.hdr";
    let (h, e) = NiftiHeader::from_file(FILE_NAME).unwrap();
    
    assert_eq!(h, MINIMAL_HEADER);
    assert_eq!(e, Endianness::BE);
}

#[test]
fn minimal_hdr_gz() {
    let MINIMAL_HEADER = NiftiHeader {
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

    const FILE_NAME: &str = "resources/minimal.hdr.gz";
    let (h, e) = NiftiHeader::from_file(FILE_NAME).unwrap();
    
    assert_eq!(h, MINIMAL_HEADER);
    assert_eq!(e, Endianness::BE);
}

#[test]
fn avg152T1_LR_hdr_gz() {
    let mut descrip = Vec::with_capacity(80);
    descrip.extend(b"FSL3.2beta");
    descrip.resize(80, 0);
    let AVG152T1_LR_HEADER = NiftiHeader {
        sizeof_hdr: 348,
        regular: b'r',
        dim: [3, 91, 109, 91, 1, 1, 1, 1],
        datatype: 2,
        bitpix: 8,
        pixdim: [0., 2., 2., 2., 1., 1., 1., 1.],
        vox_offset: 0.,
        scl_slope: 0.,
        scl_inter: 0.,
        xyzt_units: 10,
        cal_max: 255.,
        cal_min: 0.,
        descrip,
        aux_file: [b'n', b'o', b'n', b'e',
            b' ', b' ', b' ', b' ', b' ', b' ', b' ', b' ', b' ', b' ',
            b' ', b' ', b' ', b' ', b' ', b' ', b' ', b' ', b' ', 0],
        qform_code: 0,
        sform_code: 4,
        srow_x: [-2., 0., 0., 90.],
        srow_y: [0., 2., 0., -126.],
        srow_z: [0., 0., 2., -72.],
        magic: *b"ni1\0",
        ..Default::default()
    };

    const FILE_NAME: &str = "resources/avg152T1_LR_nifti.hdr.gz";
    let (h, e) = NiftiHeader::from_file(FILE_NAME).unwrap();
    
    assert_eq!(h, AVG152T1_LR_HEADER);
    assert_eq!(e, Endianness::BE);
}