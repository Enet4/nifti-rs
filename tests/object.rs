extern crate nifti;
extern crate flate2;
#[macro_use] extern crate pretty_assertions;
#[cfg(feature = "ndarray_volumes")] extern crate ndarray;

use nifti::{NiftiHeader, InMemNiftiObject, NiftiObject, NiftiVolume, NiftiType};

#[test]
fn minimal_nii_gz() {
    let minimal_hdr = NiftiHeader {
        sizeof_hdr: 348,
        dim: [3, 64, 64, 10, 0, 0, 0, 0],
        datatype: 2,
        bitpix: 8,
        pixdim: [0., 3., 3., 3., 0., 0., 0., 0.],
        vox_offset: 352.,
        scl_slope: 0.,
        scl_inter: 0.,
        magic: *b"n+1\0",
        ..Default::default()
    };

    const FILE_NAME: &str = "resources/minimal.nii.gz";
    let obj = InMemNiftiObject::from_file(FILE_NAME).unwrap();
    assert_eq!(obj.header(), &minimal_hdr);
    let volume = obj.volume();
    assert_eq!(volume.data_type(), NiftiType::Uint8);
    assert_eq!(volume.dim(), [64, 64, 10].as_ref());
}

#[test]
fn minimal_nii() {
    let minimal_hdr = NiftiHeader {
        sizeof_hdr: 348,
        dim: [3, 64, 64, 10, 0, 0, 0, 0],
        datatype: 2,
        bitpix: 8,
        pixdim: [0., 3., 3., 3., 0., 0., 0., 0.],
        vox_offset: 352.,
        scl_slope: 0.,
        scl_inter: 0.,
        magic: *b"n+1\0",
        ..Default::default()
    };

    const FILE_NAME: &str = "resources/minimal.nii";
    let obj = InMemNiftiObject::from_file(FILE_NAME).unwrap();
    assert_eq!(obj.header(), &minimal_hdr);
    let volume = obj.volume();
    assert_eq!(volume.data_type(), NiftiType::Uint8);
    assert_eq!(volume.dim(), [64, 64, 10].as_ref());
}

#[test]
fn minimal_by_hdr() {
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

    const FILE_NAME: &str = "resources/minimal.hdr";
    let obj = InMemNiftiObject::from_file(FILE_NAME).unwrap();
    assert_eq!(obj.header(), &minimal_hdr);
    let volume = obj.volume();
    assert_eq!(volume.data_type(), NiftiType::Uint8);
    assert_eq!(volume.dim(), [64, 64, 10].as_ref());
}


#[test]
fn minimal_by_hdr_and_img_gz() {
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

    const FILE_NAME: &str = "resources/minimal2.hdr";
    // should attempt to read "resources/minimal2.img.gz"
    let obj = InMemNiftiObject::from_file(FILE_NAME).unwrap();
    assert_eq!(obj.header(), &minimal_hdr);
    let volume = obj.volume();
    assert_eq!(volume.data_type(), NiftiType::Uint8);
    assert_eq!(volume.dim(), [64, 64, 10].as_ref());
}

#[test]
fn minimal_by_hdr_gz() {
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

    const FILE_NAME: &str = "resources/minimal.hdr.gz";
    let obj = InMemNiftiObject::from_file(FILE_NAME).unwrap();
    assert_eq!(obj.header(), &minimal_hdr);
    let volume = obj.volume();
    assert_eq!(volume.data_type(), NiftiType::Uint8);
    assert_eq!(volume.dim(), [64, 64, 10].as_ref());
}

#[test]
fn minimal_by_pair() {
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

    const HDR_FILE_NAME: &str = "resources/minimal.hdr.gz";
    const IMG_FILE_NAME: &str = "resources/minimal.img.gz";
    let obj = InMemNiftiObject::from_file_pair(HDR_FILE_NAME, IMG_FILE_NAME).unwrap();
    assert_eq!(obj.header(), &minimal_hdr);
    let volume = obj.volume();
    assert_eq!(volume.data_type(), NiftiType::Uint8);
    assert_eq!(volume.dim(), [64, 64, 10].as_ref());
}

#[test]
fn f32_nii_gz() {
    let f32_hdr = NiftiHeader {
        sizeof_hdr: 348,
        dim: [3, 11, 11, 11, 1, 1, 1, 1],
        datatype: 16,
        bitpix: 32,
        pixdim: [1., 1., 1., 1., 1., 1., 1., 1.],
        vox_offset: 352.,
        scl_slope: 1.,
        scl_inter: 0.,
        srow_x: [1., 0., 0., 0.],
        srow_y: [0., 1., 0., 0.],
        srow_z: [0., 0., 1., 0.],
        sform_code: 2,
        magic: *b"n+1\0",
        ..Default::default()
    };

    const FILE_NAME: &str = "resources/f32.nii.gz";
    let obj = InMemNiftiObject::from_file(FILE_NAME).unwrap();
    assert_eq!(obj.header(), &f32_hdr);
    let volume = obj.volume();
    assert_eq!(volume.data_type(), NiftiType::Float32);
    assert_eq!(volume.dim(), [11, 11, 11].as_ref());

    assert_eq!(volume.get_f32(&[5, 5, 5]).unwrap(), 1.);
    assert_eq!(volume.get_f32(&[5, 0, 0]).unwrap(), 0.);
    assert_eq!(volume.get_f32(&[0, 5, 0]).unwrap(), 0.);
    assert_eq!(volume.get_f32(&[0, 0, 5]).unwrap(), 0.);
    assert_eq!(volume.get_f32(&[5, 0, 4]).unwrap(), 0.4);
    assert_eq!(volume.get_f32(&[0, 8, 5]).unwrap(), 0.8);
}
