extern crate flate2;
#[cfg(feature = "ndarray_volumes")]
extern crate ndarray;
extern crate nifti;
#[macro_use]
extern crate pretty_assertions;

use nifti::{
    Endianness, InMemNiftiObject, StreamedNiftiObject, NiftiHeader, NiftiObject, NiftiType,
    NiftiVolume, RandomAccessNiftiVolume, XForm,
};

mod util;

use util::{minimal_header_hdr_gt, minimal_header_nii_gt};

#[test]
fn minimal_nii_gz() {
    let minimal_hdr = minimal_header_nii_gt();

    const FILE_NAME: &str = "resources/minimal.nii.gz";
    let obj = InMemNiftiObject::from_file(FILE_NAME).unwrap();
    assert_eq!(obj.header(), &minimal_hdr);
    let volume = obj.volume();
    assert_eq!(volume.data_type(), NiftiType::Uint8);
    assert_eq!(volume.dim(), [64, 64, 10].as_ref());
}

#[test]
fn streamed_minimal_nii_gz() {
    let minimal_hdr = minimal_header_nii_gt();

    const FILE_NAME: &str = "resources/minimal.nii.gz";
    let obj = StreamedNiftiObject::from_file(FILE_NAME).unwrap();
    assert_eq!(obj.header(), &minimal_hdr);
    let volume = obj.volume();
    assert_eq!(volume.data_type(), NiftiType::Uint8);
    assert_eq!(volume.dim(), [64, 64, 10].as_ref());
}

#[test]
fn minimal_nii() {
    let minimal_hdr = minimal_header_nii_gt();

    const FILE_NAME: &str = "resources/minimal.nii";
    let obj = InMemNiftiObject::from_file(FILE_NAME).unwrap();
    assert_eq!(obj.header(), &minimal_hdr);
    let volume = obj.volume();
    assert_eq!(volume.data_type(), NiftiType::Uint8);
    assert_eq!(volume.dim(), [64, 64, 10].as_ref());
}

#[test]
fn streamed_minimal_nii() {
    let minimal_hdr = minimal_header_nii_gt();

    const FILE_NAME: &str = "resources/minimal.nii";
    let obj = StreamedNiftiObject::from_file(FILE_NAME).unwrap();
    assert_eq!(obj.header(), &minimal_hdr);
    let volume = obj.volume();
    assert_eq!(volume.data_type(), NiftiType::Uint8);
    assert_eq!(volume.dim(), [64, 64, 10].as_ref());
}

#[test]
fn minimal_by_hdr() {
    let minimal_hdr = minimal_header_hdr_gt();

    const FILE_NAME: &str = "resources/minimal.hdr";
    let obj = InMemNiftiObject::from_file(FILE_NAME).unwrap();
    assert_eq!(obj.header(), &minimal_hdr);
    let volume = obj.volume();
    assert_eq!(volume.data_type(), NiftiType::Uint8);
    assert_eq!(volume.dim(), [64, 64, 10].as_ref());
}

#[test]
fn streamed_minimal_by_hdr() {
    let minimal_hdr = minimal_header_hdr_gt();

    const FILE_NAME: &str = "resources/minimal.hdr";
    let obj = StreamedNiftiObject::from_file(FILE_NAME).unwrap();
    assert_eq!(obj.header(), &minimal_hdr);
    let volume = obj.volume();
    assert_eq!(volume.data_type(), NiftiType::Uint8);
    assert_eq!(volume.dim(), [64, 64, 10].as_ref());
}

#[test]
fn minimal_by_hdr_and_img_gz() {
    let minimal_hdr = minimal_header_hdr_gt();

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
    let minimal_hdr = minimal_header_hdr_gt();

    const FILE_NAME: &str = "resources/minimal.hdr.gz";
    let obj = InMemNiftiObject::from_file(FILE_NAME).unwrap();
    assert_eq!(obj.header(), &minimal_hdr);
    let volume = obj.volume();
    assert_eq!(volume.data_type(), NiftiType::Uint8);
    assert_eq!(volume.dim(), [64, 64, 10].as_ref());
}

#[test]
fn minimal_by_pair() {
    let minimal_hdr = minimal_header_hdr_gt();

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
        vox_offset: 352.,
        scl_slope: 1.,
        scl_inter: 0.,
        srow_x: [1., 0., 0., 0.],
        srow_y: [0., 1., 0., 0.],
        srow_z: [0., 0., 1., 0.],
        sform_code: 2,
        qform_code: 0,
        magic: *b"n+1\0",
        endianness: Endianness::Little,
        ..Default::default()
    };

    const FILE_NAME: &str = "resources/f32.nii.gz";
    let obj = InMemNiftiObject::from_file(FILE_NAME).unwrap();
    assert_eq!(obj.header(), &f32_hdr);

    assert_eq!(obj.header().sform().unwrap(), XForm::AlignedAnat);

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

#[test]
fn streamed_f32_nii_gz() {
    let f32_hdr = NiftiHeader {
        sizeof_hdr: 348,
        dim: [3, 11, 11, 11, 1, 1, 1, 1],
        datatype: 16,
        bitpix: 32,
        vox_offset: 352.,
        scl_slope: 1.,
        scl_inter: 0.,
        srow_x: [1., 0., 0., 0.],
        srow_y: [0., 1., 0., 0.],
        srow_z: [0., 0., 1., 0.],
        sform_code: 2,
        qform_code: 0,
        magic: *b"n+1\0",
        endianness: Endianness::Little,
        ..Default::default()
    };

    const FILE_NAME: &str = "resources/f32.nii.gz";
    let obj = StreamedNiftiObject::from_file(FILE_NAME).unwrap();
    assert_eq!(obj.header(), &f32_hdr);

    assert_eq!(obj.header().sform().unwrap(), XForm::AlignedAnat);

    let volume = obj.into_volume();
    assert_eq!(volume.data_type(), NiftiType::Float32);
    assert_eq!(volume.dim(), [11, 11, 11].as_ref());
    assert_eq!(volume.slice_dim(), &[11, 11]);

    let slices: Vec<_> = volume
        .map(|r| r.expect("slice construction should not fail"))
        .collect();

    for slice in &slices {
        assert_eq!(slice.data_type(), NiftiType::Float32);
        assert_eq!(slice.dim(), &[11, 11]);
    }

    assert_eq!(slices[5].get_f32(&[5, 5]).unwrap(), 1.);
    assert_eq!(slices[0].get_f32(&[5, 0]).unwrap(), 0.);
    assert_eq!(slices[0].get_f32(&[0, 5]).unwrap(), 0.);
    assert_eq!(slices[5].get_f32(&[0, 0]).unwrap(), 0.);
    assert_eq!(slices[4].get_f32(&[5, 0]).unwrap(), 0.4);
    assert_eq!(slices[5].get_f32(&[0, 8]).unwrap(), 0.8);
}

#[test]
fn bad_file_1() {
    let _ = InMemNiftiObject::from_file("resources/fuzz_artifacts/crash-1.nii");
    // must not panic
}

#[test]
fn bad_file_2() {
    let _ = InMemNiftiObject::from_file("resources/fuzz_artifacts/crash-98aa054390f8d3f932f190dc22ef62c6ff2d6619");
    // must not panic or abort
}

#[test]
fn bad_file_3() {
    let _ = InMemNiftiObject::from_file("resources/fuzz_artifacts/crash-d03c6346fe83a026738f6b8cd2a9335a3f8cb158");
    // must not panic or abort
}

#[test]
fn bad_file_4() {
    let _ = InMemNiftiObject::from_file("resources/fuzz_artifacts/crash-08123ef33416bd6f0c5fa63d44b681b8581d62a0");
    // must not panic or abort
}
