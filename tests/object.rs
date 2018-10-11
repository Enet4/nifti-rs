extern crate flate2;
#[cfg(feature = "ndarray_volumes")]
extern crate ndarray;
extern crate nifti;
#[macro_use]
extern crate pretty_assertions;

use nifti::{Endianness, InMemNiftiObject, NiftiHeader, NiftiObject, NiftiType, NiftiVolume};

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
        endianness: Endianness::BE,
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
        endianness: Endianness::BE,
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
        endianness: Endianness::BE,
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
        endianness: Endianness::BE,
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
        endianness: Endianness::BE,
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
        endianness: Endianness::BE,
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
        endianness: Endianness::LE,
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

#[cfg(feature = "ndarray_volumes")]
#[test]
fn test_false_4d() {
    use ndarray::Ix3;
    use nifti:: {InMemNiftiVolume, IntoNdArray};

    let (w, h, d) = (5, 5, 5);
    let mut header = NiftiHeader {
        dim: [4, w, h, d, 1, 1, 1, 1],
        datatype: 2,
        bitpix: 8,
        ..Default::default()
    };
    let raw_data = vec![0; (w * h * d) as usize];
    let mut volume = InMemNiftiVolume::from_raw_data(&header, raw_data).unwrap();
    assert_eq!(header.dim[0], 4);
    assert_eq!(volume.dimensionality(), 4);
    if header.dim[header.dim[0] as usize] == 1 {
        header.dim[0] -= 1;
        volume = InMemNiftiVolume::from_raw_data(&header, volume.to_raw_data()).unwrap();
    }
    assert_eq!(volume.dimensionality(), 3);
    let dyn_data = volume.to_ndarray::<f32>().unwrap();
    assert_eq!(dyn_data.ndim(), 3);
    let data = dyn_data.into_dimensionality::<Ix3>().unwrap();
    assert_eq!(data.ndim(), 3); // Obvious, but it's to avoid being optimized away
}
