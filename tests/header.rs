extern crate nifti;
#[macro_use]
extern crate pretty_assertions;

use nifti::{
    Endianness, Intent, Nifti1Header, Nifti2Header, NiftiHeader, NiftiType, SliceOrder, Unit, XForm,
};
use std::{fs::File, io::Seek};

mod util;

use util::{minimal_header_hdr_gt, minimal_header_nii_gt};

#[test]
fn minimal_hdr() {
    let minimal_hdr = minimal_header_hdr_gt();

    const FILE_NAME: &str = "resources/minimal.hdr";
    let header = NiftiHeader::from_file(FILE_NAME).unwrap();

    assert_eq!(header, minimal_hdr);

    assert_eq!(header.intent().unwrap(), Intent::None);
    assert_eq!(header.data_type().unwrap(), NiftiType::Uint8);
    assert_eq!(header.xyzt_units().unwrap(), (Unit::Unknown, Unit::Unknown));
    assert_eq!(header.slice_order().unwrap(), SliceOrder::Unknown);
    assert_eq!(header.qform().unwrap(), XForm::Unknown);
    assert_eq!(header.sform().unwrap(), XForm::Unknown);
}

#[test]
fn minimal_hdr_gz() {
    let minimal_hdr = minimal_header_hdr_gt();

    const FILE_NAME: &str = "resources/minimal.hdr.gz";
    let header = NiftiHeader::from_file(FILE_NAME).unwrap();

    assert_eq!(header, minimal_hdr);

    assert_eq!(header.intent().unwrap(), Intent::None);
    assert_eq!(header.data_type().unwrap(), NiftiType::Uint8);
    assert_eq!(header.xyzt_units().unwrap(), (Unit::Unknown, Unit::Unknown));
    assert_eq!(header.slice_order().unwrap(), SliceOrder::Unknown);
    assert_eq!(header.qform().unwrap(), XForm::Unknown);
    assert_eq!(header.sform().unwrap(), XForm::Unknown);
}

#[test]
fn minimal_nii() {
    let minimal_hdr = minimal_header_nii_gt();

    const FILE_NAME: &str = "resources/minimal.nii";
    let file = File::open(FILE_NAME).unwrap();
    let header = NiftiHeader::from_reader(file).unwrap();

    assert_eq!(header, minimal_hdr);
    assert_eq!(header.endianness(), Endianness::Big);

    assert_eq!(header.intent().unwrap(), Intent::None);
    assert_eq!(header.data_type().unwrap(), NiftiType::Uint8);
    assert_eq!(header.xyzt_units().unwrap(), (Unit::Unknown, Unit::Unknown));
    assert_eq!(header.slice_order().unwrap(), SliceOrder::Unknown);
    assert_eq!(header.qform().unwrap(), XForm::Unknown);
    assert_eq!(header.sform().unwrap(), XForm::Unknown);
}

#[test]
#[allow(non_snake_case)]
fn avg152T1_LR_hdr_gz() {
    let mut descrip = [0; 80];
    descrip[..10].copy_from_slice(b"FSL3.2beta");
    let avg152t1_lr_hdr = Nifti1Header {
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
        aux_file: *b"none                   \0",
        qform_code: 0,
        sform_code: 4,
        srow_x: [-2., 0., 0., 90.],
        srow_y: [0., 2., 0., -126.],
        srow_z: [0., 0., 2., -72.],
        magic: *b"ni1\0",
        endianness: Endianness::Big,
        ..Default::default()
    }
    .into();

    const FILE_NAME: &str = "resources/avg152T1_LR_nifti.hdr.gz";
    let header = NiftiHeader::from_file(FILE_NAME).unwrap();

    assert_eq!(header, avg152t1_lr_hdr);
    assert_eq!(header.endianness(), Endianness::Big);

    assert_eq!(header.intent().unwrap(), Intent::None);
    assert_eq!(header.data_type().unwrap(), NiftiType::Uint8);
    assert_eq!(header.xyzt_units().unwrap(), (Unit::Mm, Unit::Sec));
    assert_eq!(header.slice_order().unwrap(), SliceOrder::Unknown);
    assert_eq!(header.qform().unwrap(), XForm::Unknown);
    assert_eq!(header.sform().unwrap(), XForm::Mni152);
}

#[test]
#[allow(non_snake_case)]
fn avg152T1_LR_nii_gz() {
    let mut descrip = [0; 80];
    descrip[..10].copy_from_slice(b"FSL3.2beta");
    let avg152t1_lr_hdr = Nifti1Header {
        sizeof_hdr: 348,
        regular: b'r',
        dim: [3, 91, 109, 91, 1, 1, 1, 1],
        datatype: 2,
        bitpix: 8,
        pixdim: [0., 2., 2., 2., 1., 1., 1., 1.],
        vox_offset: 352.,
        scl_slope: 0.,
        scl_inter: 0.,
        xyzt_units: 10,
        cal_max: 255.,
        cal_min: 0.,
        descrip,
        aux_file: *b"none                   \0",
        qform_code: 0,
        sform_code: 4,
        srow_x: [-2., 0., 0., 90.],
        srow_y: [0., 2., 0., -126.],
        srow_z: [0., 0., 2., -72.],
        magic: *b"n+1\0",
        endianness: Endianness::Big,
        ..Default::default()
    }
    .into();

    const FILE_NAME: &str = "resources/avg152T1_LR_nifti.nii.gz";
    let header = NiftiHeader::from_file(FILE_NAME).unwrap();

    assert_eq!(header, avg152t1_lr_hdr);

    assert_eq!(header.intent().unwrap(), Intent::None);
    assert_eq!(header.data_type().unwrap(), NiftiType::Uint8);
    assert_eq!(header.xyzt_units().unwrap(), (Unit::Mm, Unit::Sec));
    assert_eq!(header.slice_order().unwrap(), SliceOrder::Unknown);
    assert_eq!(header.qform().unwrap(), XForm::Unknown);
    assert_eq!(header.sform().unwrap(), XForm::Mni152);
}

#[test]
fn zstat1_nii_gz() {
    let mut descrip = [0; 80];
    descrip[..10].copy_from_slice(b"FSL3.2beta");
    let zstat1_hdr = Nifti1Header {
        sizeof_hdr: 348,
        regular: b'r',
        dim: [3, 64, 64, 21, 1, 1, 1, 1],
        intent_code: 5,
        datatype: 16,
        bitpix: 32,
        pixdim: [-1., 4., 4., 6., 1., 1., 1., 1.],
        srow_x: [0.; 4],
        srow_y: [0.; 4],
        srow_z: [0.; 4],
        vox_offset: 352.,
        scl_slope: 0.,
        scl_inter: 0.,
        xyzt_units: 10,
        cal_max: 25500.,
        cal_min: 3.,
        descrip,
        qform_code: 1,
        sform_code: 0,
        quatern_b: 0.,
        quatern_c: 1.,
        magic: *b"n+1\0",
        endianness: Endianness::Big,
        ..Default::default()
    }
    .into();

    const FILE_NAME: &str = "resources/zstat1.nii.gz";
    let header = NiftiHeader::from_file(FILE_NAME).unwrap();

    assert_eq!(header, zstat1_hdr);

    assert_eq!(header.data_type().unwrap(), NiftiType::Float32);
    assert_eq!(header.intent().unwrap(), Intent::Zscore);
    assert_eq!(header.xyzt_units().unwrap(), (Unit::Mm, Unit::Sec));
    assert_eq!(header.slice_order().unwrap(), SliceOrder::Unknown);
    assert_eq!(header.qform().unwrap(), XForm::ScannerAnat);
    assert_eq!(header.sform().unwrap(), XForm::Unknown);
}

#[test]
fn test_read_ones_dscalar() {
    let descrip = [0; 80];

    let one_dscalar_header = Nifti2Header {
        sizeof_hdr: 540,
        magic: *b"n+2\0\r\n\x1A\n",
        datatype: 16,
        bitpix: 32,
        dim: [6, 1, 1, 1, 1, 1, 91282, 1],
        pixdim: [0., 1., 1., 1., 1., 1., 1., 1.],
        srow_x: [0.; 4],
        srow_y: [0.; 4],
        srow_z: [0.; 4],
        vox_offset: 630784,
        scl_slope: 1.,
        scl_inter: 0.,
        cal_max: 0.,
        cal_min: 0.,
        qform_code: 0,
        sform_code: 0,
        descrip,
        xyzt_units: 12,
        intent_code: 3006,
        intent_name: *b"ConnDenseScalar\0",
        endianness: Endianness::Little,
        ..Default::default()
    }
    .into();

    const FILE_NAME: &str = "resources/cifti/ones.dscalar.nii";
    let mut reader = File::open(FILE_NAME).unwrap();

    let header = NiftiHeader::from_reader(&mut reader).unwrap();
    assert_eq!(header, one_dscalar_header);
    assert_eq!(reader.seek(std::io::SeekFrom::Current(0)).unwrap(), 540);
}
