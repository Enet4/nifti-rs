extern crate nifti;
extern crate flate2;
#[macro_use] extern crate pretty_assertions;
#[cfg(feature = "ndarray_volumes")] extern crate ndarray;

use nifti::{NiftiHeader, InMemNiftiObject, NiftiObject, NiftiVolume, NiftiType};

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
    
    for i in 0..64 {
        for j in 0..64 {
            let expected_value = j as f32;
            for k in 0..10 {
                let coords = [i, j, k];
                let got_value = volume.get_f32(&coords).unwrap();
                assert_eq!(expected_value, got_value, "bad value at coords {:?}", &coords);
            }
        }
    }
}
