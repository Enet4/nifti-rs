extern crate flate2;
#[cfg(feature = "ndarray_volumes")]
extern crate ndarray;
extern crate nifti;
#[macro_use]
extern crate pretty_assertions;

use nifti::{Endianness, InMemNiftiVolume, NiftiHeader, NiftiVolume};

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

#[cfg(feature = "ndarray_volumes")]
mod ndarray_volumes {
    use nifti::{Endianness, NiftiHeader, NiftiVolume, InMemNiftiVolume};
    use ndarray::{Array, Axis, IxDyn, ShapeBuilder};

    #[test]
    fn minimal_img_gz_ndarray() {
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

        let volume = volume.to_ndarray::<f32>().unwrap();
        
        assert_eq!(volume.shape(), [64, 64, 10].as_ref());  

        let slices = volume.axis_iter(Axis(1));
        let mut e = Array::zeros(IxDyn(&[64, 10]).f());
        for (j, slice) in slices.enumerate() {
            e.fill(j as f32);
            assert!(slice == e, "slice was:\n{:?}\n, expected:\n{:?}", &slice, &e);
        }
    }

}