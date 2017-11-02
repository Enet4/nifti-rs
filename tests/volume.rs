#[macro_use]
extern crate approx;
extern crate flate2;
extern crate nifti;
#[macro_use]
extern crate pretty_assertions;

#[cfg(feature = "ndarray_volumes")]
extern crate ndarray;

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
                assert_eq!(
                    expected_value,
                    got_value,
                    "bad value at coords {:?}",
                    &coords
                );
            }
        }
    }
}

#[cfg(feature = "ndarray_volumes")]
mod ndarray_volumes {
    use nifti::{Endianness, InMemNiftiObject, InMemNiftiVolume, NiftiHeader, NiftiObject,
                NiftiVolume, NiftiType, IntoNdArray};
    use ndarray::{Array, Axis, IxDyn, ShapeBuilder};

    #[test]
    fn minimal_img_gz_ndarray_f32() {
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
            assert!(
                slice == e,
                "slice was:\n{:?}\n, expected:\n{:?}",
                &slice,
                &e
            );
        }
    }

    #[test]
    fn minimal_img_gz_ndarray_u8() {
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
        assert_eq!(volume.data_type(), NiftiType::Uint8);
        assert_eq!(volume.dim(), [64, 64, 10].as_ref());

        let volume = volume.to_ndarray::<u8>().unwrap();

        assert_eq!(volume.shape(), [64, 64, 10].as_ref());

        let slices = volume.axis_iter(Axis(1));
        let mut e = Array::zeros(IxDyn(&[64, 10]).f());
        for (j, slice) in slices.enumerate() {
            e.fill(j as u8);
            assert!(
                slice == e,
                "slice was:\n{:?}\n, expected:\n{:?}",
                &slice,
                &e
            );
        }
    }

    #[test]
    fn f32_nii_gz_ndarray() {
        const FILE_NAME: &str = "resources/f32.nii.gz";
        let volume = InMemNiftiObject::from_file(FILE_NAME)
            .unwrap()
            .into_volume();
        assert_eq!(volume.data_type(), NiftiType::Float32);

        let volume = volume.to_ndarray::<f32>().unwrap();

        assert_eq!(volume.shape(), [11, 11, 11].as_ref());

        assert!(volume.iter().any(|v| *v != 0.));

        assert_ulps_eq!(volume[[5, 0, 0]], 0.0);
        assert_ulps_eq!(volume[[0, 5, 0]], 0.0);
        assert_ulps_eq!(volume[[0, 0, 5]], 0.0);
        assert_ulps_eq!(volume[[5, 0, 4]], 0.4);
        assert_ulps_eq!(volume[[0, 8, 5]], 0.8);
        assert_ulps_eq!(volume[[5, 5, 5]], 1.0);
    }

    #[test]
    fn f32_nii_gz_ndarray_f64() {
        const FILE_NAME: &str = "resources/f32.nii.gz";
        let volume = InMemNiftiObject::from_file(FILE_NAME)
            .unwrap()
            .into_volume();
        assert_eq!(volume.data_type(), NiftiType::Float32);

        let volume = volume.to_ndarray::<f64>().unwrap();

        assert_eq!(volume.shape(), [11, 11, 11].as_ref());

        assert!(volume.iter().any(|v| *v != 0.));

        assert_ulps_eq!(volume[[5, 0, 0]], 0.0);
        assert_ulps_eq!(volume[[0, 5, 0]], 0.0);
        assert_ulps_eq!(volume[[0, 0, 5]], 0.0);
        assert_ulps_eq!(volume[[5, 0, 4]], 0.4_f32 as f64);
        assert_ulps_eq!(volume[[0, 8, 5]], 0.8_f32 as f64);
        assert_ulps_eq!(volume[[5, 5, 5]], 1.0_f32 as f64);
    }
}
