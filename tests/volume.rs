extern crate flate2;
extern crate nifti;
#[macro_use]
extern crate pretty_assertions;

#[cfg(feature = "ndarray_volumes")]
#[macro_use]
extern crate approx;
#[cfg(feature = "ndarray_volumes")]
extern crate ndarray;
#[cfg(feature = "ndarray_volumes")]
extern crate num_traits;
#[cfg(feature = "ndarray_volumes")]
extern crate safe_transmute;

use nifti::{InMemNiftiVolume, NiftiVolume, RandomAccessNiftiVolume};

mod util;

use util::minimal_header_hdr_gt;

#[test]
fn minimal_img_gz() {
    let minimal_hdr = minimal_header_hdr_gt();

    const FILE_NAME: &str = "resources/minimal.img.gz";
    let volume = InMemNiftiVolume::from_file(FILE_NAME, &minimal_hdr).unwrap();

    assert_eq!(volume.dim(), [64, 64, 10].as_ref());

    for i in 0..64 {
        for j in 0..64 {
            let expected_value = j as f32;
            for k in 0..10 {
                let coords = [i, j, k];
                let got_value = volume.get_f32(&coords).unwrap();
                assert_eq!(
                    expected_value, got_value,
                    "bad value at coords {:?}",
                    &coords
                );
            }
        }
    }
}

#[cfg(feature = "ndarray_volumes")]
mod ndarray_volumes {
    use super::util::minimal_header_hdr_gt;
    use ndarray::{Array, Axis, IxDyn, ShapeBuilder};
    use nifti::{
        DataElement, InMemNiftiVolume, IntoNdArray, NiftiObject, NiftiType, NiftiVolume,
        ReaderOptions, ReaderStreamedOptions,
    };
    use num_traits::AsPrimitive;
    use std::fmt;
    use std::ops::{Add, Mul};

    #[test]
    fn minimal_img_gz_ndarray_f32() {
        let minimal_hdr = minimal_header_hdr_gt();

        const FILE_NAME: &str = "resources/minimal.img.gz";
        let volume = InMemNiftiVolume::from_file(FILE_NAME, &minimal_hdr).unwrap();

        assert_eq!(volume.dim(), [64, 64, 10].as_ref());

        let volume = volume.into_ndarray::<f32>().unwrap();

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
        let minimal_hdr = minimal_header_hdr_gt();

        const FILE_NAME: &str = "resources/minimal.img.gz";
        let volume = InMemNiftiVolume::from_file(FILE_NAME, &minimal_hdr).unwrap();
        assert_eq!(volume.data_type(), NiftiType::Uint8);
        assert_eq!(volume.dim(), [64, 64, 10].as_ref());

        let volume = volume.into_ndarray::<u8>().unwrap();

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
        let volume = ReaderOptions::new()
            .read_file(FILE_NAME)
            .unwrap()
            .into_volume();
        assert_eq!(volume.data_type(), NiftiType::Float32);

        let volume = volume.into_ndarray::<f32>().unwrap();

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
        let volume = ReaderOptions::new()
            .read_file(FILE_NAME)
            .unwrap()
            .into_volume();
        assert_eq!(volume.data_type(), NiftiType::Float32);

        let volume = volume.into_ndarray::<f64>().unwrap();

        assert_eq!(volume.shape(), [11, 11, 11].as_ref());

        assert!(volume.iter().any(|v| *v != 0.));

        assert_ulps_eq!(volume[[5, 0, 0]], 0.0);
        assert_ulps_eq!(volume[[0, 5, 0]], 0.0);
        assert_ulps_eq!(volume[[0, 0, 5]], 0.0);
        assert_ulps_eq!(volume[[5, 0, 4]], 0.4_f32 as f64);
        assert_ulps_eq!(volume[[0, 8, 5]], 0.8_f32 as f64);
        assert_ulps_eq!(volume[[5, 5, 5]], 1.0_f32 as f64);
    }

    #[test]
    fn streamed_f32_nii_gz_ndarray_f64() {
        const FILE_NAME: &str = "resources/f32.nii.gz";
        let volume = ReaderStreamedOptions::new()
            .read_file(FILE_NAME)
            .unwrap()
            .into_volume();
        assert_eq!(volume.data_type(), NiftiType::Float32);

        let slices: Vec<_> = volume
            .map(|r| r.expect("slice construction should not fail"))
            .map(|slice| slice.into_ndarray::<f64>().unwrap())
            .collect();

        for slice in &slices {
            assert_eq!(slice.shape(), &[11, 11]);
        }
        assert!(slices[4].iter().any(|v| *v != 0.));
        assert!(slices[5].iter().any(|v| *v != 0.));

        assert_ulps_eq!(slices[0][[5, 0]], 0.0);
        assert_ulps_eq!(slices[0][[0, 5]], 0.0);
        assert_ulps_eq!(slices[5][[0, 0]], 0.0);
        assert_ulps_eq!(slices[4][[5, 0]], 0.4_f32 as f64);
        assert_ulps_eq!(slices[5][[0, 8]], 0.8_f32 as f64);
        assert_ulps_eq!(slices[5][[5, 5]], 1.0_f32 as f64);
    }

    #[test]
    fn test_i8() {
        const FILE_NAME: &str = "resources/27/int8.nii";
        test_all(FILE_NAME, NiftiType::Int8);
    }

    #[test]
    fn test_u8() {
        const FILE_NAME: &str = "resources/27/uint8.nii";
        test_all(FILE_NAME, NiftiType::Uint8);
    }

    #[test]
    fn test_i16() {
        const FILE_NAME: &str = "resources/27/int16.nii";
        test_all(FILE_NAME, NiftiType::Int16);
    }

    #[test]
    fn test_u16() {
        const FILE_NAME: &str = "resources/27/uint16.nii";
        test_all(FILE_NAME, NiftiType::Uint16);
    }

    #[test]
    fn test_i32() {
        const FILE_NAME: &str = "resources/27/int32.nii";
        test_all(FILE_NAME, NiftiType::Int32);
    }

    #[test]
    fn test_u32() {
        const FILE_NAME: &str = "resources/27/uint32.nii";
        test_all(FILE_NAME, NiftiType::Uint32);
    }

    #[test]
    fn test_i64() {
        const FILE_NAME: &str = "resources/27/int64.nii";
        test_all(FILE_NAME, NiftiType::Int64);
    }

    #[test]
    fn test_u64() {
        const FILE_NAME: &str = "resources/27/uint64.nii";
        test_all(FILE_NAME, NiftiType::Uint64);
    }

    #[test]
    fn test_f32() {
        const FILE_NAME: &str = "resources/27/float32.nii";
        test_all(FILE_NAME, NiftiType::Float32);
    }

    #[test]
    fn test_f64() {
        const FILE_NAME: &str = "resources/27/float64.nii";
        test_all(FILE_NAME, NiftiType::Float64);
    }

    fn test_all(path: &str, dtype: NiftiType) {
        test_types::<i8>(path, dtype);
        test_types::<u8>(path, dtype);
        test_types::<i16>(path, dtype);
        test_types::<u16>(path, dtype);
        test_types::<i32>(path, dtype);
        test_types::<u32>(path, dtype);
        test_types::<i64>(path, dtype);
        test_types::<u64>(path, dtype);
        test_types::<f32>(path, dtype);
        test_types::<f64>(path, dtype);
    }

    fn test_types<T>(path: &str, dtype: NiftiType)
    where
        T: fmt::Debug,
        T: Add<Output = T>,
        T: Mul<Output = T>,
        T: DataElement,
        T: PartialEq<T>,
        u8: AsPrimitive<T>,
        i8: AsPrimitive<T>,
        u16: AsPrimitive<T>,
        i16: AsPrimitive<T>,
        u32: AsPrimitive<T>,
        i32: AsPrimitive<T>,
        u64: AsPrimitive<T>,
        i64: AsPrimitive<T>,
        f32: AsPrimitive<T>,
        f64: AsPrimitive<T>,
        usize: AsPrimitive<T>,
    {
        let volume = ReaderOptions::new()
            .read_file(path)
            .expect("Can't read input file.")
            .into_volume();
        assert_eq!(volume.data_type(), dtype);

        let data = volume.into_ndarray::<T>().unwrap();
        for (idx, val) in data.iter().enumerate() {
            assert_eq!(idx.as_(), *val);
        }
    }
}
