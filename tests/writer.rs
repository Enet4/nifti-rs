#[cfg(feature = "ndarray_volumes")]
extern crate ndarray;
#[cfg(feature = "ndarray_volumes")]
extern crate nifti;
#[cfg(feature = "ndarray_volumes")]
extern crate tempfile;

#[cfg(feature = "ndarray_volumes")]
mod tests {
    use std::{
        fs,
        ops::{Add, Mul},
        path::{Path, PathBuf},
    };

    use ndarray::{Array, Array2, Axis, Dimension, IxDyn, ShapeBuilder, s};
    use tempfile::tempdir;

    use nifti::{
        header::{MAGIC_CODE_NI1, MAGIC_CODE_NIP1},
        object::NiftiObject,
        writer::{write_nifti, write_rgb_nifti},
        InMemNiftiObject, IntoNdArray, NiftiHeader, NiftiType,
    };

    fn get_temporary_path(ext: &str) -> PathBuf {
        let dir = tempdir().unwrap();
        let mut path = dir.into_path();
        if !ext.is_empty() {
            path.push(ext);
        }
        path
    }

    pub fn generate_nifti_header(
        dim: [u16; 8],
        scl_slope: f32,
        scl_inter: f32,
        datatype: NiftiType,
    ) -> NiftiHeader {
        NiftiHeader {
            dim,
            datatype: datatype as i16,
            bitpix: (datatype.size_of() * 8) as i16,
            magic: *MAGIC_CODE_NIP1,
            scl_slope,
            scl_inter,
            ..NiftiHeader::default()
        }
    }

    fn read_as_ndarray<P, D>(path: P) -> (NiftiHeader, Array<f32, D>)
    where
        P: AsRef<Path>,
        D: Dimension,
    {
        let nifti_object = InMemNiftiObject::from_file(path).expect("Nifti file is unreadable.");
        let header = nifti_object.header().clone();
        let volume = nifti_object.into_volume();
        let dyn_data = volume.into_ndarray().unwrap();
        (header, dyn_data.into_dimensionality::<D>().unwrap())
    }

    fn test_write_read(arr: &Array<f32, IxDyn>, path: &str) {
        let path = get_temporary_path(path);
        let mut dim = [1; 8];
        dim[0] = arr.ndim() as u16;
        for (i, s) in arr.shape().iter().enumerate() {
            dim[i + 1] = *s as u16;
        }
        let header = generate_nifti_header(dim, 1.0, 0.0, NiftiType::Float32);
        write_nifti(&path, &arr, Some(&header)).unwrap();

        let read_nifti: Array2<f32> = read_as_ndarray(path).1;
        assert!(read_nifti.all_close(&arr, 1e-10));
    }

    fn f_order_array() -> Array<f32, IxDyn> {
        let dim = vec![4, 4];
        let vec = (0..16).map(|x| x as f32).collect();
        Array::from_shape_vec(IxDyn(&dim).f(), vec).unwrap()
    }

    fn c_order_array() -> Array<f32, IxDyn> {
        let dim = vec![4, 4];
        let vec = (0..16).map(|x| x as f32).collect();
        Array::from_shape_vec(IxDyn(&dim), vec).unwrap()
    }

    #[test]
    fn fortran_writing() {
        // Test .nii
        let arr = f_order_array();
        test_write_read(&arr, "test.nii");
        let mut arr = f_order_array();
        arr.invert_axis(Axis(1));
        test_write_read(&arr, "test_non_contiguous.nii");

        // Test .nii.gz
        let arr = f_order_array();
        test_write_read(&arr, "test.nii.gz");
        let mut arr = f_order_array();
        arr.invert_axis(Axis(1));
        test_write_read(&arr, "test_non_contiguous.nii.gz");
    }

    #[test]
    fn c_writing() {
        // Test .nii
        let arr = c_order_array();
        test_write_read(&arr, "test.nii");

        let mut arr = c_order_array();
        arr.invert_axis(Axis(1));
        test_write_read(&arr, "test_non_contiguous.nii");

        // Test .nii.gz
        let arr = c_order_array();
        test_write_read(&arr, "test.nii.gz");

        let mut arr = c_order_array();
        arr.invert_axis(Axis(1));
        test_write_read(&arr, "test_non_contiguous.nii.gz");
    }

    #[test]
    fn header_slope_inter() {
        let arr = f_order_array();
        let slope = 2.2;
        let inter = 101.1;

        let path = get_temporary_path("test_slope_inter.nii");
        let mut dim = [1; 8];
        dim[0] = arr.ndim() as u16;
        for (i, s) in arr.shape().iter().enumerate() {
            dim[i + 1] = *s as u16;
        }
        let header = generate_nifti_header(dim, slope, inter, NiftiType::Float32);
        let transformed_data = arr.mul(slope).add(inter);
        write_nifti(&path, &transformed_data, Some(&header)).unwrap();

        let read_nifti: Array2<f32> = read_as_ndarray(path).1;
        assert!(read_nifti.all_close(&transformed_data, 1e-10));
    }

    #[test]
    fn write_hdr_standard() {
        let mut data = Array::zeros((10, 11, 12));
        data[(5, 0, 0)] = 1.0;
        data[(6, 0, 0)] = 2.0;

        for fname in &["3d.hdr", "3d.hdr.gz"] {
            let path = get_temporary_path(fname);
            write_nifti(&path, &data, None).unwrap();
            let data_read = read_as_ndarray(path).1;
            assert_eq!(data, data_read);
        }
    }

    #[test]
    fn write_non_contiguous() {
        let mut data = Array::from_elem((3, 4, 11), 1.5);
        data.slice_mut(s![.., .., ..;2]).fill(42.0);

        let path = get_temporary_path("non_contiguous_0.nii.gz");
        write_nifti(&path, &data.slice(s![.., .., ..;2]), None).unwrap();
        assert_eq!(read_as_ndarray(path).1, Array::from_elem((3, 4, 6), 42.0));

        let path = get_temporary_path("non_contiguous_1.nii.gz");
        write_nifti(&path, &data.slice(s![.., .., 1..;2]), None).unwrap();
        assert_eq!(read_as_ndarray(path).1, Array::from_elem((3, 4, 5), 1.5));
    }

    #[test]
    fn write_wrong_description() {
        let dim = [3, 3, 4, 5, 1, 1, 1, 1];
        let mut header = generate_nifti_header(dim, 1.0, 0.0, NiftiType::Float32);
        let path = get_temporary_path("error_description.nii");
        let data = Array::from_elem((3, 4, 5), 1.5);

        // Manual descrip. The original header won't be "repaired", but the written description
        // should be right. To compare the header, we must fix it ourselves.
        let v = "äbcdé".as_bytes().to_vec();
        header.descrip = v.clone();
        write_nifti(&path, &data, Some(&header)).unwrap();
        let (new_header, new_data) = read_as_ndarray(&path);
        header.set_description(&v).unwrap(); // Manual fix
        assert_eq!(new_header, header);
        assert_eq!(new_data, data);

        // set_description
        header.set_description(&"ひらがな".as_bytes().to_vec()).unwrap();
        write_nifti(&path, &data, Some(&header)).unwrap();
        let (new_header, new_data) = read_as_ndarray(&path);
        assert_eq!(new_header, header);
        assert_eq!(new_data, data);

        // set_description_str
        header.set_description_str(&"русский").unwrap();
        write_nifti(&path, &data, Some(&header)).unwrap();
        let (new_header, new_data) = read_as_ndarray(&path);
        assert_eq!(new_header, header);
        assert_eq!(new_data, data);
    }

    #[should_panic]
    #[test]
    fn write_descrip_panic() {
        let dim = [3, 3, 4, 5, 1, 1, 1, 1];
        let mut header = generate_nifti_header(dim, 1.0, 0.0, NiftiType::Float32);
        header.descrip = (0..84).into_iter().collect();
        let path = get_temporary_path("error_description.nii");
        let data = Array::from_elem((3, 4, 5), 1.5);
        write_nifti(&path, &data, Some(&header)).unwrap();
    }

    #[should_panic]
    #[test]
    fn write_set_description_panic() {
        let dim = [3, 3, 4, 5, 1, 1, 1, 1];
        let mut header = generate_nifti_header(dim, 1.0, 0.0, NiftiType::Float32);
        header.set_description(&(0..81).into_iter().collect()).unwrap();
    }

    #[should_panic]
    #[test]
    fn write_set_description_str_panic() {
        let dim = [3, 3, 4, 5, 1, 1, 1, 1];
        let mut header = generate_nifti_header(dim, 1.0, 0.0, NiftiType::Float32);
        let description: String = (0..41).into_iter().map(|_| 'é').collect();
        header.set_description_str(&description).unwrap();
    }

    #[test]
    fn write_3d_rgb_hdr() {
        let mut data = Array::from_elem((3, 3, 3), [0u8, 0u8, 0u8]);
        data[(0, 0, 0)] = [55, 55, 0];
        data[(0, 0, 1)] = [55, 0, 55];
        data[(0, 1, 0)] = [0, 55, 55];

        let header_path = get_temporary_path("3d.hdr");
        let data_path = header_path.with_extension("img");
        write_rgb_nifti(&header_path, &data, None).unwrap();

        // Until we are able to read RGB images, we simply compare the bytes of the newly created
        // image to the bytes of the prepared 3D RGB image in ressources/rgb/. However, we need to
        // set the bytes of vox_offset to 0.0 and of magic to MAGIC_CODE_NI1. The data bytes should
        // be identical though.
        let mut gt_bytes = fs::read("resources/rgb/3D.nii").unwrap();
        for i in 110..114 {
            gt_bytes[i] = 0;
        }
        for i in 0..4 {
            gt_bytes[344 + i] = MAGIC_CODE_NI1[i];
        }
        assert_eq!(fs::read(header_path).unwrap(), &gt_bytes[..352]);
        assert_eq!(fs::read(data_path).unwrap(), &gt_bytes[352..]);
    }

    #[test]
    fn write_3d_rgb() {
        let mut data = Array::from_elem((3, 3, 3), [0u8, 0u8, 0u8]);
        data[(0, 0, 0)] = [55, 55, 0];
        data[(0, 0, 1)] = [55, 0, 55];
        data[(0, 1, 0)] = [0, 55, 55];

        let path = get_temporary_path("rgb.nii");
        write_rgb_nifti(&path, &data, None).unwrap();

        // Until we are able to read RGB images, we simply compare the bytes of the newly created
        // image to the bytes of the prepared 3D RGB image in ressources/rgb/.
        assert_eq!(
            fs::read(path).unwrap(),
            fs::read("resources/rgb/3D.nii").unwrap()
        );
    }

    #[test]
    fn write_4d_rgb() {
        let mut data = Array::from_elem((3, 3, 3, 2), [0u8, 0u8, 0u8]);
        data[(0, 0, 0, 0)] = [55, 55, 0];
        data[(0, 0, 1, 0)] = [55, 0, 55];
        data[(0, 1, 0, 0)] = [0, 55, 55];
        data[(0, 0, 0, 1)] = [55, 55, 0];
        data[(0, 1, 0, 1)] = [55, 0, 55];
        data[(1, 0, 0, 1)] = [0, 55, 55];

        let path = get_temporary_path("rgb.nii");
        write_rgb_nifti(&path, &data, None).unwrap();

        // Until we are able to read RGB images, we simply compare the bytes of the newly created
        // image to the bytes of the prepared 4D RGB image in ressources/rgb/.
        assert_eq!(
            fs::read(path).unwrap(),
            fs::read("resources/rgb/4D.nii").unwrap()
        );
    }
}
