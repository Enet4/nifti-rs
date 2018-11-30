#[cfg(feature = "ndarray_volumes")]
extern crate ndarray;
#[cfg(feature = "ndarray_volumes")]
extern crate nifti;
#[cfg(feature = "ndarray_volumes")]
extern crate tempfile;

#[cfg(feature = "ndarray_volumes")]
mod tests {
    use std::{
        fs::{self, File},
        io::Read,
        ops::{Add, Mul},
        path::{Path, PathBuf},
    };

    use ndarray::{Array, Array2, Axis, Dimension, IxDyn, ShapeBuilder};
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
        datatype: i16,
    ) -> NiftiHeader {
        NiftiHeader {
            dim,
            datatype,
            bitpix: (NiftiType::Float32.size_of() * 8) as i16,
            magic: *MAGIC_CODE_NIP1,
            scl_slope,
            scl_inter,
            ..NiftiHeader::default()
        }
    }

    fn read_as_ndarray<P, D>(path: P) -> Array<f32, D>
    where
        P: AsRef<Path>,
        D: Dimension,
    {
        let nifti_object = InMemNiftiObject::from_file(path).expect("Nifti file is unreadable.");
        let volume = nifti_object.into_volume();
        let dyn_data = volume.into_ndarray().unwrap();
        dyn_data.into_dimensionality::<D>().unwrap()
    }

    fn test_write_read(arr: &Array<f32, IxDyn>, path: &str) {
        let path = get_temporary_path(path);
        let mut dim = [1; 8];
        dim[0] = arr.ndim() as u16;
        for (i, s) in arr.shape().iter().enumerate() {
            dim[i + 1] = *s as u16;
        }
        let header = generate_nifti_header(dim, 1.0, 0.0, 16);
        write_nifti(&path, &arr, Some(&header)).unwrap();

        let read_nifti: Array2<f32> = read_as_ndarray(path);
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
        let header = generate_nifti_header(dim, slope, inter, 16);
        let transformed_data = arr.mul(slope).add(inter);
        write_nifti(&path, &transformed_data, Some(&header)).unwrap();

        let read_nifti: Array2<f32> = read_as_ndarray(path);
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
            let data_read = read_as_ndarray(path);
            assert_eq!(data, data_read);
        }
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
        let mut rgb_bytes = vec![];
        let _ = File::open(path)
            .unwrap()
            .read_to_end(&mut rgb_bytes)
            .unwrap();
        let mut gt_bytes = vec![];
        let _ = File::open("resources/rgb/3D.nii")
            .unwrap()
            .read_to_end(&mut gt_bytes)
            .unwrap();
        assert_eq!(rgb_bytes, gt_bytes);
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
        let mut rgb_bytes = vec![];
        let _ = File::open(path)
            .unwrap()
            .read_to_end(&mut rgb_bytes)
            .unwrap();
        let mut gt_bytes = vec![];
        let _ = File::open("resources/rgb/4D.nii")
            .unwrap()
            .read_to_end(&mut gt_bytes)
            .unwrap();
        assert_eq!(rgb_bytes, gt_bytes);
    }
}
