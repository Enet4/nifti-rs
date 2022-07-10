#[cfg(feature = "ndarray_volumes")]
extern crate ndarray;
#[cfg(feature = "ndarray_volumes")]
extern crate nifti;
#[cfg(feature = "ndarray_volumes")]
extern crate num_traits;
#[cfg(feature = "ndarray_volumes")]
extern crate tempfile;

mod util;

#[cfg(feature = "ndarray_volumes")]
mod tests {
    use std::{
        fs,
        ops::{Add, Mul},
        path::{Path, PathBuf},
    };

    use approx::assert_abs_diff_eq;
    use ndarray::{
        s, Array, Array1, Array2, Array3, Array4, Array5, Axis, Dimension, Ix2, IxDyn, ShapeBuilder,
    };
    use tempfile::tempdir;

    use nifti::{
        header::{MAGIC_CODE_NI1, MAGIC_CODE_NIP1},
        object::NiftiObject,
        volume::shape::Dim,
        writer::WriterOptions,
        DataElement, IntoNdArray, Nifti1Header, NiftiHeader, NiftiType, ReaderOptions,
    };

    use super::util::rgb_header_gt;

    fn get_temporary_path(ext: &str) -> PathBuf {
        let dir = tempdir().unwrap();
        let mut path = dir.into_path();
        if !ext.is_empty() {
            path.push(ext);
        }
        path
    }

    pub fn generate_nifti1_header(
        dim: [u16; 8],
        scl_slope: f32,
        scl_inter: f32,
        datatype: NiftiType,
    ) -> Nifti1Header {
        Nifti1Header {
            dim,
            datatype: datatype as i16,
            bitpix: (datatype.size_of() * 8) as u16,
            magic: *MAGIC_CODE_NIP1,
            scl_slope,
            scl_inter,
            ..Nifti1Header::default()
        }
    }

    fn read_as_ndarray<P, T, D>(path: P) -> (NiftiHeader, Array<T, D>)
    where
        P: AsRef<Path>,
        T: Mul<Output = T>,
        T: Add<Output = T>,
        T: DataElement,
        D: Dimension,
    {
        let nifti_object = ReaderOptions::new()
            .read_file(path)
            .expect("Nifti file is unreadable.");
        let header = nifti_object.header().clone();
        let volume = nifti_object.into_volume();
        let dyn_data = volume.into_ndarray::<T>().unwrap();
        (header, dyn_data.into_dimensionality::<D>().unwrap())
    }

    fn test_write_read(arr: Array<f32, IxDyn>, path: &str) {
        let path = get_temporary_path(path);
        let dim = *Dim::from_slice(arr.shape()).unwrap().raw();
        let header =
            generate_nifti1_header(dim.map(|d| d as u16), 1.0, 0.0, NiftiType::Float32).into();
        WriterOptions::new(&path)
            .reference_header(&header)
            .write_nifti(&arr)
            .unwrap();

        let gt = arr.into_dimensionality::<Ix2>().unwrap();
        let read_nifti: Array2<f32> = read_as_ndarray(path).1;
        assert_abs_diff_eq!(read_nifti, gt, epsilon = 1e-10);
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
        test_write_read(arr, "test.nii");
        let mut arr = f_order_array();
        arr.invert_axis(Axis(1));
        test_write_read(arr, "test_non_contiguous.nii");

        // Test .nii.gz
        let arr = f_order_array();
        test_write_read(arr, "test.nii.gz");
        let mut arr = f_order_array();
        arr.invert_axis(Axis(1));
        test_write_read(arr, "test_non_contiguous.nii.gz");
    }

    #[test]
    fn c_writing() {
        // Test .nii
        let arr = c_order_array();
        test_write_read(arr, "test.nii");

        let mut arr = c_order_array();
        arr.invert_axis(Axis(1));
        test_write_read(arr, "test_non_contiguous.nii");

        // Test .nii.gz
        let arr = c_order_array();
        test_write_read(arr, "test.nii.gz");

        let mut arr = c_order_array();
        arr.invert_axis(Axis(1));
        test_write_read(arr, "test_non_contiguous.nii.gz");
    }

    #[test]
    fn header_slope_inter() {
        let arr = f_order_array();
        let slope = 2.2;
        let inter = 101.1;

        let path = get_temporary_path("test_slope_inter.nii");
        let dim = *Dim::from_slice(arr.shape()).unwrap().raw();
        let header =
            generate_nifti1_header(dim.map(|d| d as u16), slope, inter, NiftiType::Float32).into();
        let transformed_data = arr.mul(slope).add(inter);
        WriterOptions::new(&path)
            .reference_header(&header)
            .write_nifti(&transformed_data)
            .unwrap();

        let gt = transformed_data.into_dimensionality::<Ix2>().unwrap();
        let read_nifti: Array2<f32> = read_as_ndarray(path).1;
        assert_abs_diff_eq!(read_nifti, gt, epsilon = 1e-10);
    }

    #[test]
    fn half_slope() {
        let data = (0..216)
            .collect::<Array1<_>>()
            .into_shape((6, 6, 6))
            .unwrap();
        let dim = [3, 6, 6, 6, 1, 1, 1, 1];
        let slope = 0.4;
        let inter = 100.1;
        let header = generate_nifti1_header(dim, slope, inter, NiftiType::Uint8).into();

        let path = get_temporary_path("test_slope_inter.nii");
        WriterOptions::new(&path)
            .reference_header(&header)
            .write_nifti(&data)
            .unwrap();

        let (read_header, read_data) = read_as_ndarray(path);
        assert_eq!(read_header.get_scl_inter(), 0.0);
        assert_eq!(read_header.get_scl_slope(), 1.0);
        assert_eq!(data, read_data);
    }

    #[test]
    fn write_hdr_standard() {
        let mut data = Array::zeros((10, 11, 12));
        data[(5, 0, 0)] = 1.0;
        data[(6, 0, 0)] = 2.0;

        for fname in &["3d.hdr", "3d.hdr.gz"] {
            let path = get_temporary_path(fname);
            WriterOptions::new(&path).write_nifti(&data).unwrap();
            let data_read = read_as_ndarray(path).1;
            assert_eq!(data, data_read);
        }
    }

    #[test]
    fn write_non_contiguous() {
        let mut data = Array::from_elem((3, 4, 11), 1.5);
        data.slice_mut(s![.., .., ..;2]).fill(42.0);

        let path = get_temporary_path("non_contiguous_0.nii.gz");
        WriterOptions::new(&path)
            .write_nifti(&data.slice(s![.., .., ..;2]))
            .unwrap();
        let loaded_data = read_as_ndarray::<_, f32, _>(path).1;
        assert_eq!(loaded_data, Array::from_elem((3, 4, 6), 42.0));

        let path = get_temporary_path("non_contiguous_1.nii.gz");
        WriterOptions::new(&path)
            .write_nifti(&data.slice(s![.., .., 1..;2]))
            .unwrap();
        let loaded_data = read_as_ndarray::<_, f32, _>(path).1;
        assert_eq!(loaded_data, Array::from_elem((3, 4, 5), 1.5));
    }

    #[test]
    fn write_wrong_description() {
        let dim = [3, 3, 4, 5, 1, 1, 1, 1];
        let header: NiftiHeader = generate_nifti1_header(dim, 1.0, 0.0, NiftiType::Float32).into();
        let path = get_temporary_path("error_description.nii");
        let data = Array::from_elem((3, 4, 5), 1.5);

        // Manual descrip. The original header won't be "repaired", but the written description
        // should be right. To compare the header, we must fix it ourselves.
        let v = "äbcdé".as_bytes();
        let mut header = header.into_nifti1().unwrap();
        header.descrip[..v.len()].copy_from_slice(&v);
        let mut header = header.into();
        WriterOptions::new(&path)
            .reference_header(&header)
            .write_nifti(&data)
            .unwrap();
        let (new_header, new_data) = read_as_ndarray::<_, f32, _>(&path);
        header.set_description(v).unwrap(); // Manual fix
        assert_eq!(new_header, header);
        assert_eq!(new_data, data);

        // set_description
        header.set_description("ひらがな".as_bytes()).unwrap();
        WriterOptions::new(&path)
            .reference_header(&header)
            .write_nifti(&data)
            .unwrap();
        let (new_header, new_data) = read_as_ndarray::<_, f32, _>(&path);
        assert_eq!(new_header, header);
        assert_eq!(new_data, data);

        // set_description_str
        header.set_description_str("русский").unwrap();
        WriterOptions::new(&path)
            .reference_header(&header)
            .write_nifti(&data)
            .unwrap();
        let (new_header, new_data) = read_as_ndarray::<_, f32, _>(&path);
        assert_eq!(new_header, header);
        assert_eq!(new_data, data);
    }

    #[test]
    fn write_set_description_panic() {
        let dim = [3, 3, 4, 5, 1, 1, 1, 1];
        let mut header: NiftiHeader =
            generate_nifti1_header(dim, 1.0, 0.0, NiftiType::Float32).into();
        assert!(header
            .set_description((0..81).into_iter().collect::<Vec<_>>())
            .is_err());
    }

    #[test]
    fn write_set_description_str_panic() {
        let dim = [3, 3, 4, 5, 1, 1, 1, 1];
        let mut header: NiftiHeader =
            generate_nifti1_header(dim, 1.0, 0.0, NiftiType::Float32).into();
        let description: String = std::iter::repeat('é').take(41).collect();
        assert!(header.set_description_str(description).is_err());
    }

    #[test]
    fn write_3d_only_1_slice() {
        // See issue #63
        let mut data = Array3::zeros((5, 5, 3));
        data.slice_mut(s![.., 2, 0]).fill(1.0);
        data.slice_mut(s![.., 3, 0]).fill(1.1);

        let path = get_temporary_path("3d_1s.nii.gz");
        WriterOptions::new(&path)
            .write_nifti(&data.select(Axis(2), &[0]))
            .unwrap();
        let loaded: Array3<f32> = read_as_ndarray(path).1;
        assert_eq!(data.index_axis(Axis(2), 0), loaded.index_axis(Axis(2), 0));
    }

    #[test]
    fn write_4d_only_1_volume() {
        // See issue #63
        let mut data = Array4::zeros((5, 5, 4, 3));
        data.slice_mut(s![.., 2, 2, 0]).fill(1.0);
        data.slice_mut(s![.., 3, 2, 0]).fill(1.1);

        let path = get_temporary_path("4d_1v.nii.gz");
        WriterOptions::new(&path)
            .write_nifti(&data.select(Axis(3), &[0]))
            .unwrap();
        let loaded: Array4<f32> = read_as_ndarray(path).1;
        assert_eq!(data.index_axis(Axis(3), 0), loaded.index_axis(Axis(3), 0));
    }

    #[test]
    fn write_5d_only_1_slice() {
        // See issue #63
        let mut data = Array5::zeros((5, 5, 4, 1, 3));
        data.slice_mut(s![.., 2, 2, 0, 0]).fill(1.0);
        data.slice_mut(s![.., 3, 2, 0, 0]).fill(1.1);

        let path = get_temporary_path("5d_1s.nii.gz");
        WriterOptions::new(&path)
            .write_nifti(&data.select(Axis(4), &[0]))
            .unwrap();
        let loaded: Array5<f32> = read_as_ndarray(path).1;
        assert_eq!(data.index_axis(Axis(4), 0), loaded.index_axis(Axis(4), 0));
    }

    #[test]
    fn write_3d_rgb_hdr() {
        let mut data = Array::from_elem((3, 3, 3), [0u8, 0u8, 0u8]);
        data[(0, 0, 0)] = [55, 55, 0];
        data[(0, 0, 1)] = [55, 0, 55];
        data[(0, 1, 0)] = [0, 55, 55];

        let header_path = get_temporary_path("3d.hdr");
        let data_path = header_path.with_extension("img");
        let header = rgb_header_gt();
        WriterOptions::new(&header_path)
            .reference_header(&header)
            .write_rgb_nifti(&data)
            .unwrap();

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
        let header = rgb_header_gt();
        WriterOptions::new(&path)
            .reference_header(&header)
            .write_rgb_nifti(&data)
            .unwrap();

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
        let header = rgb_header_gt();
        WriterOptions::new(&path)
            .reference_header(&header)
            .write_rgb_nifti(&data)
            .unwrap();

        // Until we are able to read RGB images, we simply compare the bytes of the newly created
        // image to the bytes of the prepared 4D RGB image in ressources/rgb/.
        assert_eq!(
            fs::read(path).unwrap(),
            fs::read("resources/rgb/4D.nii").unwrap()
        );
    }
}
