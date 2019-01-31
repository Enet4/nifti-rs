#[cfg(feature = "nalgebra_affine")] extern crate nalgebra;
#[cfg(feature = "nalgebra_affine")] extern crate nifti;

#[cfg(feature = "nalgebra_affine")]
mod nalgebra_affine {
    use nalgebra::Vector4;
    use nifti::{affine::Affine4, NiftiHeader};

        #[test]
    fn affine() {
        let mut header = NiftiHeader::default();
        let affine = Affine4::from_diagonal(&Vector4::new(2.0, 2.0, 2.0, 1.0));
        header.set_affine(&affine);
        assert_eq!(affine, header.affine());
        assert_eq!(header.sform_code, 2);
        assert_eq!(header.qform_code, 0);
    }

    #[test]
    fn sform() {
        let mut header = NiftiHeader::default();
        header.sform_code = 1;
        header.qform_code = 0;
        header.srow_x = [2.4, -0.0008, -0.0411765, -114.766396];
        header.srow_y = [0.1, 2.4995277, 0.0485984, -97.420204];
        header.srow_z = [0.4, -0.0485, 2.4991884, -89.12282];

        assert_eq!(header.affine(), Affine4::new(
            2.4, -0.0008,    -0.0411765, -114.766396,
            0.1,  2.4995277,  0.0485984,  -97.420204,
            0.4, -0.0485,     2.4991884,  -89.12282,
            0.0, 0.0, 0.0, 1.0
        ));
    }

    #[test]
    fn qform() {
        let mut header = NiftiHeader::default();
        header.sform_code = 0;
        header.qform_code = 1;
        header.pixdim = [-1.0, 0.9375, 0.9375, 3.0, 0.0, 0.0, 0.0, 0.0];
        header.quatern_b = 0.0;
        header.quatern_c = 1.0;
        header.quatern_d = 0.0;
        header.quatern_x = 59.557503;
        header.quatern_y = 73.172;
        header.quatern_z = 43.4291;

        assert_eq!(header.affine(), Affine4::new(
            -0.9375, 0.0,    0.0, 59.557503,
            0.0,     0.9375, 0.0, 73.172,
            0.0,     0.0,    3.0, 43.4291,
            0.0,     0.0,    0.0, 1.0
        ));
    }

    #[test]
    fn both_valid() {
        let mut header = NiftiHeader::default();
        header.sform_code = 1;
        header.srow_x = [2.4, 0.0, 0.0, -114.766396];
        header.srow_y = [0.1, 2.4, 0.0, -97.420204];
        header.srow_z = [0.4, 0.4, 2.4, -89.12282];

        // All this should be ignored
        header.qform_code = 1;
        header.pixdim = [-1.0, 0.9375, 0.9375, 3.0, 0.0, 0.0, 0.0, 0.0];
        header.quatern_b = 0.0;
        header.quatern_c = 1.0;
        header.quatern_d = 0.0;
        header.quatern_x = 59.0;
        header.quatern_y = 73.0;
        header.quatern_z = 43.0;

        assert_eq!(header.affine(), Affine4::new(
            2.4, 0.0, 0.0, -114.766396,
            0.1, 2.4, 0.0, -97.420204,
            0.4, 0.4, 2.4, -89.12282,
            0.0, 0.0, 0.0, 1.0
        ));
    }

    #[test]
    fn none_valid() {
        let mut header = NiftiHeader::default();
        header.dim = [3, 100, 100, 100, 0, 0, 0, 0];
        header.pixdim = [-1.0, 0.9, 0.9, 3.0, 0.0, 0.0, 0.0, 0.0];

        // All this should be ignored, because only `shape_zoom_affine` will be called.
        header.sform_code = 0;
        header.srow_x = [1.0, 0.0, 0.0, 1.0];
        header.srow_y = [0.0, 1.0, 0.0, 1.0];
        header.srow_z = [0.0, 0.0, 1.0, 1.0];
        header.qform_code = 0;
        header.quatern_b = 0.0;
        header.quatern_c = 1.0;
        header.quatern_d = 0.0;
        header.quatern_x = 59.0;
        header.quatern_y = 73.0;
        header.quatern_z = 43.0;

        assert_eq!(header.affine(), Affine4::new(
            -0.9, 0.0, 0.0,   44.55,
            0.0,  0.9, 0.0,  -44.55,
            0.0,  0.0, 3.0, -148.5,
            0.0, 0.0, 0.0, 1.0
        ));
    }
}
