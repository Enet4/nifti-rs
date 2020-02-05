use nifti::{Endianness, NiftiHeader, NiftiType};

/// Known meta-data for the "minimal.nii" test file.
#[allow(dead_code)]
pub fn minimal_header_nii_gt() -> NiftiHeader {
    NiftiHeader {
        vox_offset: 352.,
        magic: *b"n+1\0",
        ..minimal_header_hdr_gt()
    }
}

/// Known meta-data for the "minimal.hdr" test file.
pub fn minimal_header_hdr_gt() -> NiftiHeader {
    NiftiHeader {
        sizeof_hdr: 348,
        dim: [3, 64, 64, 10, 0, 0, 0, 0],
        datatype: NiftiType::Uint8 as i16,
        bitpix: 8,
        pixdim: [0., 3., 3., 3., 0., 0., 0., 0.],
        srow_x: [0.; 4],
        srow_y: [0.; 4],
        srow_z: [0.; 4],
        vox_offset: 0.,
        scl_slope: 0.,
        scl_inter: 0.,
        sform_code: 0,
        qform_code: 0,
        magic: *b"ni1\0",
        endianness: Endianness::Big,
        ..Default::default()
    }
}

/// Known meta-data for the RGB volume test file.
#[allow(dead_code)]
pub fn rgb_header_gt() -> NiftiHeader {
    NiftiHeader {
        datatype: NiftiType::Rgb24 as i16,
        sform_code: 2,
        qform_code: 0,
        endianness: Endianness::Little,
        ..NiftiHeader::default()
    }
}
