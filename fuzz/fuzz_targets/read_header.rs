#![no_main]
use libfuzzer_sys::fuzz_target;
use nifti::NiftiHeader;

fuzz_target!(|data: &[u8]| {
    if let Ok(header) = NiftiHeader::from_reader(data) {
        let _ = header.dim();
        let _ = header.data_type();
        let _ = header.qform();
        let _ = header.sform();
        let _ = header.intent();
        let _ = header.slice_order();
        let _ = header.xyzt();
        let _ = header.clone().validate_description();
    }
});
