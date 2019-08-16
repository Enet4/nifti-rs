#![no_main]
use libfuzzer_sys::fuzz_target;
use nifti::{NiftiObject, StreamedNiftiObject};

fuzz_target!(|data: &[u8]| {
    if let Ok(obj) = StreamedNiftiObject::from_reader(data) {
        for _slice in obj.into_volume() {}
    }
});
