#![no_main]
use libfuzzer_sys::fuzz_target;
use nifti::InMemNiftiObject;

fuzz_target!(|data: &[u8]| {
    let _ = InMemNiftiObject::from_reader(data);
});
