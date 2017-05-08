#[derive(Debug, PartialEq, Clone, Copy)]
pub struct NiftiExtender {
    extension: [u8; 4],
}

#[derive(Debug, PartialEq, Clone)]
pub struct NiftiExtension {
    esize: i32,
    ecode: i32,
    edata: Vec<u8>,
}
