use std::error::Error;

pub type NiftiError = Box<Error>;

pub type Result<T> = ::std::result::Result<T, NiftiError>;
