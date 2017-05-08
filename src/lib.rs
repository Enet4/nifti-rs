extern crate byteorder;
extern crate acc_reader;
extern crate seek_bufread;
#[macro_use]
extern crate quick_error;
extern crate flate2;

pub mod header;
pub mod object;
pub mod volume;
pub mod error;
pub mod typedef;
mod util;

use std::fs::File;
use std::path::Path;
use util::ReadSeek;

use header::MAGIC_CODE_NI1;

pub use header::NiftiHeader;
pub use volume::NiftiVolume;
pub use util::Endianness;

#[cfg(test)]
mod tests {
}
