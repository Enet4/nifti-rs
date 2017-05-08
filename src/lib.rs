//! Rust implementation of the NIfTI-1 file format.
//!
//! 
//! 
#![deny(missing_debug_implementations)]
#![warn(missing_docs, unused_extern_crates, trivial_casts, unused_results)]

#[macro_use] extern crate quick_error;
#[macro_use] extern crate num_derive;

#[cfg(ndarray_volumes)]
extern crate ndarray;

extern crate num;
extern crate byteorder;
extern crate flate2;

pub mod header;
pub mod object;
pub mod volume;
pub mod error;
pub mod typedef;
mod util;

pub use header::NiftiHeader;
pub use volume::NiftiVolume;
pub use volume::InMemNiftiVolume;
pub use util::Endianness;

#[cfg(test)]
mod tests {
}
