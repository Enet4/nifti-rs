//! Rust implementation of the NIfTI-1 file format.
//!
//! # Example
//!
//! ```no_run
//! use nifti::{NiftiObject, InMemNiftiObject, NiftiVolume};
//! # use nifti::error::Result;
//! 
//! # fn run() -> Result<()> {
//! let obj = InMemNiftiObject::from_file("myvolume.nii.gz")?;
//! // use obj
//! let header = obj.header();
//! let volume = obj.volume();
//! let dims = volume.dim();
//! # Ok(())
//! # }
//! run().unwrap();
//! ```
//! 
#![deny(missing_debug_implementations)]
#![warn(missing_docs, unused_extern_crates, trivial_casts, unused_results)]

#[macro_use] extern crate quick_error;
#[macro_use] extern crate num_derive;
#[macro_use] extern crate derive_builder;
#[cfg(feature = "ndarray_volumes")] extern crate ndarray;

#[cfg(ndarray_volumes)]
extern crate ndarray;

extern crate num;
extern crate byteorder;
extern crate flate2;

pub mod extension;
pub mod header;
pub mod object;
pub mod volume;
pub mod error;
pub mod typedef;
mod util;

pub use object::{NiftiObject, InMemNiftiObject};
pub use extension::{Extender, Extension, ExtensionSequence};
pub use header::NiftiHeader;
pub use volume::{NiftiVolume, InMemNiftiVolume};
pub use typedef::NiftiType;
pub use util::Endianness;
