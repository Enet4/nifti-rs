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
//! ```
//!
//! The library will automatically look for the respective volume when
//! specifying just the header file:
//!
//! ```no_run
//! use nifti::{NiftiObject, InMemNiftiObject};
//! # use nifti::error::Result;
//! # fn run() -> Result<()> {
//! let obj = InMemNiftiObject::from_file("myvolume.hdr.gz")?;
//! # Ok(())
//! # }
//! ```
//!
//! You can also convert a volume to an [`ndarray::Array`](https://docs.rs/ndarray/0.9.1/ndarray/index.html)
//! and work from there:
//!
//! ```no_run
//! # #[cfg(feature = "ndarray_volumes")]
//! # fn run() {
//! # use nifti::{NiftiObject, InMemNiftiObject};
//! # let obj = InMemNiftiObject::from_file("myvolume.hdr.gz").unwrap();
//! let volume = obj.into_volume().to_ndarray::<f32>();
//! # }
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
pub use typedef::{NiftiType, Unit, Intent, XForm, SliceOrder};
pub use util::Endianness;
