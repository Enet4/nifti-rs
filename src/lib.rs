//! Rust implementation of the NIfTI-1 file format.
//!
//! # Example
//!
//! ```no_run
//! use nifti::{NiftiObject, ReaderOptions, NiftiVolume};
//! # use nifti::error::Result;
//!
//! # fn run() -> Result<()> {
//! let obj = ReaderOptions::new().read_file("myvolume.nii.gz")?;
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
//! use nifti::{NiftiObject, ReaderOptions};
//! # use nifti::error::Result;
//! # fn run() -> Result<()> {
//! let obj = ReaderOptions::new().read_file("myvolume.hdr.gz")?;
//! # Ok(())
//! # }
//! ```
//!
//! With the `ndarray_volumes` Cargo feature enabled, you can also convert a
//! volume to an [`ndarray::Array`](https://docs.rs/ndarray) and work from
//! there:
//!
//! ```no_run
//! # #[cfg(feature = "ndarray_volumes")]
//! # use nifti::error::Result;
//! # #[cfg(feature = "ndarray_volumes")]
//! # fn run() -> Result<()> {
//! # use nifti::{NiftiObject, ReaderOptions};
//! use nifti::IntoNdArray;
//! # let obj = ReaderOptions::new().read_file("myvolume.hdr.gz").unwrap();
//! let volume = obj.into_volume().into_ndarray::<f32>()?;
//! # Ok(())
//! # }
//! ```
//!
//! An additional volume API is also available for reading large volumes slice
//! by slice.
//!
//! ```no_run
//! # use nifti::{NiftiObject, ReaderStreamedOptions};
//! # use nifti::error::NiftiError;
//! let obj = ReaderStreamedOptions::new().read_file("minimal.nii.gz")?;
//!
//! let volume = obj.into_volume();
//! for slice in volume {
//!     let slice = slice?;
//!     // manipulate slice here
//! }
//! # Ok::<(), NiftiError>(())
//! ```
//!
#![deny(missing_debug_implementations)]
#![warn(missing_docs, unused_extern_crates, trivial_casts, unused_results)]
#![allow(clippy::unit_arg)]
#![recursion_limit = "128"]

#[cfg(all(test, feature = "nalgebra_affine"))]
#[macro_use]
extern crate approx;

#[cfg(feature = "nalgebra_affine")]
pub mod affine;
pub mod error;
pub mod extension;
pub mod header;
pub mod object;
pub mod typedef;
mod util;
pub mod volume;
#[cfg(feature = "ndarray_volumes")]
pub mod writer;

pub use byteordered::Endianness;
pub use error::{NiftiError, Result};
pub use extension::{Extender, Extension, ExtensionSequence};
pub use header::{NiftiHeader, Nifti1Header, Nifti2Header};
pub use object::{
    InMemNiftiObject, NiftiObject, ReaderOptions, ReaderStreamedOptions, StreamedNiftiObject,
};
pub use typedef::{Intent, NiftiType, SliceOrder, Unit, XForm};
pub use volume::element::DataElement;
#[cfg(feature = "ndarray_volumes")]
pub use volume::ndarray::IntoNdArray;
pub use volume::{
    InMemNiftiVolume, NiftiVolume, RandomAccessNiftiVolume, Sliceable, StreamedNiftiVolume,
};
