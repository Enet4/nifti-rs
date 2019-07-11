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
//! With the `ndarray_volumes` Cargo feature enabled, you can also convert a
//! volume to an [`ndarray::Array`](https://docs.rs/ndarray) and work from
//! there:
//!
//! ```no_run
//! # #[cfg(feature = "ndarray_volumes")]
//! # use nifti::error::Result;
//! # #[cfg(feature = "ndarray_volumes")]
//! # fn run() -> Result<()> {
//! # use nifti::{NiftiObject, InMemNiftiObject};
//! use nifti::IntoNdArray;
//! # let obj = InMemNiftiObject::from_file("myvolume.hdr.gz").unwrap();
//! let volume = obj.into_volume().into_ndarray::<f32>()?;
//! # Ok(())
//! # }
//! ```
//!
//! An additional volume API is also available for reading large volumes slice
//! by slice.
//! 
//! ```no_run
//! # use nifti::{NiftiObject, StreamedNiftiObject};
//! # use nifti::error::NiftiError;
//! let obj = StreamedNiftiObject::from_file("minimal.nii.gz")?;
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
#![recursion_limit = "128"]

#[cfg(all(test, feature = "nalgebra_affine"))] #[macro_use] extern crate approx;

#[cfg(feature = "nalgebra_affine")] pub mod affine;
pub mod extension;
pub mod header;
pub mod object;
pub mod volume;
pub mod error;
pub mod typedef;
#[cfg(feature = "ndarray_volumes")] pub mod writer;
mod util;

pub use error::{NiftiError, Result};
pub use object::{NiftiObject, InMemNiftiObject, StreamedNiftiObject};
pub use extension::{Extender, Extension, ExtensionSequence};
pub use header::NiftiHeader;
pub use volume::{NiftiVolume, RandomAccessNiftiVolume, InMemNiftiVolume, StreamedNiftiVolume, Sliceable};
pub use volume::element::DataElement;
#[cfg(feature = "ndarray_volumes")] pub use volume::ndarray::IntoNdArray;
pub use typedef::{NiftiType, Unit, Intent, XForm, SliceOrder};
pub use byteordered::Endianness;
