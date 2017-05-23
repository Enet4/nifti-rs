# NIFTI-rs

[![Build Status](https://travis-ci.org/Enet4/nifti-rs.svg?branch=master)](https://travis-ci.org/Enet4/nifti-rs)

This library is a pure Rust implementation for reading files in the NIFTI-1 format.

It is currently a work in progress.


## Example

```rust
use nifti::{NiftiObject, InMemNiftiObject, NiftiVolume};
 
let obj = InMemNiftiObject::from_file("myvolume.nii.gz")?;
// use obj
let header = obj.header();
let volume = obj.volume();
let dims = volume.dim();
```

The library will automatically look for the respective volume when
specifying just the header file:

```rust
use nifti::{NiftiObject, InMemNiftiObject};

let obj = InMemNiftiObject::from_file("myvolume.hdr.gz")?;
```

With the "ndarray_volumes" feature enabled, you can also convert a volume to an [`ndarray::Array`](https://docs.rs/ndarray/0.9.1/ndarray/index.html) and work from there:

```rust
let volume = obj.into_volume().to_ndarray::<f32>();
```

# License

Apache-2/MIT
