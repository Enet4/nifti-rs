# NIFTI-rs &emsp; [![Latest Version](https://img.shields.io/crates/v/nifti.svg)](https://crates.io/crates/nifti) [![Build Status](https://travis-ci.org/Enet4/nifti-rs.svg?branch=master)](https://travis-ci.org/Enet4/nifti-rs)

This library is a pure Rust implementation for reading files in the [NIfTI](https://nifti.nimh.nih.gov/nifti-1/) format (more specifically NIfTI-1.1).

## Example

```rust
use nifti::{NiftiObject, InMemNiftiObject, NiftiVolume};
 
let obj = InMemNiftiObject::with_file("myvolume.nii.gz")?;
// use obj
let header = obj.header();
let volume = obj.volume();
let dims = volume.dim();
```

The library will automatically look for the respective volume when
specifying just the header file:

```rust
use nifti::{NiftiObject, InMemNiftiObject};

let obj = InMemNiftiObject::with_file("myvolume.hdr.gz")?;
```

With the "ndarray_volumes" feature enabled, you can also convert a volume to an [`ndarray::Array`](https://docs.rs/ndarray/0.9.1/ndarray/index.html) and work from there:

```rust
let volume = obj.into_volume().to_ndarray::<f32>();
```

## Roadmap

This library should hopefully fulfil a good number of use cases. However, it still is a bit far
from a complete solution. In particular, future versions should be able to:

- Write NIFTI files;
- Provide a more elegant volume API;
- Handle more kinds of volumes;
- Provide a real spatial-temporal interpretation of the volume (rather than just voxel-indexed);
- Maybe add support for NIFTI-2?

There are no deadlines for these features, so your help is much appreciated. Consider filing an [issue](https://github.com/Enet4/nifti-rs/issues) in case something is missing for your use case to work. Pull requests are also welcome.

## License

Apache-2/MIT
