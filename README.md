# NIFTI-rs &emsp; [![Latest Version](https://img.shields.io/crates/v/nifti.svg)](https://crates.io/crates/nifti) [![Build Status](https://travis-ci.org/Enet4/nifti-rs.svg?branch=master)](https://travis-ci.org/Enet4/nifti-rs) [![dependency status](https://deps.rs/repo/github/Enet4/nifti-rs/status.svg)](https://deps.rs/repo/github/Enet4/nifti-rs)

This library is a pure Rust implementation for reading files in the [NIfTI](https://nifti.nimh.nih.gov/nifti-1/) format (more specifically NIfTI-1.1).

## Example

Please see [the documentation](https://docs.rs/nifti) for more.

```rust
use nifti::{NiftiObject, ReaderOptions, NiftiVolume};

let obj = ReaderOptions::new().read_file("myvolume.nii.gz")?;
// use obj
let header = obj.header();
let volume = obj.volume();
let dims = volume.dim();
```

The library will automatically look for the respective volume when
specifying just the header file:

```rust
use nifti::{NiftiObject, ReaderOptions};

let obj = ReaderOptions::new().read_file("myvolume.hdr.gz")?;
```

With the `ndarray_volumes` feature enabled, you can also convert a volume to an [`ndarray::Array`](https://docs.rs/ndarray/0.12.0/ndarray/index.html) and work from there:

```rust
let volume = obj.into_volume().into_ndarray::<f32>();
```

## Roadmap

This library should hopefully fulfil a good number of use cases. However, some gaps
are yet to be filled fill. In particular, future versions should be able to:

- Write NIFTI files;
- Provide a more elegant volume API;
- Handle more kinds of volumes;
- Provide a real spatial-temporal interpretation of the volume (rather than just voxel-indexed);

There are no deadlines for these features, so your help is much appreciated. Consider filing an [issue](https://github.com/Enet4/nifti-rs/issues) in case something is missing for your use case to work. Pull requests are also welcome.

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
