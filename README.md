# NIFTI-rs &emsp; [![Latest Version](https://img.shields.io/crates/v/nifti.svg)](https://crates.io/crates/nifti) [![Continuous integration status](https://github.com/Enet4/nifti-rs/actions/workflows/rust.yml/badge.svg?branch=master)](https://github.com/Enet4/nifti-rs/actions/workflows/rust.yml) [![dependency status](https://deps.rs/repo/github/Enet4/nifti-rs/status.svg)](https://deps.rs/repo/github/Enet4/nifti-rs)

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

With the `ndarray_volumes` feature (enabled by default),
you can also convert a volume to an [`ndarray::Array`] and work from there:

```rust
let volume = obj.into_volume().into_ndarray::<f32>();
```

In addition, the `nalgebra_affine` feature unlocks the `affine` module,
for useful affine transformations.

[`ndarray::Array`]: https://docs.rs/ndarray/0.15.1/ndarray/index.html

## Roadmap

This library should hopefully fulfil a good number of use cases.
However, not all features of the format are fully available.
There are no deadlines for these features, so your help is much appreciated.
Please visit the [issue tracker](https://github.com/Enet4/nifti-rs/issues) and [tracker for version 1.0](https://github.com/Enet4/nifti-rs/issues/62).
In case something is missing for your use case to work,
please find an equivalent issue of file a new one.
Pull requests are also welcome.

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
