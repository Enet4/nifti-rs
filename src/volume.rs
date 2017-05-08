use std::io::BufRead;
use header::NiftiHeader;
use error::Result;
use util::Endianness;

pub trait NiftiVolume {
    /// Get the dimensions of the volume. Unlike how NIFTI-1
    /// stores dimensions, the returned slice does not include
    /// `dim[0]` and is clipped to the effective number of dimensions.
    fn dim(&self) -> &[i16];

    /// Fetch a single voxel's value in the given coordinates.
    /// All necessary conversions and transformations are made
    /// when reading the voxel, including scaling.
    fn get_f32(&self, coords: &[u16]) -> Result<f32>;

    // TODO
}

#[derive(Debug)]
pub struct InMemNiftiVolume {
    dim: [i16; 8],
    scl_slope: f32,
    scl_inter: f32,
    raw_data: Vec<u8>,
    endianness: Endianness,
}

impl InMemNiftiVolume {
    pub fn from_stream<R: BufRead>(header: &NiftiHeader, mut source: R, endianness: Endianness) -> Result<Self> {
        let ndims = header.dim[0];
        let resolution: usize = header.dim[1..(ndims+1) as usize].iter()
            .map(|d| *d as usize)
            .product();
        let nbytes = resolution * header.bitpix as usize / 8;
        let mut raw_data = vec![0u8; nbytes];
        source.read_exact(raw_data.as_mut_slice())?;

        Ok(InMemNiftiVolume {
            dim: header.dim,
            scl_slope: header.scl_slope,
            scl_inter: header.scl_inter,
            raw_data,
            endianness,
        })
    }
}

impl NiftiVolume for InMemNiftiVolume {
    fn dim(&self) -> &[i16] {
        &self.dim[1..(self.dim[0] + 1) as usize]
    }

    /// Fetch a single voxel's value in the given coordinates.
    fn get_f32(&self, coords: &[u16]) -> Result<f32> {
        unimplemented!()
    }
}