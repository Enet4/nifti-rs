use util::ReadSeek;
use header::NIFTIHeader;

#[derive(Debug)]
pub struct NIFTIVolume<R: ReadSeek> {
    dim: [i16; 8],
    scl_slope: f32,
    scl_inter: f32,
    source: R,
}

impl<'h, R: ReadSeek> NIFTIVolume<R> {
    pub fn new(header: &'h NIFTIHeader, source: R) -> Self {
        NIFTIVolume {
            dim: header.dim,
            scl_slope: header.scl_slope,
            scl_inter: header.scl_inter,
            source: source,
        }
    }
}