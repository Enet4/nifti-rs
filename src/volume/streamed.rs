//! Streamed interface of a NIfTI volume and implementation.
//!
//! This API provides slice-by-slice reading of volumes, thus lowering
//! memory requirements and better supporting the manipulation of
//! large volumes.

use super::inmem::InMemNiftiVolume;
use super::NiftiVolume;
use error::Result;
use header::NiftiHeader;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use typedef::{Dim, NiftiType};
use util::nb_bytes_for_dim_datatype;
use byteordered::Endianness;

/// A NIfTI-1 volume instance that is read slice by slice from a byte stream.
///
/// Since volumes are physically persisted in column major order, each slice
/// will cover the full range of the first axes of the volume, thus traversing
/// the rightmost axes in each iteration. As an example, a volume of dimensions
/// `[256, 256, 128]` may produce slices of dimensions `[256, 256]`.
///
/// Slices may also have an arbitrary rank (dimensionality), as long as it
/// is smaller than the original volume's rank. A good default is `R - 1`.
///

#[derive(Debug)]
pub struct StreamedNiftiVolume<R> {
    source: R,
    dim: Dim,
    slice_dim: Dim,
    datatype: NiftiType,
    scl_slope: f32,
    scl_inter: f32,
    endianness: Endianness,
    slices_read: usize,
    slices_left: usize,
}

impl StreamedNiftiVolume<BufReader<File>> {
    /// Read a NIFTI volume from an uncompressed file. The header and expected
    /// byte order of the volume's data must be known in advance. It is also
    /// expected that the file starts with the first voxels of the volume, not
    /// with the extensions.
    pub fn from_file<P>(path: P, header: &NiftiHeader) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        let file = BufReader::new(File::open(path)?);
        Self::from_reader(file, header)
    }
}

impl<R> StreamedNiftiVolume<R>
where
    R: Read,
{
    /// Read a NIFTI volume from a stream of raw voxel data. The header and
    /// expected byte order of the volume's data must be known in advance. It
    /// is also expected that the following bytes represent the first voxels of
    /// the volume (and not part of the extensions).
    ///
    /// By default, the slice's rank is the original volume's rank minus 1.
    pub fn from_reader(source: R, header: &NiftiHeader) -> Result<Self> {
        let dim = Dim::new(header.dim)?;
        let slice_rank = dim.rank() - 1;
        StreamedNiftiVolume::from_reader_rank(source, header, slice_rank as u16)
    }

    /// Read a NIFTI volume from a stream of data. The header and expected byte order
    /// of the volume's data must be known in advance. It it also expected that the
    /// following bytes represent the first voxels of the volume (and not part of the
    /// extensions).
    ///
    /// The slice rank defines how many dimensions each slice should have.
    pub fn from_reader_rank(source: R, header: &NiftiHeader, slice_rank: u16) -> Result<Self> {
        // TODO recoverable error if #dim == 0
        let dim = Dim::new(header.dim)?; // check dim consistency
        let datatype = header.data_type()?;
        let slice_dim = calculate_slice_dims(&dim, slice_rank);
        let slices_left = calculate_total_slices(&dim, slice_rank);
        Ok(StreamedNiftiVolume {
            source,
            dim,
            slice_dim,
            datatype,
            scl_slope: header.scl_slope,
            scl_inter: header.scl_inter,
            endianness: header.endianness,
            slices_read: 0,
            slices_left,
        })
    }

    /// Retrieve the full volume shape.
    pub fn dim(&self) -> &[u16] {
        self.dim.as_ref()
    }

    /// Retrieve the shape of the slices.
    pub fn slice_dim(&self) -> &[u16] {
        self.slice_dim.as_ref()
    }

    /// Retrieve the number of slices already read
    pub fn slices_read(&self) -> usize {
        self.slices_read
    }

    /// Retrieve the number of slices left
    pub fn slices_left(&self) -> usize {
        self.slices_left
    }

    /// Read a volume slice from the data source, producing an in-memory
    /// sub-volume.
    pub fn read_slice(&mut self) -> Result<InMemNiftiVolume> {
        self.read_slice_inline(Vec::new())
    }

    /// Read a volume slice from the data source, producing an in-memory
    /// sub-volume. This method reuses the given `buffer` to avoid
    /// reallocations. Any data that the buffer previously had is
    /// discarded.
    pub fn read_slice_inline(&mut self, buffer: Vec<u8>) -> Result<InMemNiftiVolume> {
        let mut raw_data = buffer;
        raw_data.resize(
            nb_bytes_for_dim_datatype(self.slice_dim(), self.datatype),
            0,
        );
        self.source.read_exact(&mut raw_data)?;

        self.slices_read += 1;
        self.slices_left = self.slices_left.saturating_sub(1);
        InMemNiftiVolume::from_raw_fields(
            *self.slice_dim.raw(),
            self.datatype,
            self.scl_slope,
            self.scl_inter,
            raw_data,
            self.endianness,
        )
    }

    /// Fetch the next slice while reusing a raw data buffer. This is the
    /// streaming iterator equivalent of `Iterator::next`. Once the output
    /// volume has been used, the method [`into_raw_data`] can be used to
    /// recover the vector for the subsequent iteration.
    ///
    /// [`into_raw_data`]: ../inmem/struct.InMemNiftiVolume.html#method.into_raw_data
    pub fn next_inline(&mut self, buffer: Vec<u8>) -> Option<Result<InMemNiftiVolume>> {
        if self.slices_left == 0 {
            return None;
        }
        Some(self.read_slice_inline(buffer))
    }
}

impl<'a, R> NiftiVolume for &'a StreamedNiftiVolume<R> {
    fn dim(&self) -> &[u16] {
        (**self).dim()
    }

    fn dimensionality(&self) -> usize {
        (**self).dimensionality()
    }

    fn data_type(&self) -> NiftiType {
        (**self).data_type()
    }
}

impl<R> NiftiVolume for StreamedNiftiVolume<R> {
    fn dim(&self) -> &[u16] {
        self.dim.as_ref()
    }

    fn dimensionality(&self) -> usize {
        self.dim.rank()
    }

    fn data_type(&self) -> NiftiType {
        self.datatype
    }
}

/**
 * The iterator pattern in a streamed NIfTI volume calls the method
 * [`read_slice`] on `next` unless all slices have already been read from the
 * volume.
 *
 * [`read_slice`](./struct.StreamedNiftiVolume.html#method.read_slice)
 */
impl<R> std::iter::Iterator for StreamedNiftiVolume<R>
where
    R: Read,
{
    type Item = Result<InMemNiftiVolume>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.slices_left == 0 {
            return None;
        }
        Some(self.read_slice())
    }
}

fn calculate_slice_dims(dim: &Dim, slice_rank: u16) -> Dim {
    assert!(dim.rank() > 0);
    assert!(usize::from(slice_rank) < dim.rank());
    let mut raw_dim = *dim.raw();
    raw_dim[0] = slice_rank;
    Dim::new(raw_dim).unwrap()
}

fn calculate_total_slices(dim: &Dim, slice_rank: u16) -> usize {
    assert!(usize::from(slice_rank) < dim.rank());
    let (_, r) = dim.split(slice_rank);
    r.element_count()
}

#[cfg(test)]
mod tests {

    use super::super::{NiftiVolume, RandomAccessNiftiVolume};
    use super::StreamedNiftiVolume;
    use byteordered::Endianness;
    use typedef::NiftiType;
    use NiftiHeader;

    #[test]
    fn test_streamed_base() {
        let volume_data = &[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23];
        let header = NiftiHeader {
            dim: [3, 2, 3, 2, 0, 0, 0, 0],
            datatype: NiftiType::Uint8 as i16,
            scl_slope: 1.,
            scl_inter: 0.,
            endianness: Endianness::native(),
            ..NiftiHeader::default()
        };

        let mut volume = StreamedNiftiVolume::from_reader(&volume_data[..], &header).unwrap();

        assert_eq!(volume.dim(), &[2, 3, 2]);
        assert_eq!(volume.slice_dim(), &[2, 3]);
        assert_eq!(volume.slices_read(), 0);

        {
            let slice = volume
                .next()
                .expect("1st slice should exist")
                .expect("should not fail to construct the volume");

            assert_eq!(slice.dim(), &[2, 3]);
            assert_eq!(slice.data_type(), NiftiType::Uint8);
            assert_eq!(slice.raw_data(), &[1, 3, 5, 7, 9, 11]);
        }
        {
            let slice = volume
                .next()
                .expect("2nd slice should exist")
                .expect("should not fail to construct the volume");

            assert_eq!(slice.dim(), &[2, 3]);
            assert_eq!(slice.data_type(), NiftiType::Uint8);
            assert_eq!(slice.raw_data(), &[13, 15, 17, 19, 21, 23]);
        }
        assert!(volume.next().is_none());
    }

    #[test]
    fn test_streamed_inline() {
        let volume_data = &[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23];
        let header = NiftiHeader {
            dim: [3, 2, 3, 2, 0, 0, 0, 0],
            datatype: NiftiType::Uint8 as i16,
            scl_slope: 1.,
            scl_inter: 0.,
            endianness: Endianness::native(),
            ..NiftiHeader::default()
        };

        let mut volume = StreamedNiftiVolume::from_reader(&volume_data[..], &header).unwrap();

        assert_eq!(volume.dim(), &[2, 3, 2]);
        assert_eq!(volume.slice_dim(), &[2, 3]);
        assert_eq!(volume.slices_read(), 0);
        let buf = Vec::with_capacity(6);
        let buf = {
            let slice = volume
                .next_inline(buf)
                .expect("1st slice should exist")
                .expect("should not fail to construct the volume");

            assert_eq!(slice.dim(), &[2, 3]);
            assert_eq!(slice.data_type(), NiftiType::Uint8);
            assert_eq!(slice.raw_data(), &[1, 3, 5, 7, 9, 11]);
            slice.into_raw_data()
        };
        {
            let slice = volume
                .next_inline(buf)
                .expect("2nd slice should exist")
                .expect("should not fail to construct the volume");

            assert_eq!(slice.dim(), &[2, 3]);
            assert_eq!(slice.data_type(), NiftiType::Uint8);
            assert_eq!(slice.raw_data(), &[13, 15, 17, 19, 21, 23]);
        }
        assert!(volume.next().is_none());
    }

    #[test]
    fn test_streamed_ranked() {
        let volume_data = &[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23];
        let header = NiftiHeader {
            dim: [4, 2, 3, 2, 1, 0, 0, 0],
            datatype: NiftiType::Uint8 as i16,
            scl_slope: 1.,
            scl_inter: 0.,
            endianness: Endianness::native(),
            ..NiftiHeader::default()
        };

        let mut volume =
            StreamedNiftiVolume::from_reader_rank(&volume_data[..], &header, 2).unwrap();

        assert_eq!(volume.dim(), &[2, 3, 2, 1]);
        assert_eq!(volume.slice_dim(), &[2, 3]);
        assert_eq!(volume.slices_read(), 0);

        {
            let slice = volume
                .next()
                .expect("1st slice should exist")
                .expect("should not fail to construct the volume");

            assert_eq!(slice.dim(), &[2, 3]);
            assert_eq!(slice.data_type(), NiftiType::Uint8);
            assert_eq!(slice.raw_data(), &[1, 3, 5, 7, 9, 11]);
        }
        {
            let slice = volume
                .next()
                .expect("2nd slice should exist")
                .expect("should not fail to construct the volume");

            assert_eq!(slice.dim(), &[2, 3]);
            assert_eq!(slice.data_type(), NiftiType::Uint8);
            assert_eq!(slice.raw_data(), &[13, 15, 17, 19, 21, 23]);
        }
        assert!(volume.next().is_none());
    }

    #[test]
    fn test_streamed_lesser_rank() {
        let volume_data = &[
            1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0, 13, 0, 15, 0, 17, 0, 19, 0, 21, 0, 23, 0,
        ];
        let header = NiftiHeader {
            dim: [3, 2, 3, 2, 0, 0, 0, 0],
            datatype: NiftiType::Uint16 as i16,
            scl_slope: 1.,
            scl_inter: 0.,
            endianness: Endianness::Little,
            ..NiftiHeader::default()
        };

        let mut volume =
            StreamedNiftiVolume::from_reader_rank(&volume_data[..], &header, 1).unwrap();

        assert_eq!(volume.dim(), &[2, 3, 2]);
        assert_eq!(volume.slice_dim(), &[2]);
        assert_eq!(volume.slices_read(), 0);

        for (i, raw_data) in [
            &[1, 0, 3, 0],
            &[5, 0, 7, 0],
            &[9, 0, 11, 0],
            &[13, 0, 15, 0],
            &[17, 0, 19, 0],
            &[21, 0, 23, 0],
        ]
        .iter()
        .enumerate()
        {
            let slice = volume
                .next()
                .unwrap_or_else(|| panic!("{}st slice should exist", i))
                .expect("should not fail to construct the volume");

            assert_eq!(slice.dim(), volume.slice_dim());
            assert_eq!(slice.data_type(), NiftiType::Uint16);
            assert_eq!(slice.raw_data(), &raw_data[..]);
            assert_eq!(slice.get_u16(&[0]).unwrap(), u16::from(raw_data[0]));
            assert_eq!(slice.get_u16(&[1]).unwrap(), u16::from(raw_data[2]));
        }

        assert!(volume.next().is_none());
    }
}