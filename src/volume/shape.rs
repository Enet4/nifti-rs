//! Shape and N-dimensional index constructs.
//!
//! The NIfTI-1 format has a hard dimensionality limit of 7. This is
//! specified in the `dim` field as an array of 8 integers where the
//! first element represents the number of dimensions. In order
//! to make dimensions and indices easier to manipulate, the types
//! [`Dim`] and [`Idx`] are provided here.
//!
//! [`Dim`]: ./struct.Dim.html
//! [`Idx`]: ./struct.Idx.html
use crate::error::{NiftiError, Result};
use crate::util::{validate_dim, validate_dimensionality};

/// A validated N-dimensional index in the NIfTI format.
#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct Idx(
    /// dimensions starting at 1, dim[0] is the dimensionality
    [u16; 8],
);

impl Idx {
    /// Validate and create a new index from the raw data field.
    ///
    /// # Example
    ///
    /// ```
    /// # use nifti::volume::shape::Idx;
    /// let idx = Idx::new([3, 1, 2, 5, 0, 0, 0, 0])?;
    /// assert_eq!(idx.as_ref(), &[1, 2, 5]);
    /// # Ok::<(), nifti::NiftiError>(())
    /// ```
    pub fn new(idx: [u16; 8]) -> Result<Self> {
        let _ = validate_dimensionality(&idx)?;
        Ok(Idx(idx))
    }

    /// Create a new index without validating it.
    ///
    /// # Safety
    ///
    /// The program may misbehave severely if the raw `idx` field is not
    /// consistent. The first element, `idx[0]`, must be a valid rank between 1
    /// and 7.
    pub unsafe fn new_unchecked(idx: [u16; 8]) -> Self {
        Idx(idx)
    }

    /// Create a new N-dimensional index using the given slice as the concrete
    /// shape (`idx[0]` is not the rank but the actual width-wide position of
    /// the index).
    ///
    /// # Example
    ///
    /// ```
    /// # use nifti::volume::shape::Idx;
    /// let idx = Idx::from_slice(&[1, 2, 5])?;
    /// assert_eq!(idx.as_ref(), &[1, 2, 5]);
    /// # Ok::<(), nifti::NiftiError>(())
    /// ```
    pub fn from_slice(idx: &[u16]) -> Result<Self> {
        if idx.len() == 0 || idx.len() > 7 {
            return Err(NiftiError::InconsistentDim(0, idx.len() as u16));
        }
        let mut raw = [0; 8];
        raw[0] = idx.len() as u16;
        for (i, d) in idx.iter().enumerate() {
            raw[i + 1] = *d;
        }
        Ok(Idx(raw))
    }

    /// Retrieve a reference to the raw field
    pub fn raw(&self) -> &[u16; 8] {
        &self.0
    }

    /// Retrieve the rank of this index (dimensionality)
    pub fn rank(&self) -> usize {
        usize::from(self.0[0])
    }
}

impl AsRef<[u16]> for Idx {
    fn as_ref(&self) -> &[u16] {
        &self.0[1..=self.rank()]
    }
}

impl AsMut<[u16]> for Idx {
    fn as_mut(&mut self) -> &mut [u16] {
        let rank = self.rank();
        &mut self.0[1..=rank]
    }
}

/// A validated NIfTI volume shape.
#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct Dim(Idx);

impl Dim {
    /// Validate and create a new volume shape.
    ///
    /// # Example
    ///
    /// ```
    /// # use nifti::volume::shape::Dim;
    /// let dim = Dim::new([3, 64, 32, 16, 0, 0, 0, 0])?;
    /// assert_eq!(dim.as_ref(), &[64, 32, 16]);
    /// # Ok::<(), nifti::NiftiError>(())
    /// ```
    pub fn new(dim: [u16; 8]) -> Result<Self> {
        let _ = validate_dim(&dim)?;
        Ok(Dim(Idx(dim)))
    }

    /// Create a new volume shape without validating it.
    ///
    /// # Safety
    ///
    /// The program may misbehave severely if the raw `dim` field is not
    /// consistent. The first element, `dim[0]`, must be a valid rank between
    /// 1 and 7, and the valid dimensions in `dim[0..rank]` must be positive.
    pub unsafe fn new_unchecked(dim: [u16; 8]) -> Self {
        Dim(Idx(dim))
    }

    /// Create a new volume shape using the given slice as the concrete
    /// shape (`dim[0]` is not the rank but the actual width of the volume).
    ///
    /// # Example
    ///
    /// ```
    /// # use nifti::volume::shape::Dim;
    /// let dim = Dim::from_slice(&[64, 32, 16])?;
    /// assert_eq!(dim.as_ref(), &[64, 32, 16]);
    /// # Ok::<(), nifti::NiftiError>(())
    /// ```
    pub fn from_slice(dim: &[u16]) -> Result<Self> {
        if dim.len() == 0 || dim.len() > 7 {
            return Err(NiftiError::InconsistentDim(0, dim.len() as u16));
        }
        let mut raw = [0; 8];
        raw[0] = dim.len() as u16;
        for (i, d) in dim.iter().enumerate() {
            raw[i + 1] = *d;
        }
        let _ = validate_dim(&raw)?;
        Ok(Dim(Idx(raw)))
    }

    /// Retrieve a reference to the raw dim field
    pub fn raw(&self) -> &[u16; 8] {
        self.0.raw()
    }

    /// Retrieve the rank of this shape (dimensionality)
    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    /// Calculate the number of elements in this shape
    pub fn element_count(&self) -> usize {
        self.as_ref().iter().cloned().map(usize::from).product()
    }

    /// Split the dimensions into two parts at the given axis. The first `Dim`
    /// will cover the first axes up to `axis`, excluding `axis` itself.
    ///
    /// # Panic
    ///
    /// Panics if `axis` is not between 0 and `self.rank()`.
    pub fn split(&self, axis: u16) -> (Dim, Dim) {
        let axis = usize::from(axis);
        assert!(axis <= self.rank());
        let (l, r) = self.as_ref().split_at(axis);
        (Dim::from_slice(l).unwrap(), Dim::from_slice(r).unwrap())
    }

    /// Provide an iterator traversing through all possible indices of a
    /// hypothetical volume with this shape.
    pub fn index_iter(&self) -> DimIter {
        DimIter::new(*self)
    }
}

impl AsRef<[u16]> for Dim {
    fn as_ref(&self) -> &[u16] {
        self.0.as_ref()
    }
}

/// An iterator of all indices in a multi-dimensional volume.
///
/// Traversal is in standard NIfTI volume order (column major).
#[derive(Debug, Clone)]
pub struct DimIter {
    shape: Dim,
    state: DimIterState,
}

#[derive(Debug, Copy, Clone)]
enum DimIterState {
    First,
    Middle(Idx),
    Fused,
}

impl DimIter {
    fn new(shape: Dim) -> Self {
        DimIter {
            shape,
            state: DimIterState::First,
        }
    }
}

impl Iterator for DimIter {
    type Item = Idx;

    fn next(&mut self) -> Option<Self::Item> {
        let (out, next_state) = match &mut self.state {
            DimIterState::First => {
                let out = Idx([self.shape.rank() as u16, 0, 0, 0, 0, 0, 0, 0]);
                dbg!((Some(out), DimIterState::Middle(out)))
            }
            DimIterState::Fused => dbg!((None, DimIterState::Fused)),
            DimIterState::Middle(mut current) => {
                let mut good = false;
                for (c, s) in Iterator::zip(current.as_mut().iter_mut(), self.shape.as_ref().iter())
                {
                    if *c < *s - 1 {
                        *c += 1;
                        good = true;
                        break;
                    }
                    *c = 0;
                }
                if good {
                    dbg!((Some(current), DimIterState::Middle(current)))
                } else {
                    dbg!((None, DimIterState::Fused))
                }
            }
        };
        self.state = next_state;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{Dim, Idx};

    #[test]
    fn test_dim() {
        let raw_dim = [3, 256, 256, 100, 0, 0, 0, 0];
        let dim = Dim::new(raw_dim).unwrap();
        assert_eq!(dim.as_ref(), &[256, 256, 100]);
        assert_eq!(dim.element_count(), 6553600);
    }

    #[test]
    fn test_dim_iter() {
        let raw_dim = [2, 3, 4, 0, 0, 0, 0, 0];
        let dim = Dim::new(raw_dim).unwrap();
        assert_eq!(dim.as_ref(), &[3, 4]);
        assert_eq!(dim.element_count(), 12);

        let idx: Vec<_> = dim.index_iter().take(13).collect();
        assert_eq!(idx.len(), dim.element_count());
        for (i, (got, expected)) in Iterator::zip(
            idx.into_iter(),
            vec![
                Idx::from_slice(&[0, 0]).unwrap(),
                Idx::from_slice(&[1, 0]).unwrap(),
                Idx::from_slice(&[2, 0]).unwrap(),
                Idx::from_slice(&[0, 1]).unwrap(),
                Idx::from_slice(&[1, 1]).unwrap(),
                Idx::from_slice(&[2, 1]).unwrap(),
                Idx::from_slice(&[0, 2]).unwrap(),
                Idx::from_slice(&[1, 2]).unwrap(),
                Idx::from_slice(&[2, 2]).unwrap(),
                Idx::from_slice(&[0, 3]).unwrap(),
                Idx::from_slice(&[1, 3]).unwrap(),
                Idx::from_slice(&[2, 3]).unwrap(),
            ]
            .into_iter(),
        )
        .enumerate()
        {
            assert_eq!(got, expected, "#{} not ok", i);
        }
    }
}
