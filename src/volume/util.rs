//! Miscellaneous volume-related functions
use error::{Result, NiftiError};

pub fn coords_to_index(coords: &[u16], dim: &[u16]) -> Result<usize> {
    if coords.len() != dim.len() || coords.is_empty() {
        return Err(NiftiError::IncorrectVolumeDimensionality(
            dim.len() as u16,
            coords.len() as u16
        ))
    }

    if !coords.iter().zip(dim).all(|(i, d)| {
        *i < (*d) as u16
    }) {
        return Err(NiftiError::OutOfBounds(Vec::from(coords)));
    }

    let mut crds = coords.into_iter();
    let start = *crds.next_back().unwrap() as usize;
    let index = crds.zip(dim).rev()
        .fold(start, |a, b| {
            a * *b.1 as usize + *b.0 as usize
    });

    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::coords_to_index;

    #[test]
    fn test_coords_to_index() {
        assert!(coords_to_index(&[0, 0], &[10, 10, 5]).is_err());
        assert!(coords_to_index(&[0, 0, 0, 0], &[10, 10, 5]).is_err());
        assert_eq!(
            coords_to_index(&[0, 0, 0], &[10, 10, 5]).unwrap(),
            0
        );

        assert_eq!(
            coords_to_index(&[1, 0, 0], &[16, 16, 3]).unwrap(),
            1
        );
        assert_eq!(
            coords_to_index(&[0, 1, 0], &[16, 16, 3]).unwrap(),
            16
        );
        assert_eq!(
            coords_to_index(&[0, 0, 1], &[16, 16, 3]).unwrap(),
            256
        );
        assert_eq!(
            coords_to_index(&[1, 1, 1], &[16, 16, 3]).unwrap(),
            273
        );

        assert_eq!(
            coords_to_index(&[15, 15, 2], &[16, 16, 3]).unwrap(),
            16 * 16 * 3 - 1
        );

        assert!(coords_to_index(&[16, 15, 2], &[16, 16, 3]).is_err());
    }
}
