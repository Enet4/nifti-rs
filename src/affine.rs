//! This module defines some affine-related utilities.

use nalgebra::{Matrix3, Matrix4, Quaternion, RowVector4, Scalar, SymmetricEigen, Vector3};

/// 3x3 affine transformation matrix.
pub type Affine3 = Matrix3<f32>;
/// 4x4 affine transformation matrix.
pub type Affine4 = Matrix4<f32>;

const QUARTERNION_THRESHOLD: f64 = -::std::f32::EPSILON as f64 * 3.0;

/// Separate a 4x4 affine into its 3x3 affine and translation components.
pub fn affine_and_translation<T>(affine: &Matrix4<T>) -> (Matrix3<T>, Vector3<T>)
where
    T: Copy + Scalar,
{
    let translation = Vector3::new(affine[12], affine[13], affine[14]);
    let affine = affine.fixed_slice::<3, 3>(0, 0).into_owned();
    (affine, translation)
}

/// Get affine implied by given shape and zooms.
///
/// We get the translations from the center of the image (implied by `shape`).
#[rustfmt::skip]
pub(crate) fn shape_zoom_affine(shape: &[u64], spacing: &[f64]) -> Matrix4<f64> {
    // Get translations from center of image
    let origin = Vector3::new(
        (shape[0] as f64 - 1.0) / 2.0,
        (shape[1] as f64 - 1.0) / 2.0,
        (shape[2] as f64 - 1.0) / 2.0,
    );
    let spacing = [-spacing[0] as f64, spacing[1] as f64, spacing[2] as f64];
    Matrix4::new(
        spacing[0], 0.0, 0.0, -origin[0] * spacing[0],
        0.0, spacing[1], 0.0, -origin[1] * spacing[1],
        0.0, 0.0, spacing[2], -origin[2] * spacing[2],
        0.0, 0.0, 0.0, 1.0,
    )
}

/// Compute unit quaternion from last 3 values.
///
/// If w, x, y, z are the values in the full quaternion, assumes w is positive.
/// Panic if w*w is estimated to be negative.
/// w = 0.0 corresponds to a 180 degree rotation.
/// The unit quaternion specifies that `wxyz.dot(wxyz) == 1.0`.
///
/// If w is positive (assumed here), w is given by:
///     w = (1.0 - (x*x + y*y + z*z)).sqrt()
/// `1.0 - (x*x + y*y + z*z)` can be near zero, which will lead to numerical instability in sqrt.
/// Here we use f64 to reduce numerical instability.
pub(crate) fn fill_positive(xyz: Vector3<f64>) -> Quaternion<f64> {
    let w2 = 1.0 - xyz.dot(&xyz);
    let w = if w2 < 0.0 {
        if w2 < QUARTERNION_THRESHOLD {
            panic!("1.0 - (x*x + y*y + z*z) should be positive, but is {}", w2);
        }
        0.0
    } else {
        w2.sqrt()
    };
    Quaternion::new(w, xyz.x, xyz.y, xyz.z)
}

/// Calculate quaternion corresponding to given rotation matrix.
///
/// Method claimed to be robust to numerical errors in `affine`. Constructs quaternion by
/// calculating maximum eigenvector for matrix `k` (constructed from input `affine`). Although this
/// is not tested, a maximum eigenvalue of 1 corresponds to a valid rotation.
///
/// A quaternion `q * -1.0` corresponds to the same rotation as `q`; thus the sign of the
/// reconstructed quaternion is arbitrary, and we return quaternions with positive `w` `(q[0])`.
///
/// Bar-Itzhack, Itzhack Y. "New method for extracting the quaternion from a rotation
/// matrix", AIAA Journal of Guidance, Control and Dynamics 23(6):1085-1087, 2000
#[rustfmt::skip]
pub(crate) fn affine_to_quaternion(affine: &Matrix3<f64>) -> RowVector4<f64> {
    // qyx refers to the contribution of the y input vector component to the x output vector
    // component. qyx is therefore the same as M[0, 1]. The notation is from the Wikipedia article.
    let qxx = affine[0];
    let qyx = affine[3];
    let qzx = affine[6];
    let qxy = affine[1];
    let qyy = affine[4];
    let qzy = affine[7];
    let qxz = affine[2];
    let qyz = affine[5];
    let qzz = affine[8];

    // Fill only lower half of symmetric matrix.
    let k = Matrix4::new(
        qxx - qyy - qzz, 0.0,             0.0,             0.0,
        qyx + qxy,       qyy - qxx - qzz, 0.0,             0.0,
        qzx + qxz,       qzy + qyz,       qzz - qxx - qyy, 0.0,
        qyz - qzy,       qzx - qxz,       qxy - qyx,       qxx + qyy + qzz,
    );

    // Use Hermitian eigenvectors, values for speed.
    let SymmetricEigen {
        eigenvalues: values,
        eigenvectors: vectors,
    } = k.symmetric_eigen();

    // Select largest eigenvector, reorder to w,x,y,z quaternion.
    let (max_idx, _) = values
        .as_slice()
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    let max_vector = vectors.fixed_columns::<1>(max_idx);
    let quaternion = RowVector4::new(max_vector[3], max_vector[0], max_vector[1], max_vector[2]);

    // Prefer quaternion with positive `w`.
    if quaternion[0] < 0.0 {
        quaternion * -1.0
    } else {
        quaternion
    }
}

/// Calculate rotation matrix corresponding to quaternion.
///
/// Rotation matrix applies to column vectors, and is applied to the left of coordinate vectors.
/// The algorithm here allows non-unit quaternions.
///
/// Algorithm from https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
#[rustfmt::skip]
#[allow(clippy::many_single_char_names)]
pub(crate) fn quaternion_to_affine(q: Quaternion<f64>) -> Matrix3<f64> {
    let nq = q.w * q.w + q.i * q.i + q.j * q.j + q.k * q.k;
    if nq < ::std::f64::EPSILON {
        return Matrix3::identity();
    }
    let s = 2.0 / nq;
    let x = q.i * s;
    let y = q.j * s;
    let z = q.k * s;
    let wx = q.w * x;
    let wy = q.w * y;
    let wz = q.w * z;
    let xx = q.i * x;
    let xy = q.i * y;
    let xz = q.i * z;
    let yy = q.j * y;
    let yz = q.j * z;
    let zz = q.k * z;
    Matrix3::new(
        1.0 - (yy + zz), xy - wz, xz + wy,
        xy + wz, 1.0 - (xx + zz), yz - wx,
        xz - wy, yz + wx, 1.0 - (xx + yy),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[rustfmt::skip]
    fn test_shape_zoom_affine() {
        let affine = shape_zoom_affine(&[3, 5, 7], &[3.0, 2.0, 1.0]);
        let real_affine = Matrix4::new(
            -3.0, 0.0, 0.0, 3.0,
            0.0, 2.0, 0.0, -4.0,
            0.0, 0.0, 1.0, -3.0,
            0.0, 0.0, 0.0, 1.0,
        );
        assert_eq!(affine, real_affine);

        let affine = shape_zoom_affine(&[256, 256, 54], &[0.9375, 0.9375, 3.0]);
        let real_affine = Matrix4::new(
            -0.9375, 0.0, 0.0, 119.53125,
            0.0, 0.9375, 0.0, -119.53125,
            0.0, 0.0, 3.0, -79.5,
            0.0, 0.0, 0.0, 1.0,
        );
        assert_eq!(affine, real_affine);
    }

    #[test]
    fn test_fill_positive() {
        let q = fill_positive(Vector3::new(0.0, 0.0, 0.0));
        assert_eq!(q, Quaternion::new(1.0, 0.0, 0.0, 0.0));

        let q = fill_positive(Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(q, Quaternion::new(0.0, 1.0, 0.0, 0.0));
        assert_eq!(q.dot(&q), 1.0);
    }

    #[test]
    fn test_affine_to_quaternion() {
        assert_eq!(
            affine_to_quaternion(&Matrix3::identity()),
            RowVector4::new(1.0, 0.0, 0.0, 0.0)
        );

        let affine = Matrix3::from_diagonal(&Vector3::new(1.0, -1.0, -1.0));
        assert_eq!(
            affine_to_quaternion(&affine),
            RowVector4::new(0.0, 1.0, 0.0, 0.0)
        );

        let affine = Matrix3::new(1.1, 0.1, 0.1, 0.2, 1.1, 0.5, 0.0, 0.0, 1.0);
        let quaternion = affine_to_quaternion(&affine);
        assert_abs_diff_eq!(
            quaternion.as_slice(),
            &[
                0.9929998817020889,
                -0.11474227051531193,
                0.017766153114299018,
                0.02167510323267152
            ][..],
            epsilon = 1e-11,
        );
    }

    #[test]
    fn test_quaternion_to_affine() {
        // Identity quaternion
        let affine = quaternion_to_affine(Quaternion::new(1.0, 0.0, 0.0, 0.0));
        assert_eq!(affine, Matrix3::identity());

        // 180 degree rotation around axis 0
        let affine = quaternion_to_affine(Quaternion::new(0.0, 1.0, 0.0, 0.0));
        assert_eq!(
            affine,
            Matrix3::from_diagonal(&Vector3::new(1.0, -1.0, -1.0))
        );
    }
}
