//! Per-constraint mass-ratio clamping for solver stability.
//!
//! When two dynamic bodies in a constraint (contact OR joint) have very
//! different masses (say 1000:1), the constraint solver's effective-mass math
//! gives the lighter body huge impulses, causing jitter, sinking, "launching,"
//! or oscillating joints. The fix is the Halo 2 / Havok trick: cap the heavier
//! body's effective mass to `MASS_RATIO_CAP` × the lighter body's mass for the
//! purposes of this constraint only. The heavy body still integrates with its
//! real mass for gravity and other constraints.
//!
//! This module returns *scale factors* (≥1) that callers apply to inverse mass
//! and inverse inertia. Skipped when either body is static or kinematic
//! (`inv_mass == 0`) — a 1kg crate must not see the floor as if it weighed
//! 16kg.

use crate::math::{AngularInertia, Real, SimdReal, Vector};
use crate::utils::SimdRealCopy;
#[cfg(feature = "dim3")]
use parry::utils::SdpMatrix3;
use simba::simd::{SimdPartialOrd, SimdValue};

/// Maximum allowed mass ratio (heavier / lighter) between two dynamic bodies
/// in a contact constraint. Halo 2 / Havok use 16:1.
pub const MASS_RATIO_CAP: Real = 16.0;

const INV_MASS_RATIO_CAP: Real = 1.0 / MASS_RATIO_CAP;

/// SIMD variant. Returns `(scale1, scale2)` to be applied to body 1's and
/// body 2's inverse mass and inverse inertia for this contact.
///
/// Scale ≥ 1 always (clamping only ever boosts inverse mass = shrinks
/// effective mass). Scale = 1 means no change. Use `im * scale` and
/// `ii_torque_dir * scale` at call sites.
#[inline]
pub fn mass_clamp_scales_simd(
    im1: Vector<SimdReal>,
    im2: Vector<SimdReal>,
) -> (SimdReal, SimdReal) {
    // Linear inverse mass is isotropic — sample x.
    let im1_s = im1.x;
    let im2_s = im2.x;

    let zero = SimdReal::splat(0.0);
    let one = SimdReal::splat(1.0);
    let inv_cap = SimdReal::splat(INV_MASS_RATIO_CAP);

    // Skip clamping if either body is static/kinematic (im == 0).
    let both_dynamic = im1_s.simd_gt(zero) & im2_s.simd_gt(zero);

    // Each body's inverse mass must be >= max(im1, im2) / MASS_RATIO_CAP.
    let im_max = im1_s.simd_max(im2_s);
    let threshold = im_max * inv_cap;

    let im1_clamped = im1_s.simd_max(threshold);
    let im2_clamped = im2_s.simd_max(threshold);

    // scale = clamped / original. When original is 0 we'd divide by zero, but
    // both_dynamic is false in that lane so the select discards the result.
    let scale1_raw = im1_clamped / im1_s;
    let scale2_raw = im2_clamped / im2_s;

    let scale1 = scale1_raw.select(both_dynamic, one);
    let scale2 = scale2_raw.select(both_dynamic, one);

    (scale1, scale2)
}

/// Scalar variant for the generic (multibody) constraint path. Returns
/// `(scale1, scale2)` with the same semantics as `mass_clamp_scales_simd`.
///
/// `dynamic_pair` must be `true` only when *both* bodies are non-multibody
/// dynamic/kinematic rigid bodies — multibodies have their mass encoded in
/// jacobians and are out of scope for this clamp.
#[inline]
pub fn mass_clamp_scales_scalar(
    im1: Vector<Real>,
    im2: Vector<Real>,
    dynamic_pair: bool,
) -> (Real, Real) {
    if !dynamic_pair {
        return (1.0, 1.0);
    }
    let im1_s = im1.x;
    let im2_s = im2.x;
    if im1_s <= 0.0 || im2_s <= 0.0 {
        return (1.0, 1.0);
    }
    let im_max = im1_s.max(im2_s);
    let threshold = im_max * INV_MASS_RATIO_CAP;
    let scale1 = (im1_s.max(threshold)) / im1_s;
    let scale2 = (im2_s.max(threshold)) / im2_s;
    (scale1, scale2)
}

/// Scale an `AngularInertia` by a scalar (in-place creation of a scaled copy).
/// Generic over `N` so it works for both `Real` and `SimdReal`. In 2D, inertia
/// is a scalar; in 3D, it's a 6-component symmetric matrix (`SdpMatrix3`).
#[cfg(feature = "dim2")]
#[inline]
pub fn scale_inertia<N: SimdRealCopy>(ii: AngularInertia<N>, scale: N) -> AngularInertia<N> {
    ii * scale
}

#[cfg(feature = "dim3")]
#[inline]
pub fn scale_inertia<N: SimdRealCopy>(ii: AngularInertia<N>, scale: N) -> AngularInertia<N> {
    SdpMatrix3 {
        m11: ii.m11 * scale,
        m12: ii.m12 * scale,
        m13: ii.m13 * scale,
        m22: ii.m22 * scale,
        m23: ii.m23 * scale,
        m33: ii.m33 * scale,
    }
}
