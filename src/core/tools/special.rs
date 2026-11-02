//! Special functions shared by the potential-based boosters
//! ([`BrownBoost`](crate::BrownBoost) and [`RobustBoost`](crate::RobustBoost)).
use libm::{erf, erfc};

/// Returns `erf(x) - erf(y)`, using `erfc` when both arguments lie in the
/// same tail so that the difference does not cancel to zero.
pub(crate) fn erf_diff(x: f64, y: f64) -> f64 {
    if x >= 0.0 && y >= 0.0 {
        erfc(y) - erfc(x)
    } else if x <= 0.0 && y <= 0.0 {
        erfc(-x) - erfc(-y)
    } else {
        erf(x) - erf(y)
    }
}

/// Returns the `x ≥ 0` with `erfc(x) = target`, for `target ∈ (0, 1)`.
///
/// `erfc` decreases from `1` at `0` to below `1e-300` at `27`, so bisection
/// on `[0, 27]` brackets every representable target above that.
pub(crate) fn erfc_inverse(target: f64) -> f64 {
    assert!(
        target > 0.0 && target < 1.0,
        "erfc_inverse needs a target in (0, 1)"
    );
    let (mut lo, mut hi) = (0.0_f64, 27.0_f64);
    while erfc(hi) > target {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if erfc(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}
