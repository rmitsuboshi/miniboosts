//! Runtime precondition checks.
//!
//! Each function returns nothing on success and **panics** when the
//! precondition fails. Boosters and solvers call them to reject invalid
//! parameters or iterates instead of producing silently wrong results.

use crate::core::Sample;
use crate::core::constants::{NUMERIC_TOLERANCE, SIMPLEX_TOLERANCE};

/// Checks that `sample` has at least one example and one feature.
///
/// Labels are not checked; see [`Sample::is_valid_binary_instance`].
///
/// # Panics
/// Panics if `sample` has no examples or no features.
///
/// # Examples
/// ```
/// use std::io::BufReader;
/// use miniboosts::Sample;
/// use miniboosts::core::checkers;
///
/// checkers::sample(&Sample::dummy(3)); // passes
///
/// // A CSV with a header but no rows has no examples.
/// let empty = Sample::from_reader(BufReader::new(&b"x,class\n"[..]), true).unwrap();
/// assert!(std::panic::catch_unwind(|| checkers::sample(&empty)).is_err());
/// ```
#[inline(always)]
pub fn sample(sample: &Sample) {
    let (n_examples, n_feature) = sample.shape();

    // `data` and `target` must have the length greater than `0`.
    // Since the previous assertion guarantees `n_data == n_target`,
    // we only need to check `n_data`.
    assert!(n_examples > 0);

    // `data` must have a feature.
    assert!(n_feature > 0);
}

/// Checks that the capping parameter satisfies `1 <= nu <= n_examples`.
///
/// `nu` caps each example weight at `1 / nu`; the capped simplex
/// `{d : 0 <= d_i <= 1/nu, Σ d_i = 1}` is nonempty iff `nu <= n_examples`.
///
/// # Panics
/// Panics if `nu` is outside `[1, n_examples]` or is NaN.
///
/// # Examples
/// ```
/// use miniboosts::core::checkers::capping_parameter;
///
/// capping_parameter(1.0, 10); // hard margin
/// capping_parameter(10.0, 10); // largest valid value: d must be uniform
/// assert!(std::panic::catch_unwind(|| capping_parameter(11.0, 10)).is_err());
/// assert!(std::panic::catch_unwind(|| capping_parameter(0.5, 10)).is_err());
/// ```
#[inline(always)]
pub fn capping_parameter(nu: f64, n_examples: usize) {
    let n_examples = n_examples as f64;
    assert!((1f64..=n_examples).contains(&nu));
}

/// Checks that a Frank–Wolfe step size lies in `[0, 1]`.
///
/// # Panics
/// Panics if `size` is outside `[0, 1]` or is NaN.
///
/// # Examples
/// ```
/// use miniboosts::core::checkers::stepsize;
///
/// stepsize(0.0);
/// stepsize(1.0);
/// assert!(std::panic::catch_unwind(|| stepsize(f64::NAN)).is_err());
/// ```
#[inline(always)]
pub fn stepsize(size: f64) {
    assert!(
        (0f64..=1f64).contains(&size),
        "step size must be in [0, 1]. got {size}."
    );
}

/// Checks that `slice` lies (approximately) in the capped simplex
/// `{d : 0 <= d_i <= 1/nu, Σ d_i = 1}`.
///
/// The sum may deviate from one by less than
/// [`SIMPLEX_TOLERANCE`], and each entry may exceed `1 / nu` by at most
/// [`NUMERIC_TOLERANCE`]. Negative entries are rejected without tolerance.
///
/// # Panics
/// Panics if `nu` is invalid for `slice.len()` (see [`capping_parameter`]),
/// if the sum condition fails, or if some entry is negative, above
/// `1 / nu + NUMERIC_TOLERANCE`, or NaN.
///
/// # Examples
/// ```
/// use miniboosts::core::checkers::capped_simplex_condition;
///
/// // A distribution with every entry at most 1/2.
/// capped_simplex_condition(&[0.5, 0.25, 0.25], 2.0);
/// // Rounding errors well below the tolerances are accepted.
/// capped_simplex_condition(&[0.1, 0.2, 0.7], 1.0); // 0.1 + 0.2 + 0.7 != 1.0 exactly
///
/// // Violations panic: an entry above the cap, or a sum other than one.
/// let too_large = std::panic::catch_unwind(|| capped_simplex_condition(&[0.8, 0.2], 2.0));
/// assert!(too_large.is_err());
/// let not_a_distribution = std::panic::catch_unwind(|| capped_simplex_condition(&[0.5, 0.4], 1.0));
/// assert!(not_a_distribution.is_err());
/// ```
#[inline(always)]
pub fn capped_simplex_condition(slice: &[f64], nu: f64) {
    let length = slice.len();
    capping_parameter(nu, length);

    let sum = slice.iter().sum::<f64>();
    assert!(
        (sum - 1f64).abs() < SIMPLEX_TOLERANCE,
        "sum(dist[..]) = {sum}"
    );

    let ub = 1f64 / nu + NUMERIC_TOLERANCE;
    assert!(
        slice.iter().all(|s| (0f64..=ub).contains(s)),
        "capping constraint is violated! all element must be in [0, {ub}].\
        slice = {slice:?}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_nu_success_01() {
        let m = 1_000;
        let nu = 0.01 * m as f64;
        capping_parameter(nu, m);
    }

    #[test]
    fn test_nu_success_02() {
        let m = 1_000;
        let nu = 1f64;
        capping_parameter(nu, m);
    }

    #[test]
    fn test_nu_success_03() {
        let m = 1_000;
        let nu = m as f64;
        capping_parameter(nu, m);
    }

    #[test]
    #[should_panic]
    fn test_nu_failure_01() {
        let m = 1_000;
        let nu = 0f64;
        capping_parameter(nu, m);
    }

    #[test]
    #[should_panic]
    fn test_nu_failure_02() {
        let m = 1_000;
        let nu = (m + 1) as f64;
        capping_parameter(nu, m);
    }

    #[test]
    fn test_stepsize_success_01() {
        let s = 0.5f64;
        stepsize(s);
    }

    #[test]
    fn test_stepsize_success_02() {
        let s = 0f64;
        stepsize(s);
    }

    #[test]
    fn test_stepsize_success_03() {
        let s = 1f64;
        stepsize(s);
    }

    #[test]
    #[should_panic]
    fn test_stepsize_failure_01() {
        let s = -0.0001;
        stepsize(s);
    }

    #[test]
    #[should_panic]
    fn test_stepsize_failure_02() {
        let s = 1.0001;
        stepsize(s);
    }
}
