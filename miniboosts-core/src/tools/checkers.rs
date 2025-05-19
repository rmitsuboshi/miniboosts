//! This file defines some functions that checks some pre-conditions
//! E.g., Shape of data

use crate::Sample;
use crate::constants::{
    NUMERIC_TOLERANCE,
    SIMPLEX_TOLERANCE,
};

/// Check whether the training sample is valid or not.
#[inline(always)]
pub fn sample(sample: &Sample)
{
    let (n_examples, n_feature) = sample.shape();

    // `data` and `target` must have the length greater than `0`.
    // Since the previous assertion guarantees `n_data == n_target`,
    // we only need to check `n_data`.
    assert!(n_examples > 0);

    // `data` must have a feature.
    assert!(n_feature > 0);
}

/// Check whether the capping parameter is valid or not.
#[inline(always)]
pub fn capping_parameter(nu: f64, n_examples: usize) {
    let n_examples = n_examples as f64;
    assert!((1f64..=n_examples).contains(&nu));
}

/// Check the stepsize
#[inline(always)]
pub fn stepsize(size: f64) {
    assert!(
        (0f64..=1f64).contains(&size),
        "step size must be in [0, 1]. got {size}."
    );
}

/// Check the deformation parameter
#[inline(always)]
pub fn deformation_parameter(t: f64) {
    assert!(
        (0f64..=1f64).contains(&t),
        "deformation parameter `t` must be in [0, 1]. got `t = {t}.`"
    );
}

#[inline(always)]
pub fn capped_simplex_condition(slice: &[f64], nu: f64) {
    let length = slice.len();
    capping_parameter(nu, length);

    let sum = slice.iter().sum::<f64>();
    assert!((sum - 1f64).abs() < SIMPLEX_TOLERANCE, "sum(dist[..]) = {sum}");

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
        let m  = 1_000;
        let nu = 0.01 * m as f64;
        capping_parameter(nu, m);
    }

    #[test]
    fn test_nu_success_02() {
        let m  = 1_000;
        let nu = 1f64;
        capping_parameter(nu, m);
    }

    #[test]
    fn test_nu_success_03() {
        let m  = 1_000;
        let nu = m as f64;
        capping_parameter(nu, m);
    }

    #[test]
    #[should_panic]
    fn test_nu_failure_01() {
        let m  = 1_000;
        let nu = 0f64;
        capping_parameter(nu, m);
    }

    #[test]
    #[should_panic]
    fn test_nu_failure_02() {
        let m  = 1_000;
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

    #[test]
    fn test_deformation_parameter_success_01() {
        let s = 0.5f64;
        deformation_parameter(s);
    }

    #[test]
    fn test_deformation_parameter_success_02() {
        let s = 0f64;
        deformation_parameter(s);
    }

    #[test]
    fn test_deformation_parameter_success_03() {
        let s = 1f64;
        deformation_parameter(s);
    }

    #[test]
    #[should_panic]
    fn test_deformation_parameter_failure_01() {
        let s = -0.0001;
        deformation_parameter(s);
    }

    #[test]
    #[should_panic]
    fn test_deformation_parameter_failure_02() {
        let s = 1.0001;
        deformation_parameter(s);
    }
}

