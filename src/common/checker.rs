//! This file defines some functions that checks some pre-conditions
//! E.g., Shape of data

use crate::Sample;


const SIMPLEX_TOLERANCE: f64 = 1e-5;


/// Check whether the training sample is valid or not.
#[inline(always)]
pub(crate) fn check_sample(sample: &Sample)
{
    let (n_sample, n_feature) = sample.shape();


    // `data` and `target` must have the length greater than `0`.
    // Since the previous assertion guarantees `n_data == n_target`,
    // we only need to check `n_data`.
    assert!(n_sample > 0);


    // `data` must have a feature.
    assert!(n_feature > 0);
}


/// Check whether the capping parameter is valid or not.
#[inline(always)]
pub(crate) fn check_nu(nu: f64, n_sample: usize) {
    let n_sample = n_sample as f64;
    assert!((1f64..=n_sample).contains(&nu));
}

/// Check the stepsize
#[inline(always)]
pub(crate) fn check_stepsize(size: f64) {
    assert!((0f64..=1f64).contains(&size));
}


#[inline(always)]
pub(crate) fn check_capped_simplex_condition(
    slice: &[f64],
    nu: f64,
)
{
    let length = slice.len();
    check_nu(nu, length);

    let sum = slice.iter().sum::<f64>();
    let diff = (sum - 1f64).abs();
    if diff > SIMPLEX_TOLERANCE {
        println!(">> diff is {diff} > {SIMPLEX_TOLERANCE}");
        println!(">> sum  = {sum}");
        assert!((sum - 1f64).abs() < SIMPLEX_TOLERANCE, "sum(dist[..]) = {sum}");
    }
    assert!((sum - 1f64).abs() < SIMPLEX_TOLERANCE, "sum(dist[..]) = {sum}");

    let ub = 1f64 / nu;
    assert!(
        slice.iter().all(|s| (0f64..=ub).contains(s)),
        "capping constraint is violated!"
    );
}

