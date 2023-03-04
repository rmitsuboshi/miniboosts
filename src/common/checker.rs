//! This file defines some functions that checks some pre-conditions
//! E.g., Shape of data

use crate::Sample;


/// Check whether the training sample is valid or not.
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
pub(crate) fn check_nu(nu: f64, n_sample: usize) {
    let n_sample = n_sample as f64;
    assert!((1.0..=n_sample).contains(&nu));
}
