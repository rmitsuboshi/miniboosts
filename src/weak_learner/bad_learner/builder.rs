use crate::{
    Sample,
};
use super::worstcase_lpboost::BadBaseLearner;

const DEFAULT_TOLERANCE: f64 = 1e-9;
const DEFAULT_NU: f64 = 1f64;


/// A struct that builds `BadBaseLearner.`
pub struct BadBaseLearnerBuilder {
    // # of examples
    n_sample: usize,
    // The tolerance parameter
    tolerance: f64,

    // The capping parameter
    nu: f64,
}


impl BadBaseLearnerBuilder {
    /// Construct a new instance of `BadBaseLearnerBuilder`.
    pub fn new(sample: &Sample) -> Self {
        let n_sample = sample.shape().0;
        Self {
            n_sample,
            tolerance: DEFAULT_TOLERANCE,
            nu: DEFAULT_NU,
        }
    }


    /// Set the `tolerance` parameter in this struct.
    /// By default, it is set as `DEFAULT_TOLERANCE: f64 = 1e-9.`
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }


    /// Set the capping parameter.
    /// By default, it is set as `DEFAULT_NU: f64 = 1f64.`
    pub fn nu(mut self, nu: f64) -> Self {
        assert!(
            (1f64..self.n_sample as f64).contains(&nu),
            "The capping parameter must be in `[1, n_sample]`"
        );
        self.nu = nu;
        self
    }


    /// Build a new instance of `BadBaseLearner.`
    pub fn build(self) -> BadBaseLearner {
        BadBaseLearner::new(self.n_sample, self.tolerance, self.nu)
    }
}
