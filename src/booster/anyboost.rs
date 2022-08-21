//! Provides `AnyBoost` by Mason et al., 1999.
use crate::{Data, Sample};
use crate::{Classifier, CombinedClassifier};
use crate::BaseLearner;
use crate::Booster;


pub struct AnyBoost<F> {
    cost: F,
    dist: Vec<f64>,
}


impl AnyBoost {

    /// Initialize `AnyBoost`.
    pub fn init<D, L>(sample: &Sample<D, L>) -> Self {
        let m = sample.len();
        assert!(m != 0);


        let uni = 1.0 / m as f64;
        Self {
            dist: vec![uni; m],
        }
    }


}


