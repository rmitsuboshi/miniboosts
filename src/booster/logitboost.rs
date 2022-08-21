//! Provides `LogitBoost` by Friedman, Hastie, and Tibshirani, 2000.
use rayon::prelude::*;


use crate::{
    Data,
    Sample,
    Classifier,
    CombinedClassifier,
    BaseLearner,
    Booster,
};


/// Defines `LogitBoost`.
pub struct LogitBoost {
    pub(self) dist: Vec<f64>,
}
