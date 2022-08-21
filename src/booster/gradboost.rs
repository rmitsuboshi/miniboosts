//! Provides the `Gradient Boosting` by Jerome H. Friedman, 1999.
//! See the paper
//! "Greedy Function Approximation: A Gradient Boosting Machine"


use crate::{Data, Sample};
use crate::{Classifier, CombinedClassifier};
use crate::BaseLearner;
use crate::Booster;


/// An enumeration of loss function.
pub enum Loss {
    /// Squared loss
    Square,
    /// Absolute loss
    Absolute,
    /// Negative binomial log-liklihood
    Nbll,
}


/// A structure for gradient boosting.
pub struct GradBoost {
    loss: Loss,
}


impl GradBoost {
}
