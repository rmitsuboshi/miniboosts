//! The core library for the base learner in the boosting protocol.
//! 
//! The base learner in the general boosting setting is as follows:
//! 
//! Given a distribution over training examples,
//! the base learner returns a hypothesis that is slightly better than
//! the random guessing, where the **edge** is the affine transformation of
//! the weighted training error.
//! 
//! In this code, we assume that the base learner returns a hypothesis
//! that **maximizes** the edge for a given distribution.
//! This assumption is stronger than the previous one, but the resulting
//! combined hypothesis becomes much stronger.
//! 
//! I'm planning to implement the code for the general base learner setting.
//! 
use crate::{Data, Sample};


/// An interface that returns a function that implements 
/// the `Classifier` trait.
pub trait BaseLearner<D: Data> {
    /// Returned hypothesis generated by `self`.
    type Clf;

    /// Returns an instance of the `Classifier` trait
    /// that maximizes the edge of the `sample`
    /// with respect to the given `distribution`.
    fn best_hypothesis(&self, sample: &Sample<D>, distribution: &[f64])
        -> Self::Clf;
}

