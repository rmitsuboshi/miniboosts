//! Provides the trait `Booster<D, L>`.
use crate::data_type::Sample;
use crate::base_learner::core::{Classifier, CombinedClassifier};
use crate::base_learner::core::BaseLearner;

/// The trait `Booster` defines the standard framework of Boosting.
/// 
/// You need to implement `run`
/// in order to write a new boosting algorithm.
pub trait Booster<C>
    where C: Classifier
{
    /// A main function that runs boosting algorithm.
    /// This method takes
    /// 
    /// - the reference of an instance of the `BaseLearner` trait,
    /// - a reference of the training examples, and
    /// - a tolerance parameter.
    fn run<B>(&mut self, base_learner: &B, sample: &Sample, tolerance: f64)
        -> CombinedClassifier<C>
        where B: BaseLearner<Clf=C>;
}

