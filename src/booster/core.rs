//! Provides the trait `Booster<C>`.

use polars::prelude::*;
use crate::{
    BaseLearner,
    Classifier,
    CombinedClassifier
};

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
    fn run<B>(&mut self,
              base_learner: &B,
              data: &DataFrame,
              target: &Series,
    ) -> CombinedClassifier<C>
        where B: BaseLearner<Clf = C>;
}

