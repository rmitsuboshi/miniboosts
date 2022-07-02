//! Provides the trait `Booster<C>`.
// use crate::{Data, Sample};
use polars::prelude::*;
use crate::{Classifier, CombinedClassifier};
use crate::BaseLearner;

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
    fn run<B>(&mut self,
              base_learner: &B,
              data: &DataFrame,
              target: &Series,
              tolerance: f64)
        -> CombinedClassifier<C>
        where B: BaseLearner<Clf = C>;
}

// pub trait Booster<D, L, C>
//     where D: Data,
//           C: Classifier<D, L>
// {
//     /// A main function that runs boosting algorithm.
//     /// This method takes
//     /// 
//     /// - the reference of an instance of the `BaseLearner` trait,
//     /// - a reference of the training examples, and
//     /// - a tolerance parameter.
//     fn run<B>(&mut self,
//               base_learner: &B,
//               sample:       &Sample<D, L>,
//               tolerance:    f64)
//         -> CombinedClassifier<D, L, C>
//         where B: BaseLearner<D, L, Clf = C>;
// }

