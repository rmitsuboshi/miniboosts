//! Provides the trait `Booster<C>`.

use polars::prelude::*;
use crate::{
    WeakLearner,
    Classifier,
    CombinedClassifier
};


/// State of the boosting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// Terminate the boosting process
    Terminate,
    /// Continue the boosting process
    Continue,
}


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
    /// - the reference of an instance of the `WeakLearner` trait,
    /// - a reference of the training examples, and
    fn run<W>(&mut self,
              weak_learner: &W,
              data: &DataFrame,
              target: &Series,
    ) -> CombinedClassifier<C>
        where W: WeakLearner<Clf = C>
    {
        self.preprocess(weak_learner, data, target);

        for it in 1.. {
            let state = self.boost(weak_learner, data, target, it);

            if state == State::Terminate {
                break;
            }
        }

        self.postprocess(weak_learner, data, target)
    }


    /// Pre-processing for `self`.
    fn preprocess<W>(
        &mut self,
        weak_learner: &W,
        data: &DataFrame,
        target: &Series,
    )
        where W: WeakLearner<Clf = C>;


    /// Boosting per iteration.
    /// This method returns `true` if the stopping criterion is satisfied,
    /// `false` otherwise.
    fn boost<W>(
        &mut self,
        weak_learner: &W,
        data: &DataFrame,
        target: &Series,
        iteration: usize,
    ) -> State
        where W: WeakLearner<Clf = C>;


    /// Post-processing.
    /// This method returns a combined hypothesis.
    fn postprocess<W>(
        &mut self,
        weak_learner: &W,
        data: &DataFrame,
        target: &Series,
    ) -> CombinedClassifier<C>
        where W: WeakLearner<Clf = C>;
}

