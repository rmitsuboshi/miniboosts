//! Provides `Booster` trait.

use polars::prelude::*;
use crate::{
    WeakLearner,
    CombinedHypothesis,
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
pub trait Booster<F> {
    /// A main function that runs boosting algorithm.
    /// This method takes
    /// 
    /// - the reference of an instance of the `WeakLearner` trait,
    /// - a reference of the training examples, and
    fn run<W>(
        &mut self,
        weak_learner: &W,
        data: &DataFrame,
        target: &Series,
    ) -> CombinedHypothesis<F>
        where W: WeakLearner<Clf = F>
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
        where W: WeakLearner<Clf = F>;


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
        where W: WeakLearner<Clf = F>;


    /// Post-processing.
    /// This method returns a combined hypothesis.
    fn postprocess<W>(
        &mut self,
        weak_learner: &W,
        data: &DataFrame,
        target: &Series,
    ) -> CombinedHypothesis<F>
        where W: WeakLearner<Clf = F>;
}

