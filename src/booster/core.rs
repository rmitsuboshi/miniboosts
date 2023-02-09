//! Provides `Booster` trait.

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


/// The trait [`Booster`](Booster) defines the standard framework of Boosting.
/// 
/// You need to implement [`Booster::preprocess`](Booster::preprocess),
/// [`Booster::boost`](Booster::boost), 
/// and [`Booster::postprocess`](Booster::postprocess)
/// to write a new boosting algorithm.
pub trait Booster<F> {
    /// A main function that runs boosting algorithm.
    /// This method takes
    /// 
    /// - the reference of 
    ///     an instance of the [`WeakLearner`](WeakLearner) trait,
    /// - a reference of the training examples, and
    fn run<W>(
        &mut self,
        weak_learner: &W,
    ) -> CombinedHypothesis<F>
        where W: WeakLearner<Hypothesis = F>
    {
        self.preprocess(weak_learner);

        for it in 1.. {
            let state = self.boost(weak_learner, it);

            if state == State::Terminate {
                break;
            }
        }

        self.postprocess(weak_learner)
    }


    /// Pre-processing for `self`.
    /// As you can see in [`Booster::run`](Booster::run),
    /// This method is called before the boosting process.
    fn preprocess<W>(
        &mut self,
        weak_learner: &W,
    )
        where W: WeakLearner<Hypothesis = F>;


    /// Boosting per iteration.
    /// This method returns `true` if the stopping criterion is satisfied,
    /// `false` otherwise.
    fn boost<W>(
        &mut self,
        weak_learner: &W,
        iteration: usize,
    ) -> State
        where W: WeakLearner<Hypothesis = F>;


    /// Post-processing.
    /// This method returns a [`CombinedHypothesis<F>`](CombinedHypothesis).
    fn postprocess<W>(
        &mut self,
        weak_learner: &W,
    ) -> CombinedHypothesis<F>
        where W: WeakLearner<Hypothesis = F>;
}

