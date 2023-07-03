//! Provides `Booster` trait.

use crate::{
    WeakLearner,
    CombinedHypothesis,
};

use std::ops::ControlFlow;


/// The trait [`Booster`](Booster) defines the standard framework of Boosting.
/// Here, the **standard framework** is defined as
/// a repeated game between **Booster** and **Weak Learner**
/// of the following form:
/// 
/// In each round `t = 1, 2, ...`,
/// 1. Booster chooses a probability distribution over
///    training instances.
/// 2. Weak Learner chooses a hypothesis that achieves 
///    some **accuracy** with respect to the distribution.
/// 
/// After sufficient rounds, Booster outputs a combined hypothesis
/// with high accuracy for any probability distribution on training examples.
/// 
/// You need to implement 
/// 
/// - [`Booster::preprocess`](Booster::preprocess),
/// - [`Booster::boost`](Booster::boost), and
/// - [`Booster::postprocess`](Booster::postprocess)
/// 
/// to write a new boosting algorithm.
pub trait Booster<F> {
    /// A main function that runs boosting algorithm.
    fn run<W>(
        &mut self,
        weak_learner: &W,
    ) -> CombinedHypothesis<F>
        where W: WeakLearner<Hypothesis = F>
    {
        self.preprocess(weak_learner);

        let _ = (1..).try_for_each(|iter| {
            self.boost(weak_learner, iter)
        });

        self.postprocess(weak_learner)
    }


    /// Pre-processing for `self`.
    /// As you can see in [`Booster::run`](Booster::run),
    /// this method is called before the boosting process.
    fn preprocess<W>(
        &mut self,
        weak_learner: &W,
    )
        where W: WeakLearner<Hypothesis = F>;


    /// Boosting step per iteration.
    /// This method returns 
    /// `ControlFlow::Continue(())` if the stopping criterion is satisfied,
    /// `ControlFlow::Break(terminated_iter)` otherwise.  
    fn boost<W>(
        &mut self,
        weak_learner: &W,
        iteration: usize,
    ) -> ControlFlow<usize>
        where W: WeakLearner<Hypothesis = F>;


    /// Post-processing.
    /// This method returns a [`CombinedHypothesis<F>`](CombinedHypothesis).
    fn postprocess<W>(
        &mut self,
        weak_learner: &W,
    ) -> CombinedHypothesis<F>
        where W: WeakLearner<Hypothesis = F>;
}

