//! Provides `Booster` trait.

use crate::WeakLearner;

use std::ops::ControlFlow;


/// The trait [`Booster`] defines the standard framework of Boosting.
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
/// - [`Booster::preprocess`],
/// - [`Booster::boost`], and
/// - [`Booster::postprocess`]
/// 
/// to write a new boosting algorithm.
pub trait Booster<H> {
    /// Returns the name of the boosting algorithm.
    fn name(&self) -> &str;



    /// Returns the information of boosting algorithm as `String`.
    /// This method is used for [`Logger`](crate::research::Logger).
    /// By default, this method returns `None`.
    fn info(&self) -> Option<String> {
        None
    }


    /// The final hypothesis output by a boosting algorithm.
    /// Most algorithms return [`CombinedHypothesis<H>`](crate::hypothesis::CombinedHypothesis),
    /// which is a weighted majority vote of base hypotheses.
    type Output;
    /// A main function that runs boosting algorithm.
    fn run<W>(
        &mut self,
        weak_learner: &W,
    ) -> Self::Output
        where W: WeakLearner<Hypothesis = H>
    {
        self.preprocess(weak_learner);

        let _ = (1..).try_for_each(|iter| {
            self.boost(weak_learner, iter)
        });

        self.postprocess(weak_learner)
    }


    /// Pre-processing for `self`.
    /// As you can see in [`Booster::run`],
    /// this method is called before the boosting process.
    fn preprocess<W>(
        &mut self,
        weak_learner: &W,
    )
        where W: WeakLearner<Hypothesis = H>;


    /// Boosting step per iteration.
    /// This method returns 
    /// `ControlFlow::Continue(())` if the stopping criterion is satisfied,
    /// `ControlFlow::Break(terminated_iter)` otherwise.  
    fn boost<W>(
        &mut self,
        weak_learner: &W,
        iteration: usize,
    ) -> ControlFlow<usize>
        where W: WeakLearner<Hypothesis = H>;


    /// Post-processing.
    /// This method returns a [`CombinedHypothesis<H>`](crate::hypothesis::CombinedHypothesis).
    fn postprocess<W>(
        &mut self,
        weak_learner: &W,
    ) -> Self::Output
        where W: WeakLearner<Hypothesis = H>;
}

