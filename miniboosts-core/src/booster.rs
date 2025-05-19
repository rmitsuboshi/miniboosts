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
/// # Required Methods
/// - [`Booster::name`]
/// - [`Booster::preprocess`]
/// - [`Booster::boost`]
/// - [`Booster::postprocess`]
/// - [`Booster::info`] ... optional.
///
/// # Provided Methods
/// - [`Booster::run`]
pub trait Booster<H> {
    /// The final hypothesis output by a boosting algorithm.
    type Output;

    /// Returns the name of the boosting algorithm.
    fn name(&self) -> &str;

    /// Returns the information of boosting algorithm as `String`.
    fn info(&self) -> Option<Vec<(&str, String)>> {
        None
    }
    /// A main function that runs boosting algorithm.
    fn run<W>(&mut self, weak_learner: &W) -> Self::Output
        where W: WeakLearner<Hypothesis = H>
    {
        self.preprocess();

        (1..).try_for_each(|iter|
            self.boost(weak_learner, iter)
        );

        self.postprocess()
    }

    /// Pre-processing for `self`.
    /// As you can see in [`Booster::run`],
    /// this method is called before the boosting process.
    fn preprocess(&mut self);

    /// Boosting step per iteration.
    fn boost<W>(
        &mut self,
        weak_learner: &W,
        iteration: usize,
    ) -> ControlFlow<usize>
        where W: WeakLearner<Hypothesis = H>;

    /// Post-processing.
    fn postprocess(&mut self) -> Self::Output;
}

