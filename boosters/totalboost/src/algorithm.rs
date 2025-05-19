//! This file defines `TotalBoost` based on the paper
//! "Totally Corrective Boosting Algorithms that Maximize the Margin"
//! by Warmuth et al.
//! 
use softboost::SoftBoost;
use miniboosts_core::{
    Sample,
    Booster,
    WeakLearner,
    Classifier,
};
use hypotheses::WeightedMajority;
use logging::CurrentHypothesis;

use std::ops::ControlFlow;

pub struct TotalBoost<'a, H> {
    softboost: SoftBoost<'a, H>,
}

impl<'a, H> TotalBoost<'a, H>
    where H: Classifier,
{
    /// Construct a new instance of `TotalBoost`.
    /// 
    /// Time complexity: `O(1)`.
    pub fn init(sample: &'a Sample) -> Self {
        let softboost = SoftBoost::init(sample).nu(1f64);

        Self { softboost }
    }

    /// Set the tolerance parameter.
    /// 
    /// Time complexity: `O(1)`.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.softboost = self.softboost.tolerance(tol);
        self
    }
}

impl<H> Booster<H> for TotalBoost<'_, H>
    where H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;

    fn name(&self) -> &str { "TotalBoost" }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        self.softboost.info()
    }

    fn preprocess(&mut self) {
        self.softboost.preprocess();
    }

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize)
        -> ControlFlow<usize>
        where W: WeakLearner<Hypothesis = H>
    {
        self.softboost.boost(weak_learner, iteration)
    }

    fn postprocess(&mut self) -> Self::Output {
        self.softboost.postprocess()
    }
}

impl<H> CurrentHypothesis for TotalBoost<'_, H>
    where H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        self.softboost.current_hypothesis()
    }
}


