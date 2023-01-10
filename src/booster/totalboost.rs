//! This file defines `TotalBoost` based on the paper
//! "Totally Corrective Boosting Algorithms that Maximize the Margin"
//! by Warmuth et al.
//! 
use polars::prelude::*;


use crate::{
    Booster,
    WeakLearner,

    State,
    Classifier,
    CombinedHypothesis,

    SoftBoost,
};

use crate::research::Logger;


/// Since we can regard TotalBoost as
/// a special case of SoftBoost (with capping param is 1.0),
/// so that we use it.
pub struct TotalBoost<'a, F> {
    softboost: SoftBoost<'a, F>,
}


impl<'a, F> TotalBoost<'a, F>
    where F: Classifier,
{
    /// initialize the `TotalBoost`.
    pub fn init(data: &'a DataFrame, target: &'a Series) -> Self {
        let softboost = SoftBoost::init(data, target)
            .nu(1.0);

        TotalBoost { softboost }
    }


    /// Returns a optimal value of the optimization problem LPBoost solves
    pub fn opt_val(&self) -> f64 {
        self.softboost.opt_val()
    }


    /// Set the tolerance parameter.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.softboost = self.softboost.tolerance(tol);
        self
    }
}


impl<F> Booster<F> for TotalBoost<'_, F>
    where F: Classifier + Clone,
{
    fn preprocess<W>(
        &mut self,
        weak_learner: &W,
    )
        where W: WeakLearner<Hypothesis = F>
    {
        self.softboost.preprocess(weak_learner);
    }


    fn boost<W>(
        &mut self,
        weak_learner: &W,
        iteration: usize,
    ) -> State
        where W: WeakLearner<Hypothesis = F>
    {
        self.softboost.boost(weak_learner, iteration)
    }


    fn postprocess<W>(
        &mut self,
        weak_learner: &W,
    ) -> CombinedHypothesis<F>
        where W: WeakLearner<Hypothesis = F>
    {
        self.softboost.postprocess(weak_learner)
    }
}


impl<F> Logger for TotalBoost<'_, F>
    where F: Classifier
{
    fn weights_on_hypotheses(&mut self) {
        self.softboost.weights_on_hypotheses();
    }

    /// AdaBoost optimizes the exp loss
    fn objective_value(&self)
        -> f64
    {
        self.softboost.objective_value()
    }


    fn prediction(&self, data: &DataFrame, i: usize) -> f64 {
        self.softboost.prediction(data, i)
    }


    fn logging<L>(
        &self,
        loss_function: &L,
        test_data: &DataFrame,
        test_target: &Series,
    ) -> (f64, f64, f64)
        where L: Fn(f64, f64) -> f64
    {
        self.softboost.logging(loss_function, test_data, test_target)
    }
}
