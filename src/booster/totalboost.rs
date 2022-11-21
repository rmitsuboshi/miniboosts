//! This file defines `TotalBoost` based on the paper
//! "Totally Corrective Boosting Algorithms that Maximize the Margin"
//! by Warmuth et al.
//! 
use polars::prelude::*;


use crate::{
    Booster,
    BaseLearner,

    State,
    Classifier,
    CombinedClassifier,

    SoftBoost,
};


/// Since we can regard TotalBoost as
/// a special case of SoftBoost (with capping param is 1.0),
/// so that we use it.
pub struct TotalBoost<C> {
    softboost: SoftBoost<C>,
}


impl<C> TotalBoost<C>
    where C: Classifier,
{
    /// initialize the `TotalBoost`.
    pub fn init(data: &DataFrame, target: &Series) -> Self {
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


impl<C> Booster<C> for TotalBoost<C>
    where C: Classifier + Clone,
{
    fn preprocess<B>(
        &mut self,
        base_learner: &B,
        data: &DataFrame,
        target: &Series,
    )
        where B: BaseLearner<Clf = C>
    {
        self.softboost.preprocess(base_learner, data, target);
    }


    fn boost<B>(
        &mut self,
        base_learner: &B,
        data: &DataFrame,
        target: &Series,
        iteration: usize,
    ) -> State
        where B: BaseLearner<Clf = C>
    {
        self.softboost.boost(base_learner, data, target, iteration)
    }


    fn postprocess<B>(
        &mut self,
        base_learner: &B,
        data: &DataFrame,
        target: &Series,
    ) -> CombinedClassifier<C>
        where B: BaseLearner<Clf = C>
    {
        self.softboost.postprocess(base_learner, data, target)
    }
}
