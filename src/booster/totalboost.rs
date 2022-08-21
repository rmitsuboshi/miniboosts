//! This file defines `TotalBoost` based on the paper
//! "Totally Corrective Boosting Algorithms that Maximize the Margin"
//! by Warmuth et al.
//! 
use polars::prelude::*;


use crate::{Classifier, CombinedClassifier};
use crate::BaseLearner;
use crate::Booster;

use super::softboost::SoftBoost;


/// Since we can regard TotalBoost as
/// a special case of SoftBoost (with capping param is 1.0),
/// so that we use it.
pub struct TotalBoost {
    softboost: SoftBoost
}


impl TotalBoost {
    /// initialize the `TotalBoost`.
    pub fn init(df: &DataFrame) -> Self {
        let softboost = SoftBoost::init(df)
            .capping(1.0);

        TotalBoost { softboost }
    }


    /// Returns a optimal value of the optimization problem LPBoost solves
    pub fn opt_val(&self) -> f64 {
        self.softboost.opt_val()
    }
}


impl<C> Booster<C> for TotalBoost
    where C: Classifier,
{
    fn run<B>(&mut self,
              base_learner: &B,
              data: &DataFrame,
              target: &Series,
              eps: f64)
        -> CombinedClassifier<C>
        where B: BaseLearner<Clf = C>,
    {
        self.softboost.run(base_learner, data, target, eps)
    }
}
