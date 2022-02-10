//! This file defines `TotalBoost` based on the paper
//! "Totally Corrective Boosting Algorithms that Maximize the Margin"
//! by Warmuth et al.
//! 
use crate::{Data, Sample};
use crate::{Classifier, CombinedClassifier};
use crate::BaseLearner;
use crate::Booster;

use crate::SoftBoost;


/// Since we can regard TotalBoost as
/// a special case of SoftBoost (with capping param is 1.0),
/// so that we use it.
pub struct TotalBoost {
    softboost: SoftBoost
}


impl TotalBoost {
    /// initialize the `TotalBoost`.
    pub fn init<T: Data>(sample: &Sample<T>) -> TotalBoost {
        let softboost = SoftBoost::init(&sample)
            .capping(1.0);

        TotalBoost { softboost }
    }


    /// Returns a optimal value of the optimization problem LPBoost solves
    pub fn opt_val(&self) -> f64 {
        self.softboost.opt_val()
    }
}


impl<D, C> Booster<D, C> for TotalBoost
    where C: Classifier<D>,
          D: Data<Output = f64>,
{
    fn run<B>(&mut self,
              base_learner: &B,
              sample:       &Sample<D>,
              eps:          f64)
        -> CombinedClassifier<D, C>
        where B: BaseLearner<D, Clf = C>,
    {
        self.softboost.run(base_learner, sample, eps)
    }
}
