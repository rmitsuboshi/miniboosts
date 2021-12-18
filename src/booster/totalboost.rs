/// This file defines `TotalBoost` based on the paper
/// "Totally Corrective Boosting Algorithms that Maximize the Margin"
/// by Warmuth et al.
/// 
use crate::data_type::{Data, Label, Sample};
use crate::booster::core::Booster;
use crate::base_learner::core::Classifier;
use crate::base_learner::core::BaseLearner;

use super::softboost::SoftBoost;


/// Since we can regard TotalBoost as a special case of SoftBoost (with capping param is 1.0),
/// so that we use it.
pub struct TotalBoost<D, L> {
    softboost: SoftBoost<D, L>
}


impl<D, L> TotalBoost<D, L> {
    pub fn init(sample: &Sample<D, L>) -> TotalBoost<D, L> {
        let softboost = SoftBoost::init(sample);

        TotalBoost { softboost }
    }
}


impl<D> Booster<D, f64> for TotalBoost<D, f64> {
    fn update_params(&mut self, h: Box<dyn Classifier<D, f64>>, sample: &Sample<D, f64>) -> Option<()> {
        self.softboost.update_params(h, sample)
    }

    fn run(&mut self, base_learner: Box<dyn BaseLearner<D, f64>>, sample: &Sample<D, f64>, eps: f64) {
        self.softboost.run(base_learner, sample, eps);
    }


    fn predict(&self, example: &Data<D>) -> Label<f64> {
        self.softboost.predict(&example)
    }

}
