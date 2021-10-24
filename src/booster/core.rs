use crate::data_type::{Data, Label, Sample};
use crate::base_learner::core::Classifier;
use crate::base_learner::core::BaseLearner;

/// The trait `Booster` defines the standard framework of Boosting.
pub trait Booster<D, L> {

    fn update_params(&mut self, h: Box<dyn Classifier<D, L>>, sample: &Sample<D, L>) -> Option<()>;

    fn run(&mut self, base_learner: Box<dyn BaseLearner<D, L>>, sample: &Sample<D, L>, eps: f64);

    fn predict(&self, example: &Data<D>) -> Label<L>;

    fn predict_all(&self, examples: &[Data<D>]) -> Vec<Label<L>> {
        examples.iter()
                .map(|example| self.predict(&example))
                .collect()
    }
}
