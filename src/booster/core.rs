//! Provides the trait `Booster<D, L>`.
use crate::data_type::{Data, Label, Sample};
use crate::base_learner::core::Classifier;
use crate::base_learner::core::BaseLearner;

/// The trait `Booster` defines the standard framework of Boosting.
/// 
/// You need to implement `update_params`, `run`, and `predict`
/// in order to write a new boosting algorithm.
pub trait Booster<D, L> {

    /// Updates the parameters of `self`.
    /// The boosting algorithms implemented in this crate
    /// terminates if `update_params` returns `None`.
    fn update_params(&mut self,
                     h: Box<dyn Classifier<D, L>>,
                     sample: &Sample<D, L>)
        -> Option<()>;


    /// A main function that runs boosting algorithm.
    /// This method takes
    /// 
    /// - the reference of an instance of the `BaseLearner<D, L>` trait,
    /// - a reference of the training examples, and
    /// - a tolerance parameter.
    fn run(&mut self,
           base_learner: &dyn BaseLearner<D, L>,
           sample:       &Sample<D, L>,
           tolerance:    f64);


    /// Predicts the label of the given example by the combined hypothesis.
    /// Panics if you call this method before calling `run`.
    fn predict(&self, example: &Data<D>) -> Label<L>;


    /// Predicts the labels of given examples.
    fn predict_all(&self, examples: &[Data<D>]) -> Vec<Label<L>> {
        examples.iter()
                .map(|example| self.predict(&example))
                .collect()
    }
}
