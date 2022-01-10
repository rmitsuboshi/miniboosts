//! Provides two traits.
use crate::data_type::{Data, Label, Sample};

/// A trait that defines the function used in the combined classifier
/// of the boosting algorithms.
pub trait Classifier<D, L> {

    /// Predicts the label of the given example.
    fn predict(&self, example: &Data<D>) -> Label<L>;


    /// Predicts the labels of the given examples.
    fn predict_all(&self, examples: &[Data<D>]) -> Vec<Label<L>> {
        examples.iter()
                .map(|example| self.predict(&example))
                .collect()
    }
}


/// An interface that returns a function that implements 
/// the `Classifier<D, L>` trait.
pub trait BaseLearner<D, L> {

    /// Returns an instance of the `Classifier<D, L>` trait
    /// that maximizes the edge of the `sample` with respect to `distribution`.
    fn best_hypothesis(&self, sample: &Sample<D, L>, distribution: &[f64])
        -> Box<dyn Classifier<D, L>>;
}

