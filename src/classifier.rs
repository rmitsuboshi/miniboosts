//! The core library for the base learner in the boosting protocol.
//! 
//! The base learner in the general boosting setting is as follows:
//! 
//! Given a distribution over training examples,
//! the base learner returns a hypothesis that is slightly better than
//! the random guessing, where the **edge** is the affine transformation of
//! the weighted training error.
//! 
//! In this code, we assume that the base learner returns a hypothesis
//! that **maximizes** the edge for a given distribution.
//! This assumption is stronger than the previous one, but the resulting
//! combined hypothesis becomes much stronger.
//! 
//! I'm planning to implement the code for the general base learner setting.
//! 
use crate::{Data, Label};
use serde::{Serialize, Deserialize};



/// A trait that defines the function used in the combined classifier
/// of the boosting algorithms.
pub trait Classifier {

    /// Predicts the label of the given example.
    fn predict(&self, example: &Data) -> Label;


    /// Predicts the labels of the given examples.
    fn predict_all(&self, examples: &[Data]) -> Vec<Label> {
        examples.iter()
                .map(|example| self.predict(&example))
                .collect()
    }
}



/// A struct that the boosting algorithms in this library return.
/// You can read/write this struct by `serde` trait.
/// TODO USE SERDE TRAIT
#[derive(Serialize, Deserialize, Debug)]
pub struct CombinedClassifier<C: Classifier> {
    /// Each element is the pair of hypothesis and its weight
    pub weighted_classifier: Vec<(f64, C)>
}


impl<C: Classifier> Classifier for CombinedClassifier<C> {
    fn predict(&self, example: &Data) -> Label {
        self.weighted_classifier
            .iter()
            .fold(0.0, |acc, (w, h)| acc + *w * h.predict(&example))
            .signum()
    }
}


