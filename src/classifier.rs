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
pub trait Classifier<T> {

    /// Predicts the label of the given example of type `T`.
    fn predict(&self, example: &T) -> Label;


    /// Predicts the labels of the given examples of type `T`.
    fn predict_all(&self, examples: &[T]) -> Vec<Label>
    {
        examples.iter()
                .map(|example| self.predict(example))
                .collect()
    }
}

use std::marker::PhantomData;

/// A struct that the boosting algorithms in this library return.
/// You can read/write this struct by `Serde` trait.
#[derive(Serialize, Deserialize, Debug)]
pub struct CombinedClassifier<D, C>
//     where D: Data,
//           C: Classifier<D>
{
    /// Each element is the pair of hypothesis and its weight
    pub inner: Vec<(f64, C)>,
    _phantom:  PhantomData<D>,
}


impl<D, C> From<Vec<(f64, C)>> for CombinedClassifier<D, C>
//     where D: Data,
//           C: Classifier<D>,
{
    fn from(inner: Vec<(f64, C)>) -> Self {
        CombinedClassifier {
            inner,
            _phantom: PhantomData
        }
    }
}


impl<D, C> Classifier<D> for CombinedClassifier<D, C>
    where D: Data,
          C: Classifier<D>,
{
    fn predict(&self, example: &D) -> Label
    {
        self.inner
            .iter()
            .fold(0.0, |acc, (w, h)| acc + *w * h.predict(example))
            .signum()
    }
}


