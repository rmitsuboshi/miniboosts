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
// use crate::{Data, Label};
use crate::Data;
use serde::{Serialize, Deserialize};



/// A trait that defines the function used in the combined classifier
/// of the boosting algorithms.
pub trait Classifier<D, L> {

    /// Predicts the label of the given example of type `T`.
    fn predict(&self, example: &D) -> L;


    /// Predicts the labels of the given examples of type `T`.
    fn predict_all(&self, examples: &[D]) -> Vec<L>
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
pub struct CombinedClassifier<D, L, C>
{
    /// Each element is the pair of hypothesis and its weight
    pub inner: Vec<(f64, C)>,
    _phantom:  PhantomData<(D, L)>,
}


impl<D, L, C> From<Vec<(f64, C)>> for CombinedClassifier<D, L, C>
{
    fn from(inner: Vec<(f64, C)>) -> Self {
        CombinedClassifier {
            inner,
            _phantom: PhantomData
        }
    }
}


impl<D, L, C> Classifier<D, L> for CombinedClassifier<D, L, C>
    where D: Data,
          L: Into<f64> + From<f64>,
          C: Classifier<D, L>,
{
    fn predict(&self, example: &D) -> L
    {
        let p = self.inner
            .iter()
            .fold(0.0, |acc, (w, h)| acc + *w * h.predict(example).into())
            .signum();

        L::from(p)
    }
}


