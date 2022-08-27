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
use polars::prelude::*;
use serde::{Serialize, Deserialize};



/// A trait that defines the function used in the combined classifier
/// of the boosting algorithms.
pub trait Classifier {
    /// Computes the confidence of the i'th row of the `df`.
    fn confidence(&self, df: &DataFrame, row: usize) -> f64;


    /// Predicts the label of the i'th row of the `df`.
    fn predict(&self, df: &DataFrame, row: usize) -> i64 {
        self.confidence(df, row).signum() as i64
    }


    /// Computes the confidence of `df`.
    fn confidence_all(&self, df: &DataFrame) -> Vec<f64> {
        let m = df.shape().0;
        (0..m).into_iter()
            .map(|row| self.confidence(df, row))
            .collect::<Vec<_>>()
    }


    /// Predicts the labels of `df`.
    fn predict_all(&self, df: &DataFrame) -> Vec<i64>
    {
        let m = df.shape().0;
        (0..m).into_iter()
            .map(|row| self.predict(df, row))
            .collect::<Vec<_>>()
    }
}


/// A struct that the boosting algorithms in this library return.
/// You can read/write this struct by `Serde` trait.
#[derive(Serialize, Deserialize, Debug)]
pub struct CombinedClassifier<C> {
    /// Each element is the pair of hypothesis and its weight
    pub inner: Vec<(f64, C)>,
}


impl<C> From<Vec<(f64, C)>> for CombinedClassifier<C>
{
    fn from(inner: Vec<(f64, C)>) -> Self {
        CombinedClassifier { inner }
    }
}


impl<C> Classifier for CombinedClassifier<C>
    where C: Classifier,
{
    fn confidence(&self, df: &DataFrame, row: usize) -> f64 {
        self.inner.iter()
            .map(|(w, h)| *w * h.confidence(df, row))
            .sum::<f64>()
    }
}


