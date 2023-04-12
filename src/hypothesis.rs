//! The core library for `Hypothesis` traits.
use serde::{Serialize, Deserialize};
use crate::{
    common::utils,
    Sample,
};


/// A trait that defines the behavor of classifier.
/// You only need to implement `confidence` method.
pub trait Classifier {
    /// Computes the confidence of the i'th row of the `df`.
    /// This code assumes that
    /// `Classifier::confidence` returns a value in `[-1.0, 1.0]`.
    /// Those hypotheses are called as **confidence-rated hypotheses**.
    fn confidence(&self, sample: &Sample, row: usize) -> f64;


    /// Predicts the label of the i'th row of the `df`.
    fn predict(&self, sample: &Sample, row: usize) -> i64 {
        let conf = self.confidence(sample, row);
        if conf >= 0.0 { 1 } else { -1 }
    }


    /// Computes the confidence of `df`.
    fn confidence_all(&self, sample: &Sample) -> Vec<f64> {
        let n_sample = sample.shape().0;
        (0..n_sample).map(|row| self.confidence(sample, row))
            .collect::<Vec<_>>()
    }


    /// Predicts the labels of `df`.
    fn predict_all(&self, sample: &Sample) -> Vec<i64>
    {
        let n_sample = sample.shape().0;
        (0..n_sample).map(|row| self.predict(sample, row))
            .collect::<Vec<_>>()
    }
}


/// A trait that defines the behavor of regressor.
/// You only need to implement `predict` method.
pub trait Regressor {
    /// Predicts the target value of the i'th row of the `df`.
    fn predict(&self, sample: &Sample, row: usize) -> f64;


    /// Predicts the labels of `df`.
    fn predict_all(&self, sample: &Sample) -> Vec<f64>
    {
        let n_sample = sample.shape().0;
        (0..n_sample).map(|row| self.predict(sample, row))
            .collect::<Vec<_>>()
    }
}


/// A struct that the boosting algorithms in this library return.
/// You can read/write this struct by `Serde` trait.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CombinedHypothesis<H> {
    /// Weights on each hypothesis in `self.hypotheses`.
    pub weights: Vec<f64>,
    /// Set of hypotheses.
    pub hypotheses: Vec<H>,
}


impl<H: Clone> CombinedHypothesis<H> {
    /// Construct a new `CombinedHypothesis` from given slices.
    #[inline]
    pub fn from_slices(weights: &[f64], hypotheses: &[H]) -> Self {
        let mut new_weights = Vec::with_capacity(weights.len());
        let mut new_hypotheses = Vec::with_capacity(hypotheses.len());

        weights.iter()
            .copied()
            .zip(hypotheses)
            .for_each(|(w, h)| {
                if w > 0.0 {
                    new_weights.push(w);
                    new_hypotheses.push(h.clone());
                }
            });
        utils::normalize(&mut new_weights[..]);


        Self { weights: new_weights, hypotheses: new_hypotheses, }
    }
}

impl<H> CombinedHypothesis<H> {
    /// Append a pair `(weight, F)` to the current combined hypothesis.
    #[inline]
    pub fn push(&mut self, weight: f64, hypothesis: H) {
        self.weights.push(weight);
        self.hypotheses.push(hypothesis);
    }


    /// Normalize `self.weights`, `\| w \|_1 = 1`.
    #[inline]
    pub fn normalize(&mut self) {
        let norm = self.weights.iter()
            .map(|w| w.abs())
            .sum::<f64>();

        if norm <= 0.0 { return; }

        self.weights.iter_mut()
            .for_each(|w| { *w /= norm; });
    }


    /// Decompose the combined hypothesis
    /// into the two vectors `Vec<f64>` and `Vec<F>`
    #[inline]
    pub fn decompose(self) -> (Vec<f64>, Vec<H>) {
        (self.weights, self.hypotheses)
    }
}


impl<F> Classifier for CombinedHypothesis<F>
    where F: Classifier,
{
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        self.weights.iter()
            .zip(&self.hypotheses[..])
            .map(|(w, h)| *w * h.confidence(sample, row))
            .sum::<f64>()
    }
}


impl<F> Regressor for CombinedHypothesis<F>
    where F: Regressor,
{
    fn predict(&self, sample: &Sample, row: usize) -> f64 {
        self.weights.iter()
            .zip(&self.hypotheses[..])
            .map(|(w, h)| *w * h.predict(sample, row))
            .sum::<f64>()
    }
}

