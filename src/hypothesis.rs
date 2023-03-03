//! The core library for `Hypothesis` traits.
use serde::{Serialize, Deserialize};
use crate::Sample;


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
        (0..n_sample).into_iter()
            .map(|row| self.confidence(sample, row))
            .collect::<Vec<_>>()
    }


    /// Predicts the labels of `df`.
    fn predict_all(&self, sample: &Sample) -> Vec<i64>
    {
        let n_sample = sample.shape().0;
        (0..n_sample).into_iter()
            .map(|row| self.predict(sample, row))
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
        (0..n_sample).into_iter()
            .map(|row| self.predict(sample, row))
            .collect::<Vec<_>>()
    }
}


/// A struct that the boosting algorithms in this library return.
/// You can read/write this struct by `Serde` trait.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CombinedHypothesis<F> {
    /// Each element is the pair of hypothesis and its weight
    pub inner: Vec<(f64, F)>,
}


impl<F> CombinedHypothesis<F> {
    /// Append a pair `(weight, F)` to the current combined hypothesis.
    #[inline]
    pub fn push(&mut self, weight: f64, hypothesis: F) {
        self.inner.push((weight, hypothesis));
    }


    /// Normalize `self.weights`, `\| w \|_1 = 1`.
    #[inline]
    pub fn normalize(&mut self) {
        let norm = self.inner.iter()
            .map(|(w, _)| w.abs())
            .sum::<f64>();

        if norm <= 0.0 { return; }

        self.inner.iter_mut()
            .for_each(|(w, _)| {
                *w /= norm;
            });
    }


    /// Decompose the combined hypothesis
    /// into the two vectors `Vec<f64>` and `Vec<F>`
    #[inline]
    pub fn decompose(self) -> (Vec<f64>, Vec<F>) {
        let n_hypotheses = self.inner.len();
        let mut weights = Vec::with_capacity(n_hypotheses);
        let mut hypotheses = Vec::with_capacity(n_hypotheses);

        self.inner.into_iter()
            .for_each(|(w, h)| {
                weights.push(w);
                hypotheses.push(h);
            });

        (weights, hypotheses)
    }
}


impl<F> From<Vec<(f64, F)>> for CombinedHypothesis<F>
{
    fn from(inner: Vec<(f64, F)>) -> Self {
        CombinedHypothesis {
            inner,
        }
    }
}


impl<F> Classifier for CombinedHypothesis<F>
    where F: Classifier,
{
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        self.inner.iter()
            .map(|(w, h)| *w * h.confidence(sample, row))
            .sum::<f64>()
    }
}


impl<F> Regressor for CombinedHypothesis<F>
    where F: Regressor,
{
    fn predict(&self, sample: &Sample, row: usize) -> f64 {
        self.inner.iter()
            .map(|(w, h)| *w * h.predict(sample, row))
            .sum::<f64>()
    }
}

