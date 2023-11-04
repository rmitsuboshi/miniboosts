//! The core library for `Hypothesis` traits.
use fixedbitset::FixedBitSet;
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
        utils::normalize(&mut self.weights);
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


/// The naive aggregation rule.
/// See the following paper for example:
///
/// [Noga Alon, Alon Gonen, Elad Hazan, and Shay Moran. Boosting Simple Learners](https://theoretics.episciences.org/10757)
///
///
/// # Description
/// Given a set of hypotheses `{ h1, h2, ..., hT } \subset {-1, +1}^X` 
/// and training instances `(x1, y1), (x2, y2), ..., (xm, ym)`,
/// one can construct the following table:
///
/// ```txt
///              | h1(x)   h2(x)   h3(x) ... hT(x) | y
///     (x1, y1) |   +       -       +   ...   +   | -
///     (x2, y2) |   -       +       +   ...   -   | +
///        ...   |                       ...       |
///     (xm, ym) |   -       -       +   ...   -   | -
/// ```
///
/// Given a new instance `x`, 
/// we can get the following binary sequence of length `T`.
///
/// ```txt
/// B := h1(x) h2(x) ... hT(x) = +-+-....-+--+
/// ```
/// The following is the behavior of `NaiveAggregation`:
/// 1. If there exists an instance (xk, yk) such that
///    `h1(xk) h2(xk) ... hT(xk) == B` and `yk == +`,
///    the predict `+`.
/// 2. If there is no instance `(xk, yk)` satisfying the above condition,
///    the predict `-`.
/// 
pub struct NaiveAggregation<H> {
    /// `hypotheses` stores the functions from `X` to `{-1, +1}`
    /// collected by the boosting algorithm.
    hypotheses: Vec<H>,
    /// `prediction` stores a bit sequence
    /// `h1(xk) h2(xk) ... hT(xk)` whose label `yk` is `+1`.
    prediction: Vec<FixedBitSet>,
}


impl<H> NaiveAggregation<H>
    where H: Classifier
{
    /// Construct a new instance of `NaiveAggregation<H>`.
    #[inline(always)]
    pub fn new(
        hypotheses: Vec<H>,
        sample: &Sample,
    ) -> Self
    {
        let targets = sample.target();
        let n_hypotheses = hypotheses.len();
        let n_pos = targets.iter()
            .copied()
            .filter(|y| *y > 0.0)
            .count();
        let mut prediction = Vec::with_capacity(n_pos);

        let iter = targets.iter()
            .copied()
            .enumerate()
            .filter_map(|(i, y)| if y > 0.0 { Some(i) } else { None });
        for i in iter {
            let mut bits = FixedBitSet::with_capacity(n_hypotheses);
            hypotheses.iter()
                .enumerate()
                .for_each(|(t, h)| {
                    if h.predict(sample, i) > 0 {
                        bits.put(t);
                    }
                });
            prediction.push(bits);
        }
        Self { hypotheses, prediction }
    }
}


impl<H> NaiveAggregation<H>
    where H: Classifier + Clone
{
    /// Construct a new instance of `NaiveAggregation<H>`
    /// from a slice of hypotheses and `sample`.
    #[inline(always)]
    pub fn from_slice(
        hypotheses: &[H],
        sample: &Sample,
    ) -> Self
    {
        let hypotheses = hypotheses.to_vec();
        Self::new(hypotheses, sample)
    }
}



impl<H> Classifier for NaiveAggregation<H>
    where H: Classifier
{
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        let n_hypotheses = self.hypotheses.len();
        let mut bits = FixedBitSet::with_capacity(n_hypotheses);
        self.hypotheses.iter()
            .enumerate()
            .for_each(|(t, h)| {
                if h.predict(sample, row) == 1 {
                    bits.put(t);
                }
            });

        if self.prediction.iter().any(|p| p.eq(&bits)) { 1.0 } else { -1.0 }
    }
}
