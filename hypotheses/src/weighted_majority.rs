use serde::{Serialize, Deserialize};
use miniboosts_core::{
    tools::helpers,
    Classifier,
    Regressor,
    Sample,
};

/// A weighted majority classifier that takes references of
/// weights and hypotheses.
pub struct RefWeightedMajority<'a, H> {
    /// Weights on each hypothesis in `self.hypotheses`.
    pub weights: &'a [f64],
    /// Set of hypotheses.
    pub hypotheses: &'a [H],
}

impl<'a, H> RefWeightedMajority<'a, H> {
    /// Construct a new `WeightedMajority` from given slices.
    #[inline]
    pub fn new(weights: &'a [f64], hypotheses: &'a [H]) -> Self {
        Self { weights, hypotheses, }
    }
}

impl<H> Classifier for RefWeightedMajority<'_, H>
    where H: Classifier,
{
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        self.weights.iter()
            .zip(self.hypotheses)
            .map(|(w, h)| *w * h.confidence(sample, row))
            .sum::<f64>()
    }
}

impl<H> Regressor for RefWeightedMajority<'_, H>
    where H: Regressor,
{
    fn predict(&self, sample: &Sample, row: usize) -> f64 {
        self.weights.iter()
            .zip(self.hypotheses)
            .map(|(w, h)| *w * h.predict(sample, row))
            .sum::<f64>()
    }
}

/// A struct that the boosting algorithms in this library return.
/// You can read/write this struct by `Serde` trait.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WeightedMajority<H> {
    /// Weights on each hypothesis in `self.hypotheses`.
    pub weights: Vec<f64>,
    /// Set of hypotheses.
    pub hypotheses: Vec<H>,
}

impl<H: Clone> WeightedMajority<H> {
    /// Construct a new `WeightedMajority` from given slices.
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
        helpers::normalize(&mut new_weights[..]);

        Self { weights: new_weights, hypotheses: new_hypotheses, }
    }
}

impl<H> WeightedMajority<H> {
    /// Append a pair `(weight, F)` to the current combined hypothesis.
    #[inline]
    pub fn push(&mut self, weight: f64, hypothesis: H) {
        self.weights.push(weight);
        self.hypotheses.push(hypothesis);
    }

    /// Normalize `self.weights`, `\| w \|_1 = 1`.
    #[inline]
    pub fn normalize(&mut self) {
        helpers::normalize(&mut self.weights);
    }

    /// Decompose the combined hypothesis
    /// into the two vectors `Vec<f64>` and `Vec<F>`
    #[inline]
    pub fn decompose(self) -> (Vec<f64>, Vec<H>) {
        (self.weights, self.hypotheses)
    }
}

impl<H> Classifier for WeightedMajority<H>
    where H: Classifier,
{
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        self.weights.iter()
            .zip(&self.hypotheses[..])
            .map(|(w, h)| *w * h.confidence(sample, row))
            .sum::<f64>()
    }
}

impl<H> Regressor for WeightedMajority<H>
    where H: Regressor,
{
    fn predict(&self, sample: &Sample, row: usize) -> f64 {
        self.weights.iter()
            .zip(&self.hypotheses[..])
            .map(|(w, h)| *w * h.predict(sample, row))
            .sum::<f64>()
    }
}

