//! An additive model `F(x) = c_0 + Σ_t c_t f_t(x)` of real-valued
//! hypotheses, returned by [`GradientBoosting`](crate::GradientBoosting)
//! and [`LogitBoost`](crate::LogitBoost).

use crate::core::{Classifier, Regressor, Sample};
use serde::{Deserialize, Serialize};

/// How an [`AdditiveModel`] turns its score `F(x)` into a confidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreLink {
    /// `tanh(F)`: for a score that is half the log-odds,
    /// `F = (1/2) ln(P(+1|x) / P(-1|x))`, this is `P(+1|x) - P(-1|x)`.
    Tanh,
    /// `F` clamped to `[-1, 1]`.
    Clamp,
}

/// The additive model `F(x) = c_0 + Σ_t c_t f_t(x)` with intercept `c_0`.
///
/// As a [`Regressor`] it predicts the score `F(x)`; as a [`Classifier`] its
/// confidence is `F(x)` mapped to `[-1, 1]` by its [`ScoreLink`], so its sign
/// (and prediction) is that of `F(x)`.
///
/// # Examples
/// ```
/// use miniboosts::{Classifier, Regressor, Sample};
/// use miniboosts::hypotheses::{AdditiveModel, ScoreLink};
///
/// struct Constant(f64);
/// impl Regressor for Constant {
///     fn predict(&self, _: &Sample, _: usize) -> f64 {
///         self.0
///     }
/// }
///
/// let sample = Sample::dummy(1);
/// let mut f = AdditiveModel::new(0.25, ScoreLink::Tanh);
/// f.push(0.5, Constant(1.0));
/// f.push(0.5, Constant(-0.5));
/// // F = 0.25 + 0.5 - 0.25 = 0.5.
/// assert_eq!(Regressor::predict(&f, &sample, 0), 0.5);
/// assert!((f.confidence(&sample, 0) - 0.5f64.tanh()).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdditiveModel<H> {
    /// The intercept `c_0`.
    pub intercept: f64,
    /// The coefficients `c_t`; `coefficients[t]` belongs to `hypotheses[t]`.
    pub coefficients: Vec<f64>,
    /// The hypotheses `f_t`.
    pub hypotheses: Vec<H>,
    /// The map from the score to a confidence.
    pub link: ScoreLink,
}

impl<H> AdditiveModel<H> {
    /// Constructs the constant model `F = intercept`.
    ///
    /// # Examples
    /// See [`AdditiveModel`].
    pub fn new(intercept: f64, link: ScoreLink) -> Self {
        Self {
            intercept,
            coefficients: Vec::new(),
            hypotheses: Vec::new(),
            link,
        }
    }

    /// Appends `coefficient · hypothesis`.
    ///
    /// # Examples
    /// See [`AdditiveModel`].
    pub fn push(&mut self, coefficient: f64, hypothesis: H) {
        self.coefficients.push(coefficient);
        self.hypotheses.push(hypothesis);
    }
}

impl<H: Regressor> Regressor for AdditiveModel<H> {
    /// Returns the score `F(x)`.
    fn predict(&self, sample: &Sample, row: usize) -> f64 {
        self.intercept
            + self
                .coefficients
                .iter()
                .zip(&self.hypotheses)
                .map(|(c, h)| c * h.predict(sample, row))
                .sum::<f64>()
    }
}

impl<H: Regressor> Classifier for AdditiveModel<H> {
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        let score = Regressor::predict(self, sample, row);
        match self.link {
            ScoreLink::Tanh => score.tanh(),
            ScoreLink::Clamp => score.clamp(-1.0, 1.0),
        }
    }
}
