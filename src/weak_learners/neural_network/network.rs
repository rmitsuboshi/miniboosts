//! The hypothesis [`TwoLayerNetworkClassifier`] and the forward and backward
//! passes used to train it.

use super::activation::{Activation, logistic};
use crate::core::{Classifier, Feature, Sample};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// How a [`TwoLayerNetworkClassifier`] maps its score `s` to a confidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkOutput {
    /// `tanh(s)`. With the logistic loss, `s` is half the log-odds and
    /// `tanh(s) = P(+1|x) - P(-1|x)`.
    Tanh,
    /// `+1` if `s ≥ 0` and `-1` otherwise.
    Sign,
}

/// The loss minimized by [`TwoLayerNetwork`](super::TwoLayerNetwork), as a
/// function of the label `y ∈ {-1, +1}` and the score `s`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkLoss {
    /// `ln(1 + e^{-2ys})`: the negative log-likelihood when
    /// `P(y|x) = 1 / (1 + e^{-2ys})`.
    Logistic,
    /// `(y - tanh(s))² / 2`.
    Squared,
}

impl NetworkLoss {
    /// The derivative of the loss with respect to `s`.
    pub(crate) fn derivative(self, y: f64, s: f64) -> f64 {
        match self {
            Self::Logistic => -2.0 * y * logistic(-2.0 * y * s),
            Self::Squared => {
                let t = s.tanh();
                -(y - t) * (1.0 - t * t)
            }
        }
    }

    /// The loss itself, used by the tests.
    #[cfg(test)]
    pub(crate) fn value(self, y: f64, s: f64) -> f64 {
        match self {
            Self::Logistic => {
                let z = -2.0 * y * s;
                z.max(0.0) + (-z.abs()).exp().ln_1p()
            }
            Self::Squared => 0.5 * (y - s.tanh()).powi(2),
        }
    }
}

/// A network with one hidden layer of `h` units:
/// ```txt
/// s(x) = Σ_j w_j φ(Σ_k W_jk z_k + b_j) + c,    z_k = (x_k - μ_k) / σ_k,
/// ```
/// where `μ_k` and `σ_k` standardize feature `k`. Its confidence is
/// `tanh(s(x))` or the sign of `s(x)` ([`NetworkOutput`]).
///
/// The features are read by name, so the hypothesis applies to any sample
/// that has them. Values are evaluated in a fixed order, so
/// [`Classifier::confidence_all`] equals [`Classifier::confidence`] bit for
/// bit.
///
/// # Examples
/// See [`TwoLayerNetwork`](super::TwoLayerNetwork).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TwoLayerNetworkClassifier {
    pub(crate) feature_names: Arc<[String]>,
    // μ_k.
    pub(crate) shift: Vec<f64>,
    // 1 / σ_k.
    pub(crate) scale: Vec<f64>,
    // W, row-major: `hidden_weights[j * p + k]` is W_jk.
    pub(crate) hidden_weights: Vec<f64>,
    // b.
    pub(crate) hidden_bias: Vec<f64>,
    // w.
    pub(crate) output_weights: Vec<f64>,
    // c.
    pub(crate) output_bias: f64,
    pub(crate) activation: Activation,
    pub(crate) output: NetworkOutput,
}

impl TwoLayerNetworkClassifier {
    /// Returns the score `s(x)` of the example in `row`.
    ///
    /// # Panics
    /// Panics if `sample` lacks one of the features.
    pub fn score(&self, sample: &Sample, row: usize) -> f64 {
        let x: Vec<f64> = self
            .feature_names
            .iter()
            .map(|name| sample[name][row])
            .collect();
        let mut hidden = vec![0.0; self.output_weights.len()];
        self.score_of_raw(&x, &mut hidden)
    }

    /// The score of the unstandardized example `x`, with `hidden` as scratch
    /// space of length `h`.
    fn score_of_raw(&self, x: &[f64], hidden: &mut [f64]) -> f64 {
        let z: Vec<f64> = x
            .iter()
            .zip(self.shift.iter().zip(&self.scale))
            .map(|(v, (m, s))| (v - m) * s)
            .collect();
        self.score_of_standardized(&z, hidden)
    }

    /// `s(x)` for the standardized example `z`. Fills `hidden` with the
    /// pre-activations `Wz + b`.
    pub(crate) fn score_of_standardized(&self, z: &[f64], hidden: &mut [f64]) -> f64 {
        let p = z.len();
        let mut s = self.output_bias;
        for (j, pre) in hidden.iter_mut().enumerate() {
            let row = &self.hidden_weights[j * p..(j + 1) * p];
            *pre = row.iter().zip(z).map(|(w, v)| w * v).sum::<f64>() + self.hidden_bias[j];
            s += self.output_weights[j] * self.activation.apply(*pre);
        }
        s
    }

    /// The confidence for the score `s`.
    pub(crate) fn link(&self, s: f64) -> f64 {
        match self.output {
            NetworkOutput::Tanh => s.tanh(),
            NetworkOutput::Sign => {
                if s >= 0.0 {
                    1.0
                } else {
                    -1.0
                }
            }
        }
    }

    /// Negates the score: `s ↦ -s` (the output weights and bias change
    /// sign). With either output, the confidence changes sign as well,
    /// except where `s = 0` exactly under [`NetworkOutput::Sign`].
    pub(crate) fn negate(&mut self) {
        for w in &mut self.output_weights {
            *w = -*w;
        }
        self.output_bias = -self.output_bias;
    }

    /// Returns the number `h` of hidden units.
    pub fn hidden_units(&self) -> usize {
        self.output_weights.len()
    }

    /// Returns the names of the input features.
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Returns `(μ_k, 1/σ_k)` for every feature: `z_k = (x_k - μ_k) / σ_k`.
    /// A feature that is constant on the training rows has `1/σ_k = 1`.
    pub fn standardization(&self) -> (&[f64], &[f64]) {
        (&self.shift, &self.scale)
    }

    /// Returns the hidden weights `W` (row-major, one row of `p` entries per
    /// hidden unit) and biases `b`.
    pub fn hidden_layer(&self) -> (&[f64], &[f64]) {
        (&self.hidden_weights, &self.hidden_bias)
    }

    /// Returns the output weights `w` and bias `c`.
    pub fn output_layer(&self) -> (&[f64], f64) {
        (&self.output_weights, self.output_bias)
    }

    /// Returns the activation function `φ`.
    pub fn activation(&self) -> Activation {
        self.activation
    }

    /// Returns the output map.
    pub fn output(&self) -> NetworkOutput {
        self.output
    }
}

impl Classifier for TwoLayerNetworkClassifier {
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        self.link(self.score(sample, row))
    }

    /// Same values as [`Classifier::confidence`] row by row, with every
    /// feature column read once.
    fn confidence_all(&self, sample: &Sample) -> Vec<f64> {
        let n_sample = sample.shape().0;
        let rows = raw_rows(&sample.columns(&self.feature_names), n_sample, None);
        let mut hidden = vec![0.0; self.hidden_units()];
        rows.iter()
            .map(|x| self.link(self.score_of_raw(x, &mut hidden)))
            .collect()
    }
}

/// The values of `columns` for every row, or for the rows in `only` (in that
/// order), as one vector per row.
pub(crate) fn raw_rows(
    columns: &[&Feature],
    n_sample: usize,
    only: Option<&[usize]>,
) -> Vec<Vec<f64>> {
    let p = columns.len();
    let n = only.map_or(n_sample, <[usize]>::len);
    // Position of each sample row among the selected rows.
    let position: Option<Vec<usize>> = only.map(|rows| {
        let mut position = vec![usize::MAX; n_sample];
        for (i, &row) in rows.iter().enumerate() {
            position[row] = i;
        }
        position
    });
    let mut out = vec![vec![0.0; p]; n];
    for (k, column) in columns.iter().enumerate() {
        match column {
            Feature::Dense { vals, .. } => match only {
                Some(rows) => {
                    for (x, &row) in out.iter_mut().zip(rows) {
                        x[k] = vals[row];
                    }
                }
                None => {
                    for (x, &v) in out.iter_mut().zip(vals) {
                        x[k] = v;
                    }
                }
            },
            Feature::Sparse { vals, .. } => {
                for &(row, v) in vals {
                    let i = position.as_ref().map_or(row, |position| position[row]);
                    if i != usize::MAX {
                        out[i][k] = v;
                    }
                }
            }
        }
    }
    out
}

/// Adds `coef · ∇_θ s(z)` to `grad` for the standardized example `z`, where
/// `θ = (W, b, w, c)` is laid out as in [`TwoLayerNetworkClassifier`] and
/// `hidden` holds the pre-activations of `z` from
/// [`TwoLayerNetworkClassifier::score_of_standardized`].
pub(crate) fn accumulate_gradient(
    network: &TwoLayerNetworkClassifier,
    z: &[f64],
    hidden: &[f64],
    coef: f64,
    grad: &mut [f64],
) {
    let p = z.len();
    let h = hidden.len();
    let (g_hidden_weights, rest) = grad.split_at_mut(h * p);
    let (g_hidden_bias, rest) = rest.split_at_mut(h);
    let (g_output_weights, g_output_bias) = rest.split_at_mut(h);
    g_output_bias[0] += coef;
    for (j, &pre) in hidden.iter().enumerate() {
        g_output_weights[j] += coef * network.activation.apply(pre);
        let back = coef * network.output_weights[j] * network.activation.derivative(pre);
        g_hidden_bias[j] += back;
        for (g, v) in g_hidden_weights[j * p..(j + 1) * p].iter_mut().zip(z) {
            *g += back * v;
        }
    }
}

/// Copies the parameters into one vector `(W, b, w, c)`.
pub(crate) fn flatten(network: &TwoLayerNetworkClassifier) -> Vec<f64> {
    let mut theta = network.hidden_weights.clone();
    theta.extend_from_slice(&network.hidden_bias);
    theta.extend_from_slice(&network.output_weights);
    theta.push(network.output_bias);
    theta
}

/// Writes the vector `(W, b, w, c)` back into the network.
pub(crate) fn unflatten(network: &mut TwoLayerNetworkClassifier, theta: &[f64]) {
    let wp = network.hidden_weights.len();
    let h = network.hidden_bias.len();
    network.hidden_weights.copy_from_slice(&theta[..wp]);
    network.hidden_bias.copy_from_slice(&theta[wp..wp + h]);
    network
        .output_weights
        .copy_from_slice(&theta[wp + h..wp + 2 * h]);
    network.output_bias = theta[wp + 2 * h];
}
