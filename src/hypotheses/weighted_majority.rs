//! Weighted-vote combinations of hypotheses.
//!
//! [`WeightedMajority`] owns its weights and hypotheses and is the output of
//! most boosters; [`RefWeightedMajority`] borrows them.

use crate::core::{Classifier, Regressor, Sample, tools::helpers};
use serde::{Deserialize, Serialize};

/// A weighted combination of hypotheses that borrows its weights and
/// hypotheses.
///
/// As a [`Classifier`], its confidence on an example `x` is
/// `Σ_t w_t h_t(x)`; as a [`Regressor`], its prediction is the same weighted
/// sum of the hypotheses' predictions. Weights are used as given (no
/// normalization or filtering). If the two slices have different lengths,
/// the extra entries of the longer slice are ignored.
///
/// See [`WeightedMajority`] for the owned version.
///
/// # Examples
/// Evaluate a prefix of an ensemble without copying it, e.g. to trace test
/// accuracy as a function of the number of rounds.
/// ```
/// use miniboosts::{Classifier, Sample};
/// use miniboosts::hypotheses::RefWeightedMajority;
///
/// struct Constant(f64);
/// impl Classifier for Constant {
///     fn confidence(&self, _: &Sample, _: usize) -> f64 {
///         self.0
///     }
/// }
///
/// let sample = Sample::dummy(2);
/// let weights = [0.5, 0.3, 0.2];
/// let hypotheses = [Constant(1.0), Constant(-1.0), Constant(-1.0)];
///
/// // First two rounds: 0.5 - 0.3 = 0.2 (tolerance: one rounding error).
/// let prefix = RefWeightedMajority::new(&weights[..2], &hypotheses[..2]);
/// assert!((prefix.confidence(&sample, 0) - 0.2).abs() < 1e-15);
/// // All rounds: 0.5 - 0.3 - 0.2 = 0 up to rounding.
/// let full = RefWeightedMajority::new(&weights, &hypotheses);
/// assert!(full.confidence(&sample, 0).abs() < 1e-15);
/// ```
pub struct RefWeightedMajority<'a, H> {
    /// Weight of each hypothesis; `weights[t]` belongs to `hypotheses[t]`.
    pub weights: &'a [f64],
    /// The combined hypotheses.
    pub hypotheses: &'a [H],
}

impl<'a, H> RefWeightedMajority<'a, H> {
    /// Constructs a `RefWeightedMajority` from the given slices without
    /// copying, normalizing, or filtering the weights.
    #[inline]
    pub fn new(weights: &'a [f64], hypotheses: &'a [H]) -> Self {
        Self {
            weights,
            hypotheses,
        }
    }
}

impl<H> Classifier for RefWeightedMajority<'_, H>
where
    H: Classifier,
{
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        self.weights
            .iter()
            .zip(self.hypotheses)
            .map(|(w, h)| *w * h.confidence(sample, row))
            .sum::<f64>()
    }
}

impl<H> Regressor for RefWeightedMajority<'_, H>
where
    H: Regressor,
{
    fn predict(&self, sample: &Sample, row: usize) -> f64 {
        self.weights
            .iter()
            .zip(self.hypotheses)
            .map(|(w, h)| *w * h.predict(sample, row))
            .sum::<f64>()
    }
}

/// A weighted combination of hypotheses; the output of every booster in this
/// crate except [`GraphSeparationBoosting`](crate::boosters::graph_separation_boosting::GraphSeparationBoosting).
///
/// As a [`Classifier`], its confidence on an example `x` is
/// ```txt
/// f(x) = Σ_t w_t h_t(x),
/// ```
/// and the predicted label is `+1` if `f(x) ≥ 0` and `-1` otherwise. As a
/// [`Regressor`], its prediction is the same weighted sum of the hypotheses'
/// predictions. If `weights` and `hypotheses` have different lengths, the
/// extra entries of the longer vector are ignored.
///
/// Boosters that build the output with [`WeightedMajority::from_slices`]
/// return weights with `‖w‖_1 = 1`. Weights can be negative (for example, an
/// AdaBoost coefficient of a hypothesis with negative edge), so the weights
/// are not necessarily a convex combination.
/// [`CorrectiveErlpBoost`](crate::boosters::corrective_erlpboost::CorrectiveErlpBoost)
/// constructs it directly and keeps weights whose sum can be below one.
///
/// It implements [`serde::Serialize`] and [`serde::Deserialize`] when `H`
/// does, so a trained model can be saved and loaded.
///
/// # Examples
/// Every booster except `GraphSeparationBoosting` returns a
/// `WeightedMajority`; its fields can be inspected directly.
/// ```
/// use std::io::BufReader;
/// use miniboosts::prelude::*;
/// use miniboosts::Sample;
///
/// let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
/// let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
///     .unwrap()
///     .set_target("class");
/// let tree = DecisionTreeBuilder::new(&sample).max_depth(1).build();
/// let f = AdaBoost::init(&sample).run(&tree);
///
/// assert_eq!(f.weights.len(), f.hypotheses.len());
/// // `from_slices` normalizes the weights: Σ |w_t| = 1 up to rounding.
/// let l1 = f.weights.iter().map(|w| w.abs()).sum::<f64>();
/// assert!((l1 - 1.0).abs() < 1e-12);
/// assert_eq!(f.predict_all(&sample), vec![-1, -1, -1, 1, 1, 1]);
/// ```
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WeightedMajority<H> {
    /// Weight of each hypothesis; `weights[t]` belongs to `hypotheses[t]`.
    pub weights: Vec<f64>,
    /// The combined hypotheses.
    pub hypotheses: Vec<H>,
}

impl<H: Clone> WeightedMajority<H> {
    /// Constructs a `WeightedMajority` from `weights` and `hypotheses`,
    /// cloning the hypotheses.
    ///
    /// Pairs whose weight is exactly `0.0` are dropped (`-0.0` is also
    /// dropped; tiny nonzero weights are kept). The remaining weights are
    /// divided by their L1 norm, so `Σ_t |w_t| = 1`. Negative weights are
    /// kept with their sign: AdaBoost assigns a negative coefficient to a
    /// hypothesis with a negative edge, and the distribution update already
    /// used that coefficient. Pairs are matched by position; if the slices
    /// have different lengths, the extra entries are ignored.
    ///
    /// If no nonzero weight remains (for example, before the first boosting
    /// round), the result is the empty ensemble. Its confidence is `0.0` on
    /// every example, so [`Classifier::predict`] returns `+1`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{Classifier, Sample, WeightedMajority};
    ///
    /// #[derive(Clone, Debug, PartialEq)]
    /// struct Constant(f64);
    /// impl Classifier for Constant {
    ///     fn confidence(&self, _: &Sample, _: usize) -> f64 {
    ///         self.0
    ///     }
    /// }
    ///
    /// let hypotheses = [Constant(1.0), Constant(0.5), Constant(-1.0)];
    /// let f = WeightedMajority::from_slices(&[-1.0, 0.0, 3.0], &hypotheses);
    ///
    /// // The zero-weight pair is dropped; the negative weight is kept.
    /// // Dividing by the L1 norm 4 is exact here.
    /// assert_eq!(f.weights, vec![-0.25, 0.75]);
    /// assert_eq!(f.hypotheses, vec![Constant(1.0), Constant(-1.0)]);
    ///
    /// // f(x) = -0.25 * 1 + 0.75 * (-1) = -1.
    /// let sample = Sample::dummy(1);
    /// assert_eq!(f.confidence(&sample, 0), -1.0);
    /// assert_eq!(f.predict(&sample, 0), -1);
    /// ```
    #[inline]
    pub fn from_slices(weights: &[f64], hypotheses: &[H]) -> Self {
        let mut new_weights = Vec::with_capacity(weights.len());
        let mut new_hypotheses = Vec::with_capacity(hypotheses.len());

        weights.iter().copied().zip(hypotheses).for_each(|(w, h)| {
            if w != 0.0 {
                new_weights.push(w);
                new_hypotheses.push(h.clone());
            }
        });
        // An empty ensemble has nothing to normalize; its confidence is 0.
        if !new_weights.is_empty() {
            helpers::normalize(&mut new_weights[..]);
        }

        Self {
            weights: new_weights,
            hypotheses: new_hypotheses,
        }
    }
}

impl<H> WeightedMajority<H> {
    /// Appends `hypothesis` with weight `weight`. The weights are not
    /// renormalized; call [`WeightedMajority::normalize`] if needed.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{Classifier, Sample, WeightedMajority};
    ///
    /// struct Constant(f64);
    /// impl Classifier for Constant {
    ///     fn confidence(&self, _: &Sample, _: usize) -> f64 {
    ///         self.0
    ///     }
    /// }
    ///
    /// let mut f = WeightedMajority { weights: Vec::new(), hypotheses: Vec::new() };
    /// f.push(2.0, Constant(1.0));
    /// f.push(1.0, Constant(-1.0));
    /// assert_eq!(f.weights, vec![2.0, 1.0]);
    /// // Unnormalized: 2 * 1 + 1 * (-1) = 1.
    /// assert_eq!(f.confidence(&Sample::dummy(1), 0), 1.0);
    /// ```
    #[inline]
    pub fn push(&mut self, weight: f64, hypothesis: H) {
        self.weights.push(weight);
        self.hypotheses.push(hypothesis);
    }

    /// Divides the weights by their L1 norm, so that `Σ_t |w_t| = 1`.
    /// Signs are preserved and zero weights are kept.
    ///
    /// # Panics
    ///
    /// Panics if all weights are zero or there are no weights.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::WeightedMajority;
    ///
    /// let mut f = WeightedMajority { weights: vec![3.0, 0.0, -1.0], hypotheses: vec![(); 3] };
    /// f.normalize();
    /// // Divided by |3| + |0| + |-1| = 4 (exact); signs and zeros are kept.
    /// assert_eq!(f.weights, vec![0.75, 0.0, -0.25]);
    /// ```
    #[inline]
    pub fn normalize(&mut self) {
        helpers::normalize(&mut self.weights);
    }

    /// Consumes the combined hypothesis and returns `(weights, hypotheses)`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::WeightedMajority;
    ///
    /// let f = WeightedMajority { weights: vec![0.5, 0.5], hypotheses: vec!["h1", "h2"] };
    /// let (weights, hypotheses) = f.decompose();
    /// assert_eq!(weights, vec![0.5, 0.5]);
    /// assert_eq!(hypotheses, vec!["h1", "h2"]);
    /// ```
    #[inline]
    pub fn decompose(self) -> (Vec<f64>, Vec<H>) {
        (self.weights, self.hypotheses)
    }
}

impl<H> Classifier for WeightedMajority<H>
where
    H: Classifier,
{
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        self.weights
            .iter()
            .zip(&self.hypotheses[..])
            .map(|(w, h)| *w * h.confidence(sample, row))
            .sum::<f64>()
    }
}

impl<H> Regressor for WeightedMajority<H>
where
    H: Regressor,
{
    fn predict(&self, sample: &Sample, row: usize) -> f64 {
        self.weights
            .iter()
            .zip(&self.hypotheses[..])
            .map(|(w, h)| *w * h.predict(sample, row))
            .sum::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct Constant(f64);
    impl Classifier for Constant {
        fn confidence(&self, _: &Sample, _: usize) -> f64 {
            self.0
        }
    }

    #[test]
    fn from_slices_keeps_negative_weights_and_drops_zeros() {
        let hypotheses = [Constant(1.0), Constant(1.0), Constant(-1.0)];
        let f = WeightedMajority::from_slices(&[-1.0, 0.0, 3.0], &hypotheses);
        assert_eq!(f.weights, vec![-0.25, 0.75]);
        assert_eq!(f.hypotheses, vec![Constant(1.0), Constant(-1.0)]);
        // -0.25 * 1 + 0.75 * (-1) = -1.
        let sample = Sample::dummy(1);
        assert_eq!(f.confidence(&sample, 0), -1.0);
    }

    #[test]
    fn from_slices_without_nonzero_weights_is_the_empty_ensemble() {
        let sample = Sample::dummy(2);
        for (weights, hypotheses) in [
            (&[][..], &[][..]),
            (&[0.0, -0.0][..], &[Constant(1.0), Constant(-1.0)][..]),
        ] {
            let f = WeightedMajority::from_slices(weights, hypotheses);
            assert!(f.weights.is_empty() && f.hypotheses.is_empty());
            assert_eq!(f.confidence(&sample, 0), 0.0);
            assert_eq!(f.predict(&sample, 0), 1);
        }
    }
}
