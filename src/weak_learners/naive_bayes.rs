//! Weighted Gaussian naive Bayes as a weak learner.
//!
//! [`NaiveBayes`] fits, under a booster's distribution `d`, a class prior
//! and an independent Gaussian per feature and class, and returns a
//! [`NaiveBayesClassifier`] whose confidence is the difference of the two
//! posterior class probabilities.
//!
//! # Examples
//! ```
//! use std::io::BufReader;
//! use miniboosts::{Sample, prelude::*};
//! use miniboosts::weak_learners::naive_bayes::NaiveBayes;
//!
//! // Labels follow the sign of `x + y`.
//! let csv = b"x,y,class\n\
//!     -2,1,-1\n-1,-2,-1\n1,-3,-1\n-3,2,-1\n0,-1,-1\n-2,-1,-1\n\
//!     2,1,1\n1,2,1\n-1,3,1\n3,-1,1\n0,2,1\n2,2,1\n";
//! let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
//!     .unwrap()
//!     .set_target("class");
//!
//! let f = AdaBoost::init(&sample).force_quit_at(20).run(&NaiveBayes::new());
//! let correct = f
//!     .predict_all(&sample)
//!     .iter()
//!     .zip(sample.target())
//!     .filter(|&(&p, &y)| p as f64 == y)
//!     .count();
//! println!("{correct} of 12 training examples correct");
//! assert!(f.confidence_all(&sample).iter().all(|c| (-1.0..=1.0).contains(c)));
//! ```
use crate::core::{Classifier, Sample, WeakLearner};
use serde::{Deserialize, Serialize};

/// Weighted Gaussian naive Bayes.
///
/// # Algorithm
/// Given a distribution `d` (normalized here to sum to one), for each label
/// `y ∈ {-1, +1}` and feature `k`:
/// ```txt
/// π_y     = Σ_{i: y_i = y} d_i,
/// μ_{y,k} = Σ_{i: y_i = y} d_i x_{ik} / π_y,
/// σ²_{y,k} = Σ_{i: y_i = y} d_i (x_{ik} - μ_{y,k})² / π_y + s,
/// ```
/// where `s = ε · max_k Var_d(x_k)` (the largest weighted variance of a
/// feature over all examples) smooths zero variances, as in
/// scikit-learn's `GaussianNB` with `var_smoothing = ε`. If every feature
/// has zero weighted variance, `s = ε`. The hypothesis is
/// ```txt
/// L(x) = ln(π_{+1} / π_{-1}) + Σ_k [ln N(x_k; μ_{+1,k}, σ²_{+1,k}) - ln N(x_k; μ_{-1,k}, σ²_{-1,k})],
/// h(x) = tanh(L(x) / 2) = P(+1 | x) - P(-1 | x) ∈ [-1, 1].
/// ```
/// If one label has zero weight, `h` is the constant `±1` of the other.
///
/// # Guarantee
/// None on the edge: naive Bayes fits a generative model and need not
/// maximize the edge over any class.
///
/// # Limitations
/// - The Gaussian model and the independence of features given the label
///   are assumptions about the data, not checked.
/// - Far from the training data, the quadratic terms dominate `L`, so the
///   confidence tends to `±1` even where no training example lies.
/// - Each call takes `O(m d)` time for `m` examples and `d` features.
///
/// # Parameters
/// | Builder method | Default | Meaning |
/// | --- | --- | --- |
/// | [`var_smoothing`](NaiveBayes::var_smoothing) | `1e-9` | `ε`, as scikit-learn's default. |
///
/// # Examples
/// See the [module documentation](self).
#[derive(Debug, Clone, Copy)]
pub struct NaiveBayes {
    var_smoothing: f64,
}

impl Default for NaiveBayes {
    fn default() -> Self {
        Self::new()
    }
}

impl NaiveBayes {
    /// Constructs the learner with `var_smoothing = 1e-9`.
    ///
    /// # Examples
    /// See the [module documentation](self).
    pub fn new() -> Self {
        Self {
            var_smoothing: 1e-9,
        }
    }

    /// Sets `ε`, the fraction of the largest feature variance added to
    /// every class variance.
    ///
    /// # Panics
    /// Panics if `var_smoothing` is not finite and positive.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{Classifier, Sample, WeakLearner};
    /// use miniboosts::weak_learners::naive_bayes::NaiveBayes;
    ///
    /// let rows = vec![vec![0.0], vec![0.0], vec![1.0], vec![1.0]];
    /// let sample = Sample::from_dense(&rows, Some(&[-1.0, -1.0, 1.0, 1.0])).unwrap();
    /// // Each class is a single point (zero variance); the smoothing keeps
    /// // the Gaussians proper. Larger smoothing gives softer confidences.
    /// let sharp = NaiveBayes::new().produce(&sample, &[0.25; 4]);
    /// let soft = NaiveBayes::new().var_smoothing(1.0).produce(&sample, &[0.25; 4]);
    /// assert!(soft.confidence(&sample, 2) < sharp.confidence(&sample, 2));
    /// assert!(soft.confidence(&sample, 2) > 0.0);
    /// ```
    pub fn var_smoothing(mut self, var_smoothing: f64) -> Self {
        assert!(
            var_smoothing.is_finite() && var_smoothing > 0.0,
            "var_smoothing must be finite and positive"
        );
        self.var_smoothing = var_smoothing;
        self
    }
}

impl WeakLearner for NaiveBayes {
    type Hypothesis = NaiveBayesClassifier;

    fn name(&self) -> &str {
        "Gaussian naive Bayes"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        Some(vec![("Variance smoothing", self.var_smoothing.to_string())])
    }

    /// Fits the weighted model; see [`NaiveBayes`].
    ///
    /// # Panics
    /// Panics if the sample has no labels, a label is not `±1`, or `dist`
    /// has the wrong length, a negative or non-finite entry, or zero sum.
    fn produce(&self, sample: &Sample, dist: &[f64]) -> Self::Hypothesis {
        let n_examples = sample.shape().0;
        let target = sample.target();
        assert_eq!(
            target.len(),
            n_examples,
            "the sample needs labels; use `Sample::set_target`"
        );
        assert!(
            target.iter().all(|y| *y == 1.0 || *y == -1.0),
            "naive Bayes needs labels -1 and +1"
        );
        assert_eq!(
            dist.len(),
            n_examples,
            "the distribution needs one entry per example"
        );
        assert!(
            dist.iter().all(|d| d.is_finite() && *d >= 0.0),
            "distribution entries must be finite and nonnegative"
        );
        let total: f64 = dist.iter().sum();
        assert!(total > 0.0, "the distribution must have positive mass");
        let dist: Vec<f64> = dist.iter().map(|d| d / total).collect();

        let feature_names: Vec<String> = sample
            .features()
            .iter()
            .map(|f| f.name().to_string())
            .collect();
        let prior = |y: f64| -> f64 {
            target
                .iter()
                .zip(&dist)
                .filter(|&(&t, _)| t == y)
                .map(|(_, d)| d)
                .sum()
        };
        let (prior_pos, prior_neg) = (prior(1.0), prior(-1.0));
        if prior_pos == 0.0 || prior_neg == 0.0 {
            let log_prior_ratio = if prior_neg == 0.0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
            return NaiveBayesClassifier {
                feature_names: Vec::new(),
                log_prior_ratio,
                positive: Vec::new(),
                negative: Vec::new(),
            };
        }

        let largest_variance = sample
            .weighted_mean_and_variance(&dist)
            .iter()
            .map(|(_, v)| *v)
            .fold(0.0, f64::max);
        let smoothing = if largest_variance > 0.0 {
            self.var_smoothing * largest_variance
        } else {
            self.var_smoothing
        };
        let gaussians = |y: f64| -> Vec<(f64, f64)> {
            sample
                .weighted_mean_and_variance_for_label(y, &dist)
                .into_iter()
                .map(|(mean, var)| (mean, var + smoothing))
                .collect()
        };
        NaiveBayesClassifier {
            feature_names,
            log_prior_ratio: prior_pos.ln() - prior_neg.ln(),
            positive: gaussians(1.0),
            negative: gaussians(-1.0),
        }
    }
}

/// The hypothesis of [`NaiveBayes`]: `h(x) = tanh(L(x) / 2)`, the
/// difference of the posterior probabilities of `+1` and `-1`.
///
/// Features are read by name, so a test sample with the same columns in
/// another order is evaluated correctly.
///
/// # Examples
/// See the [module documentation](self).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NaiveBayesClassifier {
    feature_names: Vec<String>,
    // ln(π_{+1} / π_{-1}); ±∞ when one label has zero weight.
    log_prior_ratio: f64,
    // (mean, variance) per feature for label +1 and -1.
    positive: Vec<(f64, f64)>,
    negative: Vec<(f64, f64)>,
}

impl NaiveBayesClassifier {
    /// Returns the log posterior odds
    /// `L(x) = ln(P(+1 | x) / P(-1 | x))` for the example in `row`;
    /// `±∞` if one label had zero weight.
    ///
    /// # Panics
    /// Panics if `sample` lacks one of the features.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{Classifier, Sample, WeakLearner};
    /// use miniboosts::weak_learners::naive_bayes::NaiveBayes;
    ///
    /// let rows = vec![vec![-1.0], vec![1.0]];
    /// let sample = Sample::from_dense(&rows, Some(&[-1.0, 1.0])).unwrap();
    /// let h = NaiveBayes::new().produce(&sample, &[0.5, 0.5]);
    /// // Symmetric classes: x = 0 is equally likely under both.
    /// let midpoint = Sample::from_dense(&[vec![0.0]], None).unwrap();
    /// assert_eq!(h.log_odds(&midpoint, 0), 0.0);
    /// ```
    pub fn log_odds(&self, sample: &Sample, row: usize) -> f64 {
        if !self.log_prior_ratio.is_finite() {
            return self.log_prior_ratio;
        }
        let log_density = |x: f64, (mean, var): (f64, f64)| {
            -0.5 * (std::f64::consts::TAU * var).ln() - (x - mean).powi(2) / (2.0 * var)
        };
        self.feature_names
            .iter()
            .zip(self.positive.iter().zip(&self.negative))
            .map(|(name, (&pos, &neg))| {
                let x = sample[name][row];
                log_density(x, pos) - log_density(x, neg)
            })
            .sum::<f64>()
            + self.log_prior_ratio
    }
}

impl Classifier for NaiveBayesClassifier {
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        (0.5 * self.log_odds(sample, row)).tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_hand_computed_posterior() {
        // Label +1 at x = 1, 3 (weights 0.2, 0.2); label -1 at x = -1
        // (weight 0.6). μ+ = 2, σ²+ = 1; μ- = -1, σ²- = 0 + s, with
        // s = ε Var_d(x), Var_d(x) = 0.2·1 + 0.2·9 + 0.6·1 - (0.2 + 0.6 - 0.6)² = 2.56.
        let rows = vec![vec![1.0], vec![3.0], vec![-1.0]];
        let sample = Sample::from_dense(&rows, Some(&[1.0, 1.0, -1.0])).unwrap();
        let eps = 0.5;
        let h = NaiveBayes::new()
            .var_smoothing(eps)
            .produce(&sample, &[0.2, 0.2, 0.6]);
        let s = eps * 2.56;
        let (var_pos, var_neg) = (1.0 + s, s);
        let x = 0.5_f64;
        let ln_n = |x: f64, m: f64, v: f64| {
            -0.5 * (std::f64::consts::TAU * v).ln() - (x - m).powi(2) / (2.0 * v)
        };
        let expected = (0.4_f64 / 0.6).ln() + ln_n(x, 2.0, var_pos) - ln_n(x, -1.0, var_neg);
        let point = Sample::from_dense(&[vec![x]], None).unwrap();
        // Weighted sums of a few terms: agreement to 1e-12.
        assert!((h.log_odds(&point, 0) - expected).abs() < 1e-12);
        assert!((h.confidence(&point, 0) - (expected / 2.0).tanh()).abs() < 1e-12);
    }

    #[test]
    fn scaling_the_distribution_changes_nothing() {
        let rows = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.5],
            vec![2.0, -1.0],
            vec![3.0, 0.0],
        ];
        let sample = Sample::from_dense(&rows, Some(&[-1.0, -1.0, 1.0, 1.0])).unwrap();
        let d = [0.1, 0.2, 0.3, 0.4];
        let scaled: Vec<f64> = d.iter().map(|v| v * 8.0).collect();
        let a = NaiveBayes::new().produce(&sample, &d);
        let b = NaiveBayes::new().produce(&sample, &scaled);
        for i in 0..4 {
            assert!((a.log_odds(&sample, i) - b.log_odds(&sample, i)).abs() < 1e-9);
        }
    }

    #[test]
    fn a_label_without_weight_gives_a_constant() {
        let rows = vec![vec![0.0], vec![1.0]];
        let sample = Sample::from_dense(&rows, Some(&[-1.0, 1.0])).unwrap();
        let h = NaiveBayes::new().produce(&sample, &[0.0, 1.0]);
        assert_eq!(h.confidence_all(&sample), vec![1.0, 1.0]);
        let h = NaiveBayes::new().produce(&sample, &[1.0, 0.0]);
        assert_eq!(h.confidence_all(&sample), vec![-1.0, -1.0]);
    }

    #[test]
    fn invalid_inputs_panic() {
        let rows = vec![vec![0.0], vec![1.0]];
        let sample = Sample::from_dense(&rows, Some(&[-1.0, 1.0])).unwrap();
        let learner = NaiveBayes::new();
        for d in [
            vec![0.0, 0.0],
            vec![-1.0, 2.0],
            vec![f64::NAN, 1.0],
            vec![1.0],
        ] {
            assert!(std::panic::catch_unwind(|| learner.produce(&sample, &d)).is_err());
        }
        assert!(std::panic::catch_unwind(|| NaiveBayes::new().var_smoothing(0.0)).is_err());
    }
}
