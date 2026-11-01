//! A finite class of Gaussian radial basis functions centered at training
//! examples, used through [`FiniteClass`].
//!
//! The class consists of `±k_c` for the chosen centers `c`, where
//! ```txt
//! k_c(x) = exp(-‖x - x_c‖² / (2σ²)) ∈ (0, 1],
//! ```
//! `x_c` is the feature vector of training example `c`, and `σ` is the
//! bandwidth. A combination `Σ_c w_c k_c(x)` is a kernel expansion of the
//! Gaussian (RBF) kernel without a bias term, so boosting over this class,
//! for example with LPBoost, learns a sparse kernel expansion one center at a
//! time. Because the class is finite, [`FiniteClass`] returns the center
//! with the largest edge exactly.
//!
//! # Examples
//! Boost RBF bumps on data no single stump separates.
//! ```
//! use std::io::BufReader;
//! use miniboosts::{Sample, prelude::*};
//! use miniboosts::weak_learners::rbf::RbfClassBuilder;
//!
//! // Labels follow the sign of `x + y`.
//! let csv = b"x,y,class\n\
//!     -2,1,-1\n-1,-2,-1\n1,-3,-1\n-3,2,-1\n0,-1,-1\n-2,-1,-1\n\
//!     2,1,1\n1,2,1\n-1,3,1\n3,-1,1\n0,2,1\n2,2,1\n";
//! let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
//!     .unwrap()
//!     .set_target("class");
//!
//! // One bump per training example (24 members with negations).
//! let learner = RbfClassBuilder::new(&sample).bandwidth(1.5).build();
//! let f = LpBoost::init(&sample).nu(1.0).tolerance(0.01).run(&learner);
//! assert_eq!(f.predict_all(&sample), sample.target().iter().map(|&y| y as i64).collect::<Vec<_>>());
//! ```
use crate::core::{Classifier, Sample};
use crate::weak_learners::finite_class::FiniteClass;
use serde::{Deserialize, Serialize};

/// The Gaussian bump `k_c(x) = exp(-‖x - x_c‖² / (2σ²))` around a center
/// `x_c`. See the [module documentation](self).
///
/// The center is stored with the feature names of the training sample, and
/// [`Classifier::confidence`] reads features by name, so a test sample with
/// the same columns in a different order is evaluated correctly.
///
/// # Examples
/// ```
/// use std::io::BufReader;
/// use miniboosts::{Classifier, Sample};
/// use miniboosts::weak_learners::rbf::RbfHypothesis;
///
/// let csv = b"x,y\n0,0\n1,1\n";
/// let sample = Sample::from_reader(BufReader::new(&csv[..]), true).unwrap();
/// let h = RbfHypothesis::new(vec!["x".into(), "y".into()], vec![0.0, 0.0], 1.0);
/// // exp(0) = 1 at the center; exp(-2 / 2) = e^-1 at distance √2.
/// assert_eq!(h.confidence(&sample, 0), 1.0);
/// assert!((h.confidence(&sample, 1) - (-1.0f64).exp()).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RbfHypothesis {
    feature_names: Vec<String>,
    center: Vec<f64>,
    bandwidth: f64,
}

impl RbfHypothesis {
    /// Constructs the bump with center `center` (one value per name in
    /// `feature_names`) and bandwidth `σ = bandwidth`.
    ///
    /// # Panics
    /// Panics if the lengths differ, a center value is not finite, or
    /// `bandwidth` is not finite and positive.
    ///
    /// # Examples
    /// See [`RbfHypothesis`].
    pub fn new(feature_names: Vec<String>, center: Vec<f64>, bandwidth: f64) -> Self {
        assert_eq!(
            feature_names.len(),
            center.len(),
            "one center value per feature"
        );
        assert!(
            center.iter().all(|v| v.is_finite()),
            "center values must be finite"
        );
        assert!(
            bandwidth.is_finite() && bandwidth > 0.0,
            "RBF bandwidth must be finite and positive"
        );
        Self {
            feature_names,
            center,
            bandwidth,
        }
    }

    /// Returns the center `x_c`, in the order of
    /// [`RbfHypothesis::feature_names`].
    pub fn center(&self) -> &[f64] {
        &self.center
    }

    /// Returns the names of the features the center refers to.
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Returns the bandwidth `σ`.
    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }
}

impl Classifier for RbfHypothesis {
    /// Returns `exp(-‖x - x_c‖² / (2σ²))` for the example in `row`.
    ///
    /// # Panics
    /// Panics if `sample` lacks one of the center's features.
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        let squared: f64 = self
            .feature_names
            .iter()
            .zip(&self.center)
            .map(|(name, c)| (sample[name][row] - c).powi(2))
            .sum();
        (-squared / (2.0 * self.bandwidth * self.bandwidth)).exp()
    }
}

/// Builds a [`FiniteClass`] of Gaussian bumps `±k_c`, one pair per center.
/// See the [module documentation](self).
///
/// # Parameters
/// | Builder method | Default | Meaning |
/// | --- | --- | --- |
/// | [`bandwidth`](RbfClassBuilder::bandwidth) | required | Bandwidth `σ` of every bump. |
/// | [`centers`](RbfClassBuilder::centers) | every training example | Rows of the training sample used as centers. |
///
/// # Limitations
/// - The bandwidth has no default; no choice is right for every scale of
///   the features, so standardize the features or tune `σ` on held-out
///   data.
/// - With `k` centers and `m` examples, building evaluates `m k` bumps
///   (`O(m k d)` time for `d` features), and the learner caches `m k`
///   margins (`8 m k` bytes): `k = m = 10⁴` needs 800 MB. Use
///   [`RbfClassBuilder::centers`] to choose fewer centers.
/// - The class has no bias term: every bump tends to `0` far from all
///   centers, where `Σ_c w_c k_c(x)` is close to `0`.
///
/// # Examples
/// See the [module documentation](self).
pub struct RbfClassBuilder<'a> {
    sample: &'a Sample,
    bandwidth: Option<f64>,
    centers: Option<Vec<usize>>,
}

impl<'a> RbfClassBuilder<'a> {
    /// Starts a builder for the training `sample`, with every example as a
    /// center. Set the bandwidth before [`RbfClassBuilder::build`].
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// See the [module documentation](self).
    pub fn new(sample: &'a Sample) -> Self {
        Self {
            sample,
            bandwidth: None,
            centers: None,
        }
    }

    /// Sets the bandwidth `σ` of every bump.
    ///
    /// # Panics
    /// Panics if `bandwidth` is not finite and positive.
    ///
    /// # Examples
    /// See the [module documentation](self).
    pub fn bandwidth(mut self, bandwidth: f64) -> Self {
        assert!(
            bandwidth.is_finite() && bandwidth > 0.0,
            "RBF bandwidth must be finite and positive"
        );
        self.bandwidth = Some(bandwidth);
        self
    }

    /// Uses the training examples in `rows` as centers, in the given order
    /// (which sets the hypotheses' indices and the tie-breaking order).
    ///
    /// # Panics
    /// Panics if `rows` is empty or contains an index outside the sample.
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::Sample;
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// use miniboosts::weak_learners::rbf::RbfClassBuilder;
    ///
    /// let learner = RbfClassBuilder::new(&sample)
    ///     .bandwidth(1.0)
    ///     .centers(vec![0, 3])
    ///     .build();
    /// assert_eq!(learner.hypotheses().len(), 2);
    /// assert_eq!(learner.hypotheses()[1].center(), &[1.0, 0.0]);
    /// ```
    pub fn centers(mut self, rows: Vec<usize>) -> Self {
        let n_examples = self.sample.shape().0;
        assert!(!rows.is_empty(), "the RBF class needs at least one center");
        assert!(
            rows.iter().all(|&r| r < n_examples),
            "center rows must index the training sample"
        );
        self.centers = Some(rows);
        self
    }

    /// Builds the learner: a [`FiniteClass`] over the bumps, with negations.
    ///
    /// Time complexity: `O(m k d)`.
    ///
    /// # Panics
    /// Panics if the bandwidth is not set, the sample has no labels, or a
    /// feature value is not finite.
    ///
    /// # Examples
    /// See the [module documentation](self).
    pub fn build(self) -> FiniteClass<RbfHypothesis> {
        let bandwidth = self
            .bandwidth
            .expect("set the RBF bandwidth with `RbfClassBuilder::bandwidth`");
        let names: Vec<String> = self
            .sample
            .features()
            .iter()
            .map(|f| f.name().to_string())
            .collect();
        let rows = self
            .centers
            .unwrap_or_else(|| (0..self.sample.shape().0).collect());
        let hypotheses = rows
            .iter()
            .map(|&row| {
                let center = self.sample.features().iter().map(|f| f[row]).collect();
                RbfHypothesis::new(names.clone(), center, bandwidth)
            })
            .collect();
        FiniteClass::new(self.sample, hypotheses).with_negations()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WeakLearner;
    use std::io::BufReader;

    fn line_sample() -> Sample {
        let csv = b"x,class\n-3,-1\n-2,-1\n2,1\n3,1\n";
        Sample::from_reader(BufReader::new(&csv[..]), true)
            .unwrap()
            .set_target("class")
    }

    #[test]
    fn bumps_follow_the_gaussian_formula() {
        let sample = line_sample();
        let learner = RbfClassBuilder::new(&sample).bandwidth(2.0).build();
        let h = &learner.hypotheses()[1]; // centered at x = -2
        for (row, x) in [-3.0, -2.0, 2.0, 3.0].into_iter().enumerate() {
            let expected = (-(x + 2.0_f64).powi(2) / 8.0).exp();
            assert!((h.confidence(&sample, row) - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn the_best_center_and_sign_are_returned() {
        let sample = line_sample();
        let learner = RbfClassBuilder::new(&sample).bandwidth(1.0).build();
        // All mass on example 0 (label -1): the bump at x = -3 has
        // confidence 1 there, so -k_0 has edge 1.
        let h = learner.produce(&sample, &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!((h.index, h.sign), (0, -1.0));
        // All mass on example 3 (label +1): +k_3.
        let h = learner.produce(&sample, &[0.0, 0.0, 0.0, 1.0]);
        assert_eq!((h.index, h.sign), (3, 1.0));
    }

    #[test]
    fn features_are_read_by_name() {
        let sample = line_sample();
        let h = RbfHypothesis::new(vec!["x".into()], vec![2.0], 1.0);
        let csv = b"class,x\n1,2\n";
        let reordered = Sample::from_reader(BufReader::new(&csv[..]), true).unwrap();
        assert_eq!(h.confidence(&reordered, 0), 1.0);
        assert_eq!(h.confidence(&sample, 2), 1.0);
    }

    #[test]
    fn missing_bandwidth_and_bad_centers_panic() {
        let sample = line_sample();
        assert!(std::panic::catch_unwind(|| RbfClassBuilder::new(&sample).build()).is_err());
        assert!(
            std::panic::catch_unwind(|| RbfClassBuilder::new(&sample).centers(vec![4])).is_err()
        );
        assert!(std::panic::catch_unwind(|| RbfClassBuilder::new(&sample).bandwidth(0.0)).is_err());
    }
}
