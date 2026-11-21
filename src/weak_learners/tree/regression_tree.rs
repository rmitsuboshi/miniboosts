//! A weighted least-squares regression tree (CART-style) as a weak learner
//! for boosters that fit real-valued working responses, such as
//! [`GradientBoosting`](crate::GradientBoosting) and
//! [`LogitBoost`](crate::LogitBoost).
//!
//! # Examples
//! Fit a step function under uniform weights.
//! ```
//! use miniboosts::{Regressor, Sample, WeakLearner};
//! use miniboosts::weak_learners::tree::regression_tree::RegressionTreeBuilder;
//!
//! let rows: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64]).collect();
//! let y = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0];
//! let sample = Sample::from_dense(&rows, Some(&y)).unwrap();
//!
//! let tree = RegressionTreeBuilder::new(&sample).max_depth(1).build();
//! let h = tree.produce(&sample, &[1.0 / 6.0; 6]);
//! assert_eq!(h.predict_all(&sample), y.to_vec());
//! ```
use crate::core::binning::{BinIndex, Bins};
use crate::core::tree::{LeftRight, Splitter};
use crate::core::{Feature, Regressor, Sample, WeakLearner};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The default number of bins per feature (as for the decision tree).
pub const DEFAULT_NBIN: usize = 255;

/// The default depth budget.
pub const DEFAULT_MAX_DEPTH: usize = 3;

/// Builds a [`RegressionTree`] for a training sample.
///
/// # Parameters
/// | Builder method | Default | Meaning |
/// | --- | --- | --- |
/// | [`max_depth`](RegressionTreeBuilder::max_depth) | [`DEFAULT_MAX_DEPTH`] (`3`) | Levels of splits. |
/// | [`n_bins`](RegressionTreeBuilder::n_bins) | `255` | Cap `k` on the bins per feature: one bin per distinct value if there are at most `k`, otherwise `k` bins holding about the same number of examples each. |
///
/// # Examples
/// See the [module documentation](self).
#[derive(Debug)]
pub struct RegressionTreeBuilder<'a> {
    sample: &'a Sample,
    max_depth: usize,
    n_bins: Option<usize>,
}

impl<'a> RegressionTreeBuilder<'a> {
    /// Starts a builder for `sample` with the defaults.
    ///
    /// # Examples
    /// See the [module documentation](self).
    pub fn new(sample: &'a Sample) -> Self {
        Self {
            sample,
            max_depth: DEFAULT_MAX_DEPTH,
            n_bins: None,
        }
    }

    /// Sets the depth budget: at most `depth` levels of splits and
    /// `2^depth` leaves; `0` gives a constant.
    ///
    /// # Examples
    /// See the [module documentation](self).
    #[must_use]
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Caps the bins per feature at `n_bins`: a feature with at most
    /// `n_bins` distinct values gets one bin per value (every split between
    /// consecutive values is a candidate); otherwise it gets `n_bins` bins
    /// holding about the same number of examples each (see
    /// `Bins::quantile`).
    ///
    /// # Panics
    /// Panics if `n_bins == 0`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{Regressor, Sample, WeakLearner};
    /// use miniboosts::weak_learners::tree::regression_tree::RegressionTreeBuilder;
    ///
    /// let rows: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64]).collect();
    /// let sample = Sample::from_dense(&rows, Some(&[0.0, 0.0, 1.0, 1.0])).unwrap();
    /// // A single bin admits no split: the tree is the weighted mean.
    /// let tree = RegressionTreeBuilder::new(&sample).n_bins(1).build();
    /// assert_eq!(tree.produce(&sample, &[0.25; 4]).predict(&sample, 0), 0.5);
    /// ```
    #[must_use]
    pub fn n_bins(mut self, n_bins: usize) -> Self {
        assert!(n_bins > 0, "a feature needs at least one bin");
        self.n_bins = Some(n_bins);
        self
    }

    /// Cuts every feature into bins and returns the learner.
    ///
    /// # Panics
    /// Panics if a feature is empty or has a non-finite value.
    ///
    /// # Examples
    /// See the [module documentation](self).
    pub fn build(self) -> RegressionTree {
        let features = self.sample.features();
        let bins: Vec<(String, Bins)> = features
            .iter()
            .map(|f| {
                // One bin per value (exact CART splits) if the values fit
                // under the cap, and bins of equal shares of the examples
                // otherwise.
                let bins = Bins::quantile(f, self.n_bins.unwrap_or(DEFAULT_NBIN));
                (f.name().to_string(), bins)
            })
            .collect();
        let bin_index = bins
            .iter()
            .zip(features)
            .map(|((_, bins), feature)| bins.index(feature))
            .collect();
        RegressionTree {
            bins,
            features: Arc::clone(self.sample.shared_features()),
            bin_index,
            max_depth: self.max_depth,
        }
    }
}

/// A weighted least-squares regression tree learner.
///
/// # Algorithm
/// Given a sample whose targets `z_i` are real values and a distribution
/// `d`, the tree is grown greedily on the examples with `d_i > 0`. Each node
/// `S` predicts the weighted mean `ẑ_S = Σ_{i ∈ S} d_i z_i / Σ_{i ∈ S} d_i`
/// and has weighted squared error `SSE_S = Σ_{i ∈ S} d_i (z_i - ẑ_S)²`. A
/// node is split by the rule `x[f] < t` that most decreases
/// `SSE_S - SSE_L - SSE_R`, over the boundaries between the bins (cut once by
/// the builder) that contain the node's examples. A feature with at most
/// [`n_bins`](RegressionTreeBuilder::n_bins) distinct values has one bin per
/// value, with boundaries at the midpoints, so every split of exact CART is
/// a candidate; a feature with more values has `n_bins` bins holding about
/// the same number of examples each. A node
/// becomes a leaf when its depth budget is `0`, its `SSE` is `0`, or no
/// split decreases it. Features and thresholds are scanned in order and
/// ties keep the first, so the result is deterministic.
///
/// # Cost
/// The builder stores the bin of every value of its sample: 1 byte per
/// value of a dense feature and per stored entry of a sparse one (2 or 4
/// bytes for a feature with more than 256 or 65536 bins). These are
/// used when [`WeakLearner::produce`] gets that sample, a clone of it, or a
/// copy made by [`Sample::with_target`], which is what
/// [`GradientBoosting`](crate::GradientBoosting) and
/// [`LogitBoost`](crate::LogitBoost) pass; a node then costs one addition per
/// example and feature. For any other sample, the bin of each value is
/// searched again (`O(log n_bins)` per value). The trees are the same in
/// both cases. Features are searched in parallel; the result does not depend
/// on the number of threads.
///
/// # Limitations
/// - A feature with more distinct values than
///   [`n_bins`](RegressionTreeBuilder::n_bins) is cut into bins of equal
///   shares of the examples, and only their boundaries are candidate
///   thresholds, so the best split may be missed.
/// - The returned [`RegressionTreeRegressor`] is a [`Regressor`], not a
///   classifier; its predictions are unbounded.
/// - `SSE` is computed from weighted sums of `z` and `z²`, which can lose
///   precision when the targets are large and nearly constant.
///
/// # Examples
/// See the [module documentation](self).
#[derive(Debug)]
pub struct RegressionTree {
    // Bins per feature, in column order.
    bins: Vec<(String, Bins)>,
    // The features of the builder's sample and, for each, the bin of every
    // value: used when `produce` gets a sample with these very features.
    features: Arc<Vec<Feature>>,
    bin_index: Vec<BinIndex>,
    max_depth: usize,
}

/// Weighted sums for a set of examples.
#[derive(Clone, Copy, Default)]
struct Moments {
    w: f64,
    wz: f64,
    wzz: f64,
}

impl Moments {
    fn add(&mut self, w: f64, z: f64) {
        self.w += w;
        self.wz += w * z;
        self.wzz += w * z * z;
    }

    fn minus(self, other: Self) -> Self {
        Self {
            w: self.w - other.w,
            wz: self.wz - other.wz,
            wzz: self.wzz - other.wzz,
        }
    }

    fn mean(&self) -> f64 {
        self.wz / self.w
    }

    /// `Σ w (z - mean)² = Σ w z² - (Σ w z)² / Σ w`, clamped at `0`.
    fn sse(&self) -> f64 {
        if self.w <= 0.0 {
            return 0.0;
        }
        (self.wzz - self.wz * self.wz / self.w).max(0.0)
    }
}

impl RegressionTree {
    fn grow(
        &self,
        sample: &Sample,
        dist: &[f64],
        indices: Vec<usize>,
        depth: usize,
        cached: bool,
    ) -> RegNode {
        let target = sample.target();
        let mut total = Moments::default();
        for &i in &indices {
            total.add(dist[i], target[i]);
        }
        let value = total.mean();
        let sse = total.sse();
        if depth == 0 || sse == 0.0 {
            return RegNode::Leaf { value };
        }

        // The best split of each feature over its bin boundaries. The
        // features are independent, so they are searched in parallel; each
        // sum runs over `indices` in order, whatever the number of threads.
        let per_feature: Vec<Option<(f64, f64)>> = self
            .bins
            .par_iter()
            .enumerate()
            .map(|(k, (name, bins))| {
                let n_bins = bins.as_slice().len();
                let mut per_bin = vec![Moments::default(); n_bins];
                match cached.then(|| &self.bin_index[k]) {
                    Some(BinIndex::Dense(bin_of_row)) => {
                        bin_of_row.for_each(&indices, |i, bin| {
                            per_bin[bin].add(dist[i], target[i]);
                        });
                    }
                    Some(BinIndex::Sparse { bin_of_entry, zero }) => {
                        let mut entries = self.features[k].cursor();
                        for &i in &indices {
                            let bin = entries.entry(i).map_or(*zero, |p| bin_of_entry.get(p));
                            per_bin[bin].add(dist[i], target[i]);
                        }
                    }
                    None => {
                        let mut feature = sample[name].cursor();
                        for &i in &indices {
                            per_bin[bins.index_of(feature.value(i))].add(dist[i], target[i]);
                        }
                    }
                }
                let nonempty: Vec<usize> = (0..n_bins).filter(|&b| per_bin[b].w > 0.0).collect();
                let mut best: Option<(f64, f64)> = None;
                let mut left = Moments::default();
                for pair in nonempty.windows(2) {
                    let (b, next) = (pair[0], pair[1]);
                    // Bins strictly between `b` and `next` are empty.
                    for m in &per_bin[b..next] {
                        left.w += m.w;
                        left.wz += m.wz;
                        left.wzz += m.wzz;
                    }
                    let right = total.minus(left);
                    let gain = sse - left.sse() - right.sse();
                    let threshold =
                        f64::midpoint(bins.as_slice()[b].end(), bins.as_slice()[next].start());
                    if gain > 0.0 && best.is_none_or(|(g, _)| gain > g) {
                        best = Some((gain, threshold));
                    }
                }
                best
            })
            .collect();
        // Ties go to the first feature in order, then to its first boundary.
        let mut best: Option<(f64, &str, f64)> = None;
        for ((name, _), candidate) in self.bins.iter().zip(per_feature) {
            if let Some((gain, threshold)) = candidate {
                if best.is_none_or(|(g, _, _)| gain > g) {
                    best = Some((gain, name.as_str(), threshold));
                }
            }
        }
        let Some((_, name, threshold)) = best else {
            return RegNode::Leaf { value };
        };

        let rule = Splitter::new(name, threshold);
        // Same rule as `Splitter::split` (NaN goes right), with the feature
        // looked up once instead of once per example.
        let (left, right) = sample[name].partition_rows(indices, threshold);
        if left.is_empty() || right.is_empty() {
            return RegNode::Leaf { value };
        }
        RegNode::Branch {
            rule,
            left: Box::new(self.grow(sample, dist, left, depth - 1, cached)),
            right: Box::new(self.grow(sample, dist, right, depth - 1, cached)),
        }
    }
}

impl WeakLearner for RegressionTree {
    type Hypothesis = RegressionTreeRegressor;

    fn name(&self) -> &str {
        "Regression tree"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let n_bins = self.bins.iter().map(|(_, b)| b.len()).max().unwrap_or(0);
        Some(vec![
            ("# of bins (max)", n_bins.to_string()),
            ("Max depth", self.max_depth.to_string()),
        ])
    }

    /// Fits the tree to `sample.target()` under the weights `dist`; see
    /// [`RegressionTree`].
    ///
    /// # Panics
    /// Panics if the sample has no targets, a target is not finite, or
    /// `dist` has the wrong length, a negative or non-finite entry, or no
    /// positive entry.
    fn produce(&self, sample: &Sample, dist: &[f64]) -> Self::Hypothesis {
        let n_examples = sample.shape().0;
        let target = sample.target();
        assert_eq!(target.len(), n_examples, "the sample needs targets");
        assert!(
            target.iter().all(|z| z.is_finite()),
            "regression targets must be finite"
        );
        assert_eq!(dist.len(), n_examples, "one weight per example");
        assert!(
            dist.iter().all(|d| d.is_finite() && *d >= 0.0),
            "weights must be finite and nonnegative"
        );
        let indices: Vec<usize> = (0..n_examples).filter(|&i| dist[i] > 0.0).collect();
        assert!(!indices.is_empty(), "the weights must have positive mass");
        let cached = Arc::ptr_eq(sample.shared_features(), &self.features);
        RegressionTreeRegressor {
            root: self.grow(sample, dist, indices, self.max_depth, cached),
        }
    }
}

/// A node of a [`RegressionTreeRegressor`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegNode {
    /// A split `x[feature] < threshold` (left) / `≥` (right).
    Branch {
        /// The splitting rule.
        rule: Splitter,
        /// The subtree for `x[feature] < threshold`.
        left: Box<Self>,
        /// The subtree for `x[feature] ≥ threshold`.
        right: Box<Self>,
    },
    /// A leaf predicting `value`.
    Leaf {
        /// The weighted mean of the training targets in the leaf.
        value: f64,
    },
}

/// The regression tree returned by [`RegressionTree`]; a [`Regressor`].
///
/// # Examples
/// See the [module documentation](self).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegressionTreeRegressor {
    /// The root node.
    pub root: RegNode,
}

impl RegNode {
    /// Writes the value of every row in `rows` to `out[row]`, looking each
    /// feature up once per node instead of once per row.
    fn predict_rows(&self, sample: &Sample, rows: Vec<usize>, out: &mut [f64]) {
        match self {
            Self::Branch { rule, left, right } => {
                // Same rule as `Splitter::split`: NaN goes right.
                let (lrows, rrows) = sample[&rule.feature].partition_rows(rows, rule.threshold);
                left.predict_rows(sample, lrows, out);
                right.predict_rows(sample, rrows, out);
            }
            Self::Leaf { value } => {
                for row in rows {
                    out[row] = *value;
                }
            }
        }
    }
}

impl Regressor for RegressionTreeRegressor {
    fn predict(&self, sample: &Sample, row: usize) -> f64 {
        let mut node = &self.root;
        loop {
            match node {
                RegNode::Leaf { value } => return *value,
                RegNode::Branch { rule, left, right } => {
                    node = match rule.split(sample, row) {
                        LeftRight::Left => left,
                        LeftRight::Right => right,
                    }
                }
            }
        }
    }

    /// Same values as [`Regressor::predict`] row by row, sending all rows
    /// down the tree together.
    fn predict_all(&self, sample: &Sample) -> Vec<f64> {
        let n_sample = sample.shape().0;
        let mut out = vec![0.0; n_sample];
        self.root
            .predict_rows(sample, (0..n_sample).collect(), &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stump_minimizes_the_weighted_squared_error() {
        // Exhaustive check over the 4 thresholds of 5 distinct points.
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let z = [1.0, 3.0, 2.0, 8.0, 7.0];
        let d = [0.1, 0.3, 0.2, 0.25, 0.15];
        let rows: Vec<Vec<f64>> = xs.iter().map(|x| vec![*x]).collect();
        let sample = Sample::from_dense(&rows, Some(&z)).unwrap();
        let h = RegressionTreeBuilder::new(&sample)
            .max_depth(1)
            .build()
            .produce(&sample, &d);
        let sse = |pred: &dyn Fn(usize) -> f64| -> f64 {
            (0..5).map(|i| d[i] * (z[i] - pred(i)).powi(2)).sum()
        };
        let fitted = sse(&|i| h.predict(&sample, i));
        for split in 1..5 {
            let mean = |r: std::ops::Range<usize>| {
                let w: f64 = r.clone().map(|i| d[i]).sum();
                r.map(|i| d[i] * z[i]).sum::<f64>() / w
            };
            let (l, r) = (mean(0..split), mean(split..5));
            let candidate = sse(&|i| if i < split { l } else { r });
            assert!(fitted <= candidate + 1e-12, "split {split}");
        }
    }

    #[test]
    fn every_split_between_distinct_values_is_a_candidate() {
        // Unevenly spaced values: equal-width bins over [-3, 3] would put 2
        // and 3 in the same bin, although they are distinct.
        let xs = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0];
        let z = [0.0, 0.0, 0.0, 0.0, 0.0, 10.0];
        let rows: Vec<Vec<f64>> = xs.iter().map(|x| vec![*x]).collect();
        let sample = Sample::from_dense(&rows, Some(&z)).unwrap();
        let h = RegressionTreeBuilder::new(&sample)
            .max_depth(1)
            .build()
            .produce(&sample, &[1.0 / 6.0; 6]);
        assert_eq!(h.predict_all(&sample), z.to_vec());
        // The threshold is the midpoint between the two values.
        let at = |x: f64| h.predict(&Sample::from_dense(&[vec![x]], None).unwrap(), 0);
        assert_eq!((at(2.49), at(2.5)), (0.0, 10.0));
    }

    #[test]
    fn adjacent_floats_stay_separable() {
        // The midpoint of adjacent floats rounds to one of them; the boundary
        // must still separate them.
        let a = 1.0_f64;
        let b = f64::from_bits(a.to_bits() + 1);
        let rows = vec![vec![a], vec![b]];
        let sample = Sample::from_dense(&rows, Some(&[0.0, 1.0])).unwrap();
        let h = RegressionTreeBuilder::new(&sample)
            .max_depth(1)
            .build()
            .produce(&sample, &[0.5, 0.5]);
        assert_eq!(h.predict_all(&sample), vec![0.0, 1.0]);
    }

    #[test]
    fn leaves_are_weighted_means_and_zero_weights_are_ignored() {
        let rows: Vec<Vec<f64>> = (0..4).map(|i| vec![f64::from(i)]).collect();
        let sample = Sample::from_dense(&rows, Some(&[1.0, 3.0, 10.0, 100.0])).unwrap();
        let tree = RegressionTreeBuilder::new(&sample).max_depth(0).build();
        // Weight 0 on the last example: mean (1·1 + 1·3 + 2·10) / 4 = 6.
        let h = tree.produce(&sample, &[0.25, 0.25, 0.5, 0.0]);
        assert_eq!(h.predict(&sample, 3), 6.0);
    }

    #[test]
    fn a_step_function_is_fitted_exactly_at_depth_two() {
        // Greedy splitting first separates {0, 1} from {5, 9} (the largest
        // decrease of the squared error), then each half.
        let rows: Vec<Vec<f64>> = (0..8).map(|i| vec![f64::from(i)]).collect();
        let z = vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 9.0, 9.0];
        let sample = Sample::from_dense(&rows, Some(&z)).unwrap();
        let h = RegressionTreeBuilder::new(&sample)
            .max_depth(2)
            .build()
            .produce(&sample, &[0.125; 8]);
        assert_eq!(h.predict_all(&sample), z);
    }

    #[test]
    fn invalid_inputs_panic() {
        let rows = vec![vec![0.0], vec![1.0]];
        let sample = Sample::from_dense(&rows, Some(&[0.0, 1.0])).unwrap();
        let tree = RegressionTreeBuilder::new(&sample).build();
        for d in [vec![0.0, 0.0], vec![-1.0, 1.0], vec![0.5]] {
            assert!(std::panic::catch_unwind(|| tree.produce(&sample, &d)).is_err());
        }
        let bad = sample.with_target(vec![f64::NAN, 1.0]);
        assert!(std::panic::catch_unwind(|| tree.produce(&bad, &[0.5, 0.5])).is_err());
    }
}
