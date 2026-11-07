use rayon::prelude::*;

use crate::core::{Sample, WeakLearner, binning::*, tree::*};

use crate::weak_learners::decision_tree::{
    classifier::DecisionTreeClassifier, node::*, split_by::SplitBy,
};

use std::collections::HashMap;
use std::fmt;

/// A weighted binary decision-tree weak learner (CART-style) for labels
/// `-1` and `+1`.
///
/// Given a sample and a distribution `d` over its examples (chosen by a
/// booster), [`WeakLearner::produce`] grows a [`DecisionTreeClassifier`]
/// whose leaves output confidence values in `[-1, 1]`. Construct it with
/// [`DecisionTreeBuilder`](crate::DecisionTreeBuilder).
///
/// # Algorithm
/// **Weighted training.** Only examples with `d_i > 0` are used. Every
/// quantity below is a sum of the booster's weights `d_i`, not of example
/// counts: for a node `S` and label `y`,
/// ```txt
/// W_S   = Σ_{i ∈ S} d_i,       W_{S,y} = Σ_{i ∈ S, y_i = y} d_i,
/// p_{y|S} = W_{S,y} / W_S.
/// ```
/// Weights are not renormalized per node.
///
/// **Leaf confidence.** Every node `S` stores the confidence
/// ```txt
/// c_S = y* (2 p_{y*|S} - 1) = (W_{S,+1} - W_{S,-1}) / (W_{S,+1} + W_{S,-1}),
/// ```
/// where `y*` is the label with the largest weight in `S` (the second form
/// holds for labels `±1`). Hence `c_S ∈ [-1, 1]`; its sign is the weighted
/// majority label and its magnitude the weighted purity. A tie gives `0`,
/// which [`crate::Classifier::predict`] maps to `+1`. Only leaf
/// confidences are used for prediction. With [`LeafConfidence::Sign`], every
/// node stores the sign of `c_S` instead (`+1` for a tie).
///
/// **Growth.** Starting from the root with the depth budget `max_depth`, a
/// node becomes a leaf if
/// 1. its depth budget is `0`,
/// 2. it is pure, i.e., its weighted misclassification mass
///    `W_S - W_{S,y*}` is exactly `0`, or
/// 3. the best split found by [`SplitBy`] sends all of its examples to one
///    side.
///
/// Otherwise the node is split by the rule `x[f] < t` (left) / `x[f] >= t`
/// (right) and both children are grown with the budget reduced by one.
/// Thus `max_depth = k` yields at most `k` levels of splits and at most
/// `2^k` leaves; `max_depth = 1` gives a decision stump.
///
/// **Candidate thresholds.** Each feature is cut once, when the builder is
/// built, into equal-width bins (see [`crate::core::binning::Bins::cut`];
/// by default `min(number of distinct values, 255)` bins), or, with
/// [`DecisionTreeBuilder::exact_splits`](crate::DecisionTreeBuilder::exact_splits),
/// at the midpoints between consecutive distinct values. At each node the
/// candidate thresholds for a feature are the boundaries between the bins
/// that contain at least one of the node's examples. The criterion is
/// evaluated for every candidate of every feature; see [`SplitBy`] for the
/// formulas.
///
/// # Theoretical guarantee
/// A decision stump (`max_depth = 1`) built with
/// [`exact_splits`](crate::DecisionTreeBuilder::exact_splits),
/// [`SplitBy::Edge`], and [`LeafConfidence::Sign`] maximizes the edge exactly
/// (up to rounding) over the class of the constants `±1` and the stumps
/// `±sign(x[f] - t)` for every feature `f` and threshold `t`. The edge
/// criterion maximizes `|Δ_R - Δ_L|` over the thresholds and the unsplit
/// node, where `Δ_A = Σ_{i ∈ A} d_i y_i`; since `Δ_L + Δ_R = Δ_S`, the edge
/// of the stump with leaves `sign(Δ_L)` and `sign(Δ_R)` is
/// `|Δ_L| + |Δ_R| = max(|Δ_R - Δ_L|, |Δ_S|)`, which is the largest edge in
/// that class.
///
/// Otherwise there is no guarantee. In particular, a depth-limited tree is
/// not guaranteed to satisfy a booster's weak-learning assumption (edge
/// `>= γ > 0` on every distribution), and deeper trees are grown greedily.
///
/// # Limitations
/// - Labels must be `-1` and `+1`. They are converted with `as i64` /
///   `as i32` and not validated here; other values (e.g., `0`/`1`) yield
///   meaningless confidences.
/// - Thresholds are restricted to bin boundaries, so the tree may miss the
///   best split on the raw values. Bins are fixed by the sample given to the
///   builder; `produce` must be called on a sample with the same feature
///   names. Values outside the builder's range fall into the outer bins.
/// - No pruning, minimum leaf size, or impurity-decrease threshold, unlike
///   the cost-complexity pruning of the original CART method.
/// - Per-bin label weights are accumulated in hash maps whose iteration
///   order varies between processes; the resulting floating-point rounding
///   differences can change the chosen split when candidates nearly tie.
///
/// # Panics
/// [`WeakLearner::produce`] panics if no entry of `dist` is positive, if
/// `dist` is shorter than the sample, if the sample has a feature unknown
/// to the builder, or if feature values or criterion scores are NaN.
///
/// # Reference
/// Leo Breiman, Jerome H. Friedman, Richard A. Olshen, and Charles J. Stone.
/// [Classification and Regression Trees](https://www.amazon.co.jp/-/en/Leo-Breiman/dp/0412048418).
/// Wadsworth, 1984.
///
/// # Examples
/// Train a decision stump directly with a distribution via
/// [`WeakLearner::produce`].
/// ```
/// use std::io::BufReader;
/// use miniboosts::{Classifier, Sample, WeakLearner};
/// use miniboosts::{DecisionTreeBuilder, SplitBy};
///
/// let csv = b"x,class\n-2,-1\n-1,-1\n1,1\n2,1\n";
/// let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
///     .unwrap()
///     .set_target("class");
///
/// // A decision stump that splits by the entropic impurity.
/// let tree = DecisionTreeBuilder::new(&sample)
///     .max_depth(1)
///     .split_by(SplitBy::Entropy)
///     .build();
///
/// let n_sample = sample.shape().0;
/// let dist = vec![1f64 / n_sample as f64; n_sample];
/// let f = tree.produce(&sample, &dist);
///
/// // Both leaves are pure, so the confidences are exactly -1 and +1.
/// assert_eq!(f.confidence_all(&sample), vec![-1.0, -1.0, 1.0, 1.0]);
/// assert_eq!(f.predict_all(&sample), vec![-1, -1, 1, 1]);
/// ```
///
/// Leaf confidences are weighted purities `(W₊ - W₋) / (W₊ + W₋)`. Here the
/// only feature is constant, so no split is possible and the tree is a
/// single leaf whose confidence depends on the distribution alone.
/// ```
/// use miniboosts::{Classifier, DecisionTreeBuilder, Sample, WeakLearner};
///
/// let rows = vec![vec![0.0]; 3];
/// let sample = Sample::from_dense(&rows, Some(&[1.0, 1.0, -1.0])).unwrap();
/// let tree = DecisionTreeBuilder::new(&sample).max_depth(2).build();
///
/// // (0.75 - 0.25) / 1 = 0.5, exact with dyadic weights.
/// let h = tree.produce(&sample, &[0.5, 0.25, 0.25]);
/// assert_eq!(h.confidence_all(&sample), vec![0.5; 3]);
///
/// // Examples with zero weight are ignored: only the negative one remains.
/// let h = tree.produce(&sample, &[0.0, 0.0, 1.0]);
/// assert_eq!(h.predict_all(&sample), vec![-1; 3]);
///
/// // For any distribution, confidences lie in [-1, 1].
/// let h = tree.produce(&sample, &[0.2, 0.3, 0.5]);
/// assert!(h.confidence_all(&sample).iter().all(|c| (-1.0..=1.0).contains(c)));
/// ```
pub struct DecisionTree<'a> {
    bins: HashMap<&'a str, Bins>,
    split_by: SplitBy,
    max_depth: Depth,
    leaf_confidence: LeafConfidence,
}

/// What a [`DecisionTree`]'s nodes predict.
///
/// # Examples
/// With weights `3/4` on `+1` and `1/4` on `-1` in a single leaf, the weighted
/// purity is `1/2` and the sign is `1`.
/// ```
/// use miniboosts::{Classifier, DecisionTreeBuilder, LeafConfidence, Sample, WeakLearner};
///
/// let sample = Sample::from_dense(&[vec![0.0], vec![0.0]], Some(&[1.0, -1.0])).unwrap();
/// let dist = [0.75, 0.25];
/// for (leaf, expected) in [(LeafConfidence::Purity, 0.5), (LeafConfidence::Sign, 1.0)] {
///     let tree = DecisionTreeBuilder::new(&sample).leaf_confidence(leaf).build();
///     assert_eq!(tree.produce(&sample, &dist).confidence(&sample, 0), expected);
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LeafConfidence {
    /// The weighted purity `(W_{S,+1} - W_{S,-1}) / (W_{S,+1} + W_{S,-1})`.
    /// This is the default.
    #[default]
    Purity,
    /// The sign of the weighted purity, with `+1` for a tie. Makes a stump an
    /// exact edge maximizer; see [`DecisionTree`].
    Sign,
}

impl fmt::Display for LeafConfidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Purity => write!(f, "Purity"),
            Self::Sign => write!(f, "Sign"),
        }
    }
}

impl<'a> DecisionTree<'a> {
    /// Initialize [`DecisionTree`].
    /// This method is called only via `DecisionTreeBuilder::build`.
    #[inline]
    pub(super) fn new(
        bins: HashMap<&'a str, Bins>,
        split_by: SplitBy,
        max_depth: Depth,
        leaf_confidence: LeafConfidence,
    ) -> Self {
        Self {
            bins,
            split_by,
            max_depth,
            leaf_confidence,
        }
    }

    /// Recursively grows the subtree for the examples `indices` with the
    /// remaining depth budget `depth`. The tree need not be full: pure
    /// nodes and nodes without a useful split become leaves early.
    #[inline]
    fn grow(
        &self,
        sample: &'a Sample,
        dist: &[f64],
        indices: Vec<usize>,
        depth: Depth,
    ) -> Box<Node> {
        // Compute the best confidence that minimizes the training error
        // on this node.
        let (conf, loss) = confidence_and_loss(sample, dist, &indices[..]);
        let conf = match self.leaf_confidence {
            LeafConfidence::Purity => conf,
            LeafConfidence::Sign => {
                if conf >= 0.0 {
                    1.0
                } else {
                    -1.0
                }
            }
        };

        // If the node is pure (zero weighted misclassification)
        // or the depth budget is exhausted, construct a leaf node.
        if loss == 0f64 || depth < 1 {
            return Box::new(Node::leaf(conf));
        }

        // Find the best pair of feature name and threshold
        // based on the `split_by`.
        let (feature, threshold) = self
            .split_by
            .best_split(&self.bins, sample, dist, &indices[..]);

        // Construct the splitting rule
        // from the best feature and threshold.
        let rule = Splitter::new(feature, threshold);

        // Split the train data for left/right childrens
        let mut lindices = Vec::new();
        let mut rindices = Vec::new();
        for i in indices {
            match rule.split(sample, i) {
                LeftRight::Left => {
                    lindices.push(i);
                }
                LeftRight::Right => {
                    rindices.push(i);
                }
            }
        }

        // If the split has no meaning, construct a leaf node.
        if lindices.is_empty() || rindices.is_empty() {
            return Box::new(Node::leaf(conf));
        }

        // At this point, `depth > 0` is guaranteed so that
        // one can grow the tree.
        let depth = depth - 1;
        let left = self.grow(sample, dist, lindices, depth);
        let right = self.grow(sample, dist, rindices, depth);

        Box::new(Node::branch(rule, left, right, conf))
    }
}

impl WeakLearner for DecisionTree<'_> {
    type Hypothesis = DecisionTreeClassifier;

    fn name(&self) -> &str {
        "Decision Tree"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let n_bins = self
            .bins
            .values()
            .map(|bin| bin.len())
            .reduce(usize::max)
            .unwrap_or(0);
        let info = Vec::from([
            ("# of bins (max)", format!("{n_bins}")),
            ("Max depth", format!("{}", self.max_depth)),
            ("Split criterion", format!("{}", self.split_by)),
            ("Leaf confidence", format!("{}", self.leaf_confidence)),
        ]);
        Some(info)
    }

    /// Grows a tree on the examples with `dist[i] > 0`, weighting each by
    /// `dist[i]`; see [`DecisionTree`] for the algorithm and panics.
    #[inline]
    fn produce(&self, sample: &Sample, dist: &[f64]) -> Self::Hypothesis {
        let n_sample = sample.shape().0;

        let indices = (0..n_sample)
            .filter(|&i| dist[i] > 0f64)
            .collect::<Vec<usize>>();
        assert_ne!(
            indices.len(),
            0,
            "zero vector is given as a distribution. dist is: {dist:?}"
        );

        // Construct a large binary tree
        let root = self.grow(sample, dist, indices, self.max_depth);

        DecisionTreeClassifier::from(root)
    }
}

/// This function returns a tuple `(c, l)` where
/// - `c = y* (2 p - 1)` is the **confidence** of the node, `y*` being the
///   label with the largest total weight `w*` and `p = w* / total`;
/// - `l = total - w*` is the weighted misclassification mass of predicting
///   `y*` on this node.
///
/// **Note that** this function assumes that the label is `+1` or `-1`.
#[inline]
fn confidence_and_loss(sample: &Sample, dist: &[f64], indices: &[usize]) -> (f64, f64) {
    assert_ne!(indices.len(), 0);
    let target = sample.target();
    let mut counter: HashMap<i64, f64> = HashMap::new();

    for &i in indices {
        let y = target[i] as i64;
        let cnt = counter.entry(y).or_insert(0f64);
        *cnt += dist[i];
    }

    let total = counter.values().sum::<f64>();

    // Compute the max (key, val) that has maximal p(j, t)
    let (y, p) = counter
        .into_par_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    // From the update rule of boosting algorithm,
    // the sum of `dist` over `indices` may become zero,
    let loss = if total > 0f64 {
        total * (1f64 - (p / total))
    } else {
        0f64
    };

    // `label` takes value in `{-1, +1}`.
    let confidence = if total > 0f64 {
        (y as f64 * (2f64 * (p / total) - 1f64)).clamp(-1f64, 1f64)
    } else {
        (y as f64).clamp(-1f64, 1f64)
    };

    (confidence, loss)
}

impl fmt::Display for DecisionTree<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\
            ----------\n\
            # Decision Tree Weak Learner\n\n\
            - Max depth: {}\n\
            - Splitting split_by: {}\n\
            - Leaf confidence: {}\n\
            - Bins:\
            ",
            self.max_depth, self.split_by, self.leaf_confidence,
        )?;

        let width = self
            .bins
            .keys()
            .map(|key| key.len())
            .max()
            .expect("Tried to print bins, but no features are found");
        let max_bin_width = self
            .bins
            .values()
            .map(|bin| bin.len().ilog10() as usize)
            .max()
            .expect("Tried to print bins, but no features are found")
            + 1;
        for (feat_name, feat_bins) in self.bins.iter() {
            let n_bins = feat_bins.len();
            writeln!(
                f,
                "\
                \t* [{feat_name: <width$} | \
                {n_bins: >max_bin_width$} bins]  \
                {feat_bins}\
                "
            )?;
        }

        write!(f, "----------")
    }
}

#[cfg(test)]
mod tests {
    use crate::{Classifier, DecisionTreeBuilder, LeafConfidence, Sample, SplitBy, WeakLearner};

    // A deterministic generator for test data (xorshift64).
    fn uniforms(seed: u64, n: usize) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1u64 << 53) as f64
            })
            .collect()
    }

    // The largest edge over the constants ±1 and the stumps ±sign(x[f] - t),
    // by enumerating every threshold between distinct values.
    fn best_stump_edge(rows: &[Vec<f64>], y: &[f64], d: &[f64]) -> f64 {
        let total: f64 = (0..y.len()).map(|i| d[i] * y[i]).sum();
        let mut best = total.abs();
        for f in 0..rows[0].len() {
            let mut values: Vec<f64> = rows.iter().map(|r| r[f]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values.dedup();
            for pair in values.windows(2) {
                let t = pair[0] / 2.0 + pair[1] / 2.0;
                let edge: f64 = (0..y.len())
                    .map(|i| d[i] * y[i] * if rows[i][f] < t { -1.0 } else { 1.0 })
                    .sum();
                best = best.max(edge.abs());
            }
        }
        best
    }

    #[test]
    fn exact_sign_stump_maximizes_the_edge() {
        let m = 40;
        for seed in 1..=20u64 {
            let u = uniforms(seed, 4 * m);
            // Coarse values in the first feature create ties.
            let rows: Vec<Vec<f64>> = (0..m)
                .map(|i| vec![(u[i] * 5.0).floor(), u[m + i] * 100.0 - 50.0])
                .collect();
            let y: Vec<f64> = (0..m)
                .map(|i| if u[2 * m + i] < 0.5 { -1.0 } else { 1.0 })
                .collect();
            let total: f64 = u[3 * m..].iter().sum();
            let d: Vec<f64> = u[3 * m..].iter().map(|w| w / total).collect();
            let sample = Sample::from_dense(&rows, Some(&y)).unwrap();
            let stump = DecisionTreeBuilder::new(&sample)
                .max_depth(1)
                .split_by(SplitBy::Edge)
                .exact_splits()
                .leaf_confidence(LeafConfidence::Sign)
                .build();
            let h = stump.produce(&sample, &d);
            let edge: f64 = (0..m).map(|i| d[i] * y[i] * h.confidence(&sample, i)).sum();
            assert!(h.confidence_all(&sample).iter().all(|c| c.abs() == 1.0));
            // Both sides sum the same 40 terms in different orders.
            assert!(
                (edge - best_stump_edge(&rows, &y, &d)).abs() < 1e-12,
                "seed {seed}"
            );
        }
    }

    #[test]
    fn the_default_tree_is_unchanged_by_the_new_options() {
        let rows = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let y = [-1.0, 1.0, -1.0, 1.0];
        let sample = Sample::from_dense(&rows, Some(&y)).unwrap();
        let d = [0.1, 0.2, 0.3, 0.4];
        let default = DecisionTreeBuilder::new(&sample)
            .build()
            .produce(&sample, &d);
        let purity = DecisionTreeBuilder::new(&sample)
            .leaf_confidence(LeafConfidence::Purity)
            .build()
            .produce(&sample, &d);
        assert_eq!(default, purity);
    }
}
