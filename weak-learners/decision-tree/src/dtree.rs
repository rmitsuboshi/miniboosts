use rayon::prelude::*;

use miniboosts_core::{
    tree::*,
    binning::*,
    Sample,
    WeakLearner,
};

use crate::{
    split_by::SplitBy,
    node::*,
    classifier::DecisionTreeClassifier,
};

use std::fmt;
use std::collections::HashMap;

/// The Decision Tree algorithm.  
/// Given a set of training examples for classification
/// and a distribution over the set,
/// [`DecisionTree`] outputs a decision tree classifier
/// named [`DecisionTreeClassifier`]
/// under the specified parameters.
///
/// The code is based on the book:  
/// [Classification and Regression Trees](https://www.amazon.com/Classification-Regression-Wadsworth-Statistics-Probability/dp/0412048418)
/// by Leo Breiman, Jerome H. Friedman, Richard A. Olshen, and Charles J. Stone.
///
/// [`DecisionTree`] is constructed 
/// by [`DecisionTreeBuilder`](crate::builder::DecisionTreeBuilder).
/// 
/// # Example
/// ```no_run
/// use miniboosts_core::{
///     Classifier,
///     Sample,
///     SampleReader,
///     WeakLearner,
/// };
/// use decision_tree::{
///     DecisionTreeBuilder,
///     SplitBy,
/// };
/// 
/// // Read the training data from the CSV file.
/// let file = "/path/to/data/file.csv";
/// let sample = SampleReader::default()
///     .file(file)
///     .has_header(true)
///     .target_feature("class")
///     .read()
///     .unwrap();
/// 
/// // Get an instance of decision tree weak learner.
/// // In this example, the output tree is at most depth 2.
/// let tree = DecisionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .split_by(SplitBy::Entropy)
///     .build();
///
/// let n_sample = sample.shape().0;
/// let dist = vec![1f64 / n_sample as f64; n_sample];
/// let f = tree.produce(&sample, &dist);
/// 
/// let predictions = f.predict_all(&sample);
/// 
/// let loss = sample.target()
///     .into_iter()
///     .zip(predictions)
///     .map(|(ty, py)| if *ty == py as f64 { 0f64 } else { 1f64 })
///     .sum::<f64>()
///     / n_sample as f64;
/// println!("loss (train) is: {loss}");
/// ```
pub struct DecisionTree<'a> {
    bins:      HashMap<&'a str, Bins>,
    split_by:  SplitBy,
    max_depth: Depth,
}

impl<'a> DecisionTree<'a> {
    /// Initialize [`DecisionTree`].
    /// This method is called only via `DecisionTreeBuilder::build`.
    #[inline]
    pub(super) fn new(
        bins:      HashMap<&'a str, Bins>,
        split_by:  SplitBy,
        max_depth: Depth,
    ) -> Self
    {
        Self { bins, split_by, max_depth, }
    }

    /// Construct a full binary tree of depth `depth`.
    #[inline]
    fn grow(
        &self,
        sample:   &'a Sample,
        dist:     &[f64],
        indices:  Vec<usize>,
        depth:    Depth,
    ) -> Box<Node>
    {
        // Compute the best confidence that minimizes the training error
        // on this node.
        let (conf, loss) = confidence_and_loss(sample, dist, &indices[..]);

        // If sum of `dist` over `train` is zero, construct a leaf node.
        if loss == 0f64 || depth < 1 {
            return Box::new(Node::leaf(conf));
        }

        // Find the best pair of feature name and threshold
        // based on the `split_by`.
        let (feature, threshold) = self.split_by.best_split(
            &self.bins, sample, dist, &indices[..]
        );

        // Construct the splitting rule
        // from the best feature and threshold.
        let rule = Splitter::new(feature, threshold);

        // Split the train data for left/right childrens
        let mut lindices = Vec::new();
        let mut rindices = Vec::new();
        for i in indices {
            match rule.split(sample, i) {
                LeftRight::Left  => { lindices.push(i); },
                LeftRight::Right => { rindices.push(i); },
            }
        }

        // If the split has no meaning, construct a leaf node.
        if lindices.is_empty() || rindices.is_empty() {
            return Box::new(Node::leaf(conf));
        }

        // At this point, `depth > 0` is guaranteed so that
        // one can grow the tree.
        let depth = depth - 1;
        let left  = self.grow(sample, dist, lindices, depth);
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
        let n_bins = self.bins.values()
            .map(|bin| bin.len())
            .reduce(usize::max)
            .unwrap_or(0);
        let info = Vec::from([
            ("# of bins (max)", format!("{n_bins}")),
            ("Max depth", format!("{}", self.max_depth)),
            ("Split split_by", format!("{}", self.split_by)),
        ]);
        Some(info)
    }

    /// This method computes as follows;
    /// 1. construct a `Node` which contains some information
    ///    to grow a tree (e.g., impurity, total distribution mass, etc.)
    /// 2. Convert `Node` to `Node` that pares redundant information
    #[inline]
    fn produce(&self, sample: &Sample, dist: &[f64])
        -> Self::Hypothesis
    {
        let n_sample = sample.shape().0;

        let indices = (0..n_sample).filter(|&i| dist[i] > 0f64)
            .collect::<Vec<usize>>();
        assert_ne!(
            indices.len(), 0,
            "zero vector is given as a distribution. dist is: {dist:?}"
        );

        // Construct a large binary tree
        let root = self.grow(sample, dist, indices, self.max_depth);

        DecisionTreeClassifier::from(root)
    }
}

/// This function returns a tuple `(c, l)` where
/// `c` is the **confidence** for some label `y`
/// that minimizes the training loss.
/// - `l` is the training loss when the confidence is `y`.
/// 
/// **Note that** this function assumes that the label is `+1` or `-1`.
#[inline]
fn confidence_and_loss(sample: &Sample, dist: &[f64], indices: &[usize])
    -> (f64, f64)
{

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
    let (y, p) = counter.into_par_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    // From the update rule of boosting algorithm,
    // the sum of `dist` over `indices` may become zero,
    let loss = if total > 0f64 { total * (1f64 - (p / total)) } else { 0f64 };

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
            - Bins:\
            ",
            self.max_depth,
            self.split_by,
        )?;

        let width = self.bins.keys()
            .map(|key| key.len())
            .max()
            .expect("Tried to print bins, but no features are found");
        let max_bin_width = self.bins.values()
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

