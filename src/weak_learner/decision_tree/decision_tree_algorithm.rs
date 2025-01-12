use rayon::prelude::*;


use crate::{Sample, WeakLearner};
use super::bin::*;


use crate::weak_learner::common::{
    type_and_struct::*,
    split_rule::*,
};
use super::{
    node::*,
    criterion::*,
    train_node::*,
    decision_tree_classifier::DecisionTreeClassifier,
};


use std::fmt;
use std::rc::Rc;
use std::collections::HashMap;


/// The Decision Tree algorithm.  
/// Given a set of training examples for classification
/// and a distribution over the set,
/// [`DecisionTree`] outputs a decision tree classifier
/// named [`DecisionTreeClassifier`]
/// under the specified parameters.
///
/// The code is based on the book:  
/// [Classification and Regression
/// Trees](https://www.amazon.com/Classification-Regression-Wadsworth-Statistics-Probability/dp/0412048418)
/// by Leo Breiman, Jerome H. Friedman, Richard A. Olshen, and Charles J. Stone.
///
/// [`DecisionTree`] is constructed 
/// by [`DecisionTreeBuilder`](crate::weak_learner::DecisionTreeBuilder).
/// 
/// # Example
/// ```no_run
/// use miniboosts::prelude::*;
/// 
/// // Read the training data from the CSV file.
/// let file = "/path/to/data/file.csv";
/// let sample = SampleReader::new()
///     .file(file)
///     .has_header(true)
///     .target_feature("class")
///     .read()
///     .unwrap();
/// 
/// 
/// // Get an instance of decision tree weak learner.
/// // In this example, the output tree is at most depth 2.
/// let tree = DecisionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .criterion(Criterion::Entropy)
///     .build();
///
/// let n_sample = sample.shape()f64;
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
    bins: HashMap<&'a str, Bins>,
    criterion: Criterion,
    max_depth: Depth,
}


impl<'a> DecisionTree<'a> {
    /// Initialize [`DecisionTree`].
    /// This method is called only via `DecisionTreeBuilder::build`.
    #[inline]
    pub(super) fn from_components(
        bins: HashMap<&'a str, Bins>,
        criterion: Criterion,
        max_depth: Depth,
    ) -> Self
    {
        Self { bins, criterion, max_depth, }
    }


    /// Construct a full binary tree of depth `depth`.
    #[inline]
    fn full_tree(
        &self,
        sample: &'a Sample,
        dist: &[f64],
        indices: Vec<usize>,
        criterion: Criterion,
        depth: Depth,
    ) -> TrainNodePtr
    {
        let total_weight = indices.par_iter()
            .copied()
            .map(|i| dist[i])
            .sum::<f64>();


        // Compute the best confidence that minimizes the training error
        // on this node.
        let (conf, loss) = confidence_and_loss(sample, dist, &indices[..]);


        // If sum of `dist` over `train` is zero, construct a leaf node.
        if loss == 0f64 || depth < 1 {
            return TrainNode::leaf(conf, total_weight, loss);
        }


        // Find the best pair of feature name and threshold
        // based on the `criterion`.
        let (feature, threshold) = criterion.best_split(
            &self.bins, sample, dist, &indices[..]
        );


        // Construct the splitting rule
        // from the best feature and threshold.
        let rule = Splitter::new(feature, Threshold::from(threshold));


        // Split the train data for left/right childrens
        let mut lindices = Vec::new();
        let mut rindices = Vec::new();
        for i in indices {
            match rule.split(sample, i) {
                LR::Left  => { lindices.push(i); },
                LR::Right => { rindices.push(i); },
            }
        }


        // If the split has no meaning, construct a leaf node.
        if lindices.is_empty() || rindices.is_empty() {
            return TrainNode::leaf(conf, total_weight, loss);
        }

        // At this point, `depth > 0` is guaranteed so that
        // one can grow the tree.
        let depth = depth - 1;
        let ltree = self.full_tree(sample, dist, lindices, criterion, depth);
        let rtree = self.full_tree(sample, dist, rindices, criterion, depth);


        TrainNode::branch(rule, ltree, rtree, conf, total_weight, loss)
    }
}


impl<'a> WeakLearner for DecisionTree<'a> {
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
            ("Split criterion", format!("{}", self.criterion)),
        ]);
        Some(info)
    }


    /// This method computes as follows;
    /// 1. construct a `TrainNode` which contains some information
    ///     to grow a tree (e.g., impurity, total distribution mass, etc.)
    /// 2. Convert `TrainNode` to `Node` that pares redundant information
    #[inline]
    fn produce(&self, sample: &Sample, dist: &[f64])
        -> Self::Hypothesis
    {
        let n_sample = sample.shape().0;

        let indices = (0..n_sample).filter(|&i| dist[i] > 0f64)
            .collect::<Vec<usize>>();
        assert_ne!(indices.len(), 0);

        let criterion = self.criterion;

        // Construct a large binary tree
        let tree = self.full_tree(
            sample, dist, indices, criterion, self.max_depth
        );


        tree.borrow_mut().remove_redundant_nodes();


        let root = Node::from(
            Rc::try_unwrap(tree)
                .expect("Root node has reference counter >= 1")
                .into_inner()
        );


        DecisionTreeClassifier::from(root)
    }
}


/// This function returns a tuple `(c, l)` where
/// - `c` is the **confidence** for some label `y`
/// that minimizes the training loss.
/// - `l` is the training loss when the confidence is `y`.
/// 
/// **Note that** this function assumes that the label is `+1` or `-1`.
#[inline]
fn confidence_and_loss(sample: &Sample, dist: &[f64], indices: &[usize])
    -> (Confidence<f64>, LossValue)
{

    assert_ne!(indices.len(), 0);
    let target = sample.target();
    let mut counter: HashMap<i64, f64> = HashMap::new();

    for &i in indices {
        let l = target[i] as i64;
        let cnt = counter.entry(l).or_insert(0f64);
        *cnt += dist[i];
    }


    let total = counter.values().sum::<f64>();

    // Compute the max (key, val) that has maximal p(j, t)
    let (label, p) = counter.into_par_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();


    // From the update rule of boosting algorithm,
    // the sum of `dist` over `indices` may become zero,
    let loss = if total > 0f64 { total * (1f64 - (p / total)) } else { 0f64 };

    // `label` takes value in `{-1, +1}`.
    let confidence = if total > 0f64 {
        (label as f64 * (2f64 * (p / total) - 1f64)).clamp(-1f64, 1f64)
    } else {
        (label as f64).clamp(-1f64, 1f64)
    };

    let confidence = Confidence::from(confidence);
    let loss = LossValue::from(loss);
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
            - Splitting criterion: {}\n\
            - Bins:\
            ",
            self.max_depth,
            self.criterion,
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
