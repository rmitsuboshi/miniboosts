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
use std::cell::RefCell;
use std::collections::HashMap;


/// `DecisionTree` is the factory that
/// generates a `DecisionTreeClassifier` for a given distribution over examples.
/// 
/// See also:
/// - [`Criterion`](Criterion)
/// 
/// # Example
/// ```no_run
/// use miniboosts::prelude::*;
/// 
/// // Read the training data from the CSV file.
/// let file = "/path/to/data/file.csv";
/// let has_header = true;
/// let sample = Sample::from_csv(file, has_header)
///     .unwrap()
///     .set_target("class");
/// 
/// 
/// // Get an instance of decision tree weak learner.
/// // In this example,
/// // the output hypothesis is at most depth 2.
/// // Further, this example uses `Criterion::Edge` for splitting rule.
/// let weak_learner = DecisionTree::init(&sample)
///     .max_depth(2)
///     .criterion(Criterion::Edge);
/// ```
pub struct DecisionTree<'a> {
    bins: HashMap<&'a str, Bins>,
    criterion: Criterion,
    max_depth: Depth,
}


impl<'a> DecisionTree<'a> {
    /// Initialize [`DecisionTree`](DecisionTree).
    /// This method is called only via `DecisionTreeBuilder::build()`.
    #[inline]
    pub(crate) fn from_components(
        bins: HashMap<&'a str, Bins>,
        criterion: Criterion,
        max_depth: Depth,
    ) -> Self
    {
        Self { bins, criterion, max_depth, }
    }


    /// Construct a full binary tree
    /// that perfectly classify the given examples.
    #[inline]
    fn full_tree(
        &self,
        sample: &'a Sample,
        dist: &[f64],
        indices: Vec<usize>,
        criterion: Criterion,
        depth: Depth,
    ) -> Rc<RefCell<TrainNode>>
    {

        let total_weight = indices.par_iter()
            .copied()
            .map(|i| dist[i])
            .sum::<f64>();


        // Compute the best confidence that minimizes the training error
        // on this node.
        let (conf, loss) = confidence_and_loss(sample, dist, &indices[..]);


        // If sum of `dist` over `train` is zero, construct a leaf node.
        if loss == 0.0 || depth <= 1 {
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

        // At this point, `depth > 1` is guaranteed so that
        // one can grow the tree.
        let depth = depth - 1;
        let ltree = self.full_tree(sample, dist, lindices, criterion, depth);
        let rtree = self.full_tree(sample, dist, rindices, criterion, depth);


        TrainNode::branch(rule, ltree, rtree, conf, total_weight, loss)
    }
}


impl<'a> WeakLearner for DecisionTree<'a> {
    type Hypothesis = DecisionTreeClassifier;
    /// This method computes as follows;
    /// 1. construct a `TrainNode` which contains some information
    ///     to grow a tree (e.g., impurity, total distribution mass, etc.)
    /// 2. Convert `TrainNode` to `Node` that pares redundant information
    #[inline]
    fn produce(&self, sample: &Sample, dist: &[f64])
        -> Self::Hypothesis
    {
        let n_sample = sample.shape().0;

        let indices = (0..n_sample).filter(|&i| dist[i] > 0.0)
            .collect::<Vec<usize>>();

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
    let target = sample.target();
    let mut counter: HashMap<i64, f64> = HashMap::new();

    for &i in indices {
        let l = target[i] as i64;
        let cnt = counter.entry(l).or_insert(0.0);
        *cnt += dist[i];
    }


    let total = counter.values().sum::<f64>();


    // Compute the max (key, val) that has maximal p(j, t)
    let (label, p) = counter.into_par_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();


    // From the update rule of boosting algorithm,
    // the sum of `dist` over `indices` may become zero,
    let loss = if total > 0.0 {
        total * (1.0 - (p / total))
    } else {
        0.0
    };

    // `label` takes value in `{-1, +1}`.
    let confidence = if total > 0.0 {
        label as f64 * (2.0 * (p / total) - 1.0)
    } else {
        label as f64
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
