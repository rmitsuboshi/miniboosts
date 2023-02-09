use polars::prelude::*;
use rayon::prelude::*;


use crate::WeakLearner;


use super::{
    node::*,
    criterion::*,
    split_rule::*,
    train_node::*,
    dtree_classifier::DTreeClassifier,
};


use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;

use std::ops;
use std::cmp;


/// Struct `Depth` defines the maximal depth of a tree.
/// This is just a wrapper for `usize`.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub(crate) struct Depth(usize);


impl ops::Sub<usize> for Depth {
    type Output = Self;
    /// Define the subtraction of the `Depth` struct.
    /// The subtraction does not return a value less than or equals to 1.
    #[inline]
    fn sub(self, other: usize) -> Self::Output {
        if self.0 <= 1 {
            self
        } else {
            Self(self.0 - other)
        }
    }
}

impl cmp::PartialEq<usize> for Depth {
    #[inline]
    fn eq(&self, rhs: &usize) -> bool {
        self.0.eq(rhs)
    }
}


impl cmp::PartialOrd<usize> for Depth {
    #[inline]
    fn partial_cmp(&self, other: &usize) -> Option<cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}


/// Generates a `DTreeClassifier` for a given distribution
/// over examples.
pub struct DTree {
    criterion: Criterion,
    max_depth: Depth,
}


impl DTree {
    /// Initialize [`DTree`](DTree).
    #[inline]
    pub fn init(data: &DataFrame, _target: &Series) -> Self {
        let criterion = Criterion::Entropy;
        let size = data.shape().0;
        let depth = ((size as f64).log10() + 1.0).ceil() as usize;

        Self {
            criterion,
            max_depth: Depth(depth),
        }
    }


    /// Specify the maximal depth of the tree.
    /// Default maximal depth is `log` of number of training examples
    pub fn max_depth(mut self, depth: usize) -> Self {
        assert!(depth > 0);
        self.max_depth = Depth(depth);

        self
    }


    /// Set criterion for node splitting.
    /// See [Criterion](Criterion).
    #[inline]
    pub fn criterion(mut self, criterion: Criterion) -> Self {
        self.criterion = criterion;
        self
    }
}


impl WeakLearner for DTree {
    type Hypothesis = DTreeClassifier;
    /// This method computes as follows;
    /// 1. construct a `TrainNode` which contains some information
    ///     to grow a tree (e.g., impurity, total distribution mass, etc.)
    /// 2. Convert `TrainNode` to `Node` that pares redundant information
    #[inline]
    fn produce(&self, data: &DataFrame, target: &Series, dist: &[f64])
        -> Self::Hypothesis
    {
        let n_sample = data.shape().0;

        let mut indices = (0..n_sample).into_iter()
            .filter(|&i| dist[i] > 0.0)
            .collect::<Vec<usize>>();

        indices.sort_by(|&i, &j| dist[i].partial_cmp(&dist[j]).unwrap());

        let criterion = self.criterion;

        // Construct a large binary tree
        let tree = full_tree(
            data, target, dist, indices, criterion, self.max_depth
        );


        tree.borrow_mut().remove_redundant_nodes();


        let root = Node::from(
            Rc::try_unwrap(tree)
                .expect("Root node has reference counter >= 1")
                .into_inner()
        );


        DTreeClassifier::from(root)
    }
}



/// Construct a full binary tree
/// that perfectly classify the given examples.
#[inline]
fn full_tree(
    data: &DataFrame,
    target: &Series,
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


    // Compute the best prediction that minimizes the training error
    // on this node.
    let (pred, loss) = calc_loss_as_leaf(target, dist, &indices[..]);


    // If sum of `dist` over `train` is zero, construct a leaf node.
    if loss == 0.0 {
        return TrainNode::leaf(pred, total_weight, loss);
    }


    // Find the best pair of feature name and threshold
    // based on the `criterion`.
    let (feature, threshold) = criterion.best_split(
        data, target, dist, &indices[..]
    );


    // Construct the splitting rule
    // from the best feature and threshold.
    let rule = Splitter::new(feature, threshold);


    // Split the train data for left/right childrens
    let mut lindices = Vec::new();
    let mut rindices = Vec::new();
    for i in indices.into_iter() {
        match rule.split(data, i) {
            LR::Left  => { lindices.push(i); },
            LR::Right => { rindices.push(i); },
        }
    }


    // If the split has no meaning, construct a leaf node.
    if lindices.is_empty() || rindices.is_empty() {
        return TrainNode::leaf(pred, total_weight, loss);
    }


    // Grow the tree.
    let ltree; // Left child
    let rtree; // Right child

    if depth <= 1 {
        // If `depth == 1`,
        // the childs from this node must be leaves.
        ltree = construct_leaf(target, dist, lindices);
        rtree = construct_leaf(target, dist, rindices);
    } else {
        // If `depth > 1`,
        // the childs from this node might be branches.
        let depth = depth - 1;
        ltree = full_tree(data, target, dist, lindices, criterion, depth);
        rtree = full_tree(data, target, dist, rindices, criterion, depth);
    }


    TrainNode::branch(rule, ltree, rtree, pred, total_weight, loss)
}


#[inline]
fn construct_leaf(
    target: &Series,
    dist: &[f64],
    indices: Vec<usize>
) -> Rc<RefCell<TrainNode>>
{
    // Compute the best prediction that minimizes the training error
    // on this node.
    let (pred, loss) = calc_loss_as_leaf(target, dist, &indices[..]);


    let total_weight = indices.iter()
            .copied()
            .map(|i| dist[i])
            .sum::<f64>();


    TrainNode::leaf(pred, total_weight, loss)
}



/// This function returns a tuple `(y, e)` where
/// - `y` is the prediction label that minimizes the training loss.
/// - `e` is the training loss when the prediction is `y`.
#[inline]
fn calc_loss_as_leaf(target: &Series, dist: &[f64], indices: &[usize])
    -> (i64, f64)
{
    let target = target.i64()
        .expect("The target class is not a dtype i64");
    let mut counter: HashMap<i64, f64> = HashMap::new();

    for &i in indices {
        let l = target.get(i).unwrap();
        let cnt = counter.entry(l).or_insert(0.0);
        *cnt += dist[i];
    }


    let total = counter.values().sum::<f64>();


    // Compute the max (key, val) that has maximal p(j, t)
    let (l, p) = counter.into_par_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();


    // From the update rule of boosting algorithm,
    // the sum of `dist` over `indices` may become zero,
    let node_err = if total > 0.0 {
        total * (1.0 - (p / total))
    } else {
        0.0
    };

    (l, node_err)
}


