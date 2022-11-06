use polars::prelude::*;
use rayon::prelude::*;


use crate::BaseLearner;


use super::node::*;
use super::criterion::*;
use super::split_rule::*;
use super::train_node::*;
use super::dtree_classifier::DTreeClassifier;


use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;



/// Generates a `DTreeClassifier` for a given distribution
/// over examples.
pub struct DTree {
    criterion: Criterion,
    max_depth: Option<usize>,
    size: usize,
}


impl DTree {
    /// Initialize `DTree`.
    #[inline]
    pub fn init(df: &DataFrame) -> Self {
        let criterion = Criterion::Entropy;
        let max_depth = None;
        let size = df.shape().0;

        Self {
            criterion,
            max_depth,
            size,
        }
    }


    /// Specify the maximal depth of the tree.
    /// Default maximul depth is `None`.
    pub fn max_depth(mut self, depth: usize) -> Self {
        assert!(depth > 0);
        self.max_depth = Some(depth);

        self
    }


    /// Set criterion for node splitting.
    #[inline]
    pub fn criterion(mut self, criterion: Criterion) -> Self {
        self.criterion = criterion;
        self
    }
}


impl BaseLearner for DTree {
    type Clf = DTreeClassifier;
    fn produce(&self, data: &DataFrame, target: &Series, dist: &[f64])
        -> Self::Clf
    {
        let mut indices = (0..self.size).into_iter()
            // .filter(|&i| dist[i] > 0.0)
            .collect::<Vec<usize>>();

        indices.sort_by(|&i, &j| dist[i].partial_cmp(&dist[j]).unwrap());

        let criterion = self.criterion;
        let depth = self.max_depth;

        // Construct a large binary tree
        let tree = full_tree(data, target, dist, indices, criterion, depth);


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
fn full_tree(data: &DataFrame,
             target: &Series,
             dist: &[f64],
             indices: Vec<usize>,
             criterion: Criterion,
             max_depth: Option<usize>)
    -> Rc<RefCell<TrainNode>>
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


    // grow the tree.
    let ltree; // Left child
    let rtree; // Right child
    match max_depth {
        Some(depth) => {
            if depth == 1 {
                // If `depth == 1`,
                // the childs from this node must be leaves.
                ltree = construct_leaf(target, dist, lindices);
                rtree = construct_leaf(target, dist, rindices);
            } else {
                // If `depth > 1`,
                // the childs from this node might be branches.
                let d = Some(depth - 1);
                ltree = full_tree(data, target, dist, lindices, criterion, d);
                rtree = full_tree(data, target, dist, rindices, criterion, d);
            }
        },
        None => {
            ltree = full_tree(data, target, dist, lindices, criterion, None);
            rtree = full_tree(data, target, dist, rindices, criterion, None);
        }
    }


    TrainNode::branch(rule, ltree, rtree, pred, total_weight, loss)
}


#[inline]
fn construct_leaf(target: &Series,
                  dist: &[f64],
                  indices: Vec<usize>)
    -> Rc<RefCell<TrainNode>>
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


