use polars::prelude::*;
use rayon::prelude::*;


use crate::BaseLearner;


use super::rtree_regressor::RTreeRegressor;
use super::split_rule::*;
use super::node::*;
use super::train_node::*;
use super::loss::Loss;


use std::rc::Rc;
use std::cell::RefCell;


/// This struct produces a regression tree for the given distribution.
pub struct RTree {
    // The maximal depth of the output trees
    max_depth: Option<usize>,

    // The number of training instances
    size: usize,


    // Loss function
    loss_type: Loss,
}


impl RTree {
    /// Initialize `RTree`.
    #[inline]
    pub fn init(df: &DataFrame) -> Self
    {
        let max_depth = None;
        let size = df.shape().0;

        Self {
            max_depth,
            size,

            loss_type: Loss::L2,
        }
    }


    /// Set the maximum depth of the resulting tree.
    #[inline]
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }


    /// Set the loss type.
    #[inline]
    pub fn loss_type(mut self, loss: Loss) -> Self {
        self.loss_type = loss;
        self
    }
}


impl BaseLearner for RTree {
    type Clf = RTreeRegressor;
    fn produce(&self, data: &DataFrame, target: &Series, dist: &[f64])
        -> Self::Clf
    {
        let indices = (0..self.size).into_iter()
            .collect::<Vec<usize>>();

        let depth = self.max_depth;

        let tree = full_tree(
            data, target, dist, indices, depth, self.loss_type
        );


        tree.borrow_mut().remove_redundant_nodes();


        let root = Node::from(
            Rc::try_unwrap(tree)
                .expect("Root node has reference counter >= 1")
                .into_inner()
        );

        RTreeRegressor::from(root)
    }
}


#[inline]
fn full_tree(
    data: &DataFrame,
    target: &Series,
    dist: &[f64],
    indices: Vec<usize>,
    max_depth: Option<usize>,
    loss_type: Loss
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


    // Find the best splitting rule.
    let (feature, threshold) = loss_type.best_split(
        data, target, dist, &indices[..],
    );

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
                ltree = full_tree(data, target, dist, lindices, d, loss_type);
                rtree = full_tree(data, target, dist, rindices, d, loss_type);
            }
        },
        None => {
            ltree = full_tree(data, target, dist, lindices, None, loss_type);
            rtree = full_tree(data, target, dist, rindices, None, loss_type);
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
/// - `y` is the mean of the target values, and
/// - `e` is the training loss when the prediction is `y`.
#[inline]
fn calc_loss_as_leaf(target: &Series, dist: &[f64], indices: &[usize])
    -> (f64, f64)
{
    let target = target.f64()
        .expect("The target class is not a dtype i64");

    let tuples = indices.into_par_iter()
        .copied()
        .map(|i| {
            let y = target.get(i).unwrap();
            (dist[i], y)
        })
        .collect::<Vec<(f64, f64)>>();

    let sum_dist = tuples.iter()
        .map(|(d, _)| *d)
        .sum::<f64>();

    if sum_dist == 0.0 {
        return (0.0, 0.0);
    }

    let mean_y = tuples.iter()
        .map(|(d, y)| *d * *y)
        .sum::<f64>()
        / sum_dist;


    let l2_loss = tuples.into_iter()
        .map(|(d, y)| d * (y - mean_y).powi(2))
        .sum::<f64>()
        / sum_dist;


    (mean_y, l2_loss)
}

