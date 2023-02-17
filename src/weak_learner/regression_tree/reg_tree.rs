use polars::prelude::*;
use rayon::prelude::*;


use crate::WeakLearner;


use crate::weak_learner::common::{
    split_rule::*,
};

use super::{
    node::*,
    train_node::*,
    loss::LossType,
    rtree_regressor::RTreeRegressor,
};


use std::rc::Rc;
use std::cell::RefCell;


/// This struct produces a regression tree for the given distribution.
pub struct RTree {
    // The maximal depth of the output trees
    max_depth: usize,

    // The number of training instances
    n_sample: usize,


    // LossType function
    loss_type: LossType,
}


impl RTree {
    /// Initialize `RTree`.
    #[inline]
    pub fn init(data: &DataFrame, _target: &Series)
        -> Self
    {
        let n_sample = data.shape().0;

        let max_depth = (n_sample as f64).log2().ceil() as usize;

        Self {
            max_depth,
            n_sample,

            loss_type: LossType::L2,
        }
    }


    /// Set the maximum depth of the resulting tree.
    #[inline]
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }


    /// Set the loss type.
    #[inline]
    pub fn loss_type(mut self, loss: LossType) -> Self {
        self.loss_type = loss;
        self
    }
}


impl WeakLearner for RTree {
    type Hypothesis = RTreeRegressor;
    fn produce(&self, data: &DataFrame, target: &Series, dist: &[f64])
        -> Self::Hypothesis
    {
        let indices = (0..self.n_sample).into_iter()
            .filter(|&i| dist[i] > 0.0)
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
    max_depth: usize,
    loss_type: LossType
) -> Rc<RefCell<TrainNode>>
{
    let total_weight = indices.par_iter()
        .copied()
        .map(|i| dist[i])
        .sum::<f64>();


    // Compute the best prediction that minimizes the training error
    // on this node.
    let (pred, loss) = loss_type.prediction_and_loss(
        target, &indices, dist
    );


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
    if max_depth <= 1 {
        // If `depth <= 1`,
        // the childs from this node must be leaves.
        ltree = construct_leaf(target, dist, lindices, loss_type);
        rtree = construct_leaf(target, dist, rindices, loss_type);
    } else {
        // If `depth > 1`,
        // the childs from this node might be branches.
        let d = max_depth - 1;
        ltree = full_tree(data, target, dist, lindices, d, loss_type);
        rtree = full_tree(data, target, dist, rindices, d, loss_type);
    }


    TrainNode::branch(rule, ltree, rtree, pred, total_weight, loss)
}


#[inline]
fn construct_leaf(
    target: &Series,
    dist: &[f64],
    indices: Vec<usize>,
    loss_type: LossType,
) -> Rc<RefCell<TrainNode>>
{
    // Compute the best prediction that minimizes the training error
    // on this node.
    let (p, l) = loss_type.prediction_and_loss(target, &indices, dist);


    let total_weight = indices.iter()
            .copied()
            .map(|i| dist[i])
            .sum::<f64>();


    TrainNode::leaf(p, total_weight, l)
}
