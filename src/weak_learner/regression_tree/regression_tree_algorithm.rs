use crate::{Sample, WeakLearner};
use super::bin::*;


use crate::weak_learner::common::{
    split_rule::*,
};

use super::{
    node::*,
    train_node::*,
    loss::LossType,
    regression_tree_regressor::RegressionTreeRegressor,
};


use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;


/// This struct produces a regression tree for the given distribution.
pub struct RegressionTree<'a> {
    bins: HashMap<&'a str, Bins>,
    // The maximal depth of the output trees
    max_depth: usize,

    // The number of training instances
    n_sample: usize,

    // Regularization parameter
    lambda_l2: f64,


    // LossType function
    loss_type: LossType,
}


impl<'a> RegressionTree<'a> {
    #[inline]
    pub(super) fn from_components(
        bins: HashMap<&'a str, Bins>,
        n_sample: usize,
        max_depth: usize,
        lambda_l2: f64,
        loss_type: LossType,
    ) -> Self
    {
        Self { bins, n_sample, max_depth, lambda_l2, loss_type, }
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


    #[inline]
    fn full_tree(
        &self,
        sample: &Sample,
        gh: &[GradientHessian],
        indices: Vec<usize>,
        max_depth: usize,
    ) -> Rc<RefCell<TrainNode>>
    {
        // Compute the best prediction that minimizes the training error
        // on this node.
        let (pred, loss) = self.loss_type.prediction_and_loss(
            &indices, gh, self.lambda_l2,
        );


        // If sum of `dist` over `train` is zero, construct a leaf node.
        if loss == 0.0 || max_depth <= 1 {
            return TrainNode::leaf(pred, loss);
        }


        // Find the best splitting rule.
        let (feature, threshold) = self.loss_type.best_split(
            &self.bins, &sample, gh, &indices[..], self.lambda_l2,
        );

        let rule = Splitter::new(feature, threshold);


        // Split the train data for left/right childrens
        let mut lindices = Vec::new();
        let mut rindices = Vec::new();
        for i in indices.into_iter() {
            match rule.split(sample, i) {
                LR::Left  => { lindices.push(i); },
                LR::Right => { rindices.push(i); },
            }
        }


        // If the split has no meaning, construct a leaf node.
        if lindices.is_empty() || rindices.is_empty() {
            return TrainNode::leaf(pred, loss);
        }

        // -----
        // At this point, `max_depth >= 1` is guaranteed
        // so that one can grow the tree.

        // grow the tree.
        // let ltree; // Left child
        // let rtree; // Right child
        // if max_depth <= 1 {
        //     // If `depth <= 1`,
        //     // the childs from this node must be leaves.
        //     ltree = construct_leaf(target, gh, lindices, self.loss_type);
        //     rtree = construct_leaf(target, gh, rindices, self.loss_type);
        // } else {
        //     // If `depth > 1`,
        //     // the childs from this node might be branches.
        //     let d = max_depth - 1;
        //     ltree = self.full_tree(sample, gh, lindices, max_depth-1);
        //     rtree = self.full_tree(sample, gh, rindices, max_depth-1);
        // }
        let ltree = self.full_tree(sample, gh, lindices, max_depth-1);
        let rtree = self.full_tree(sample, gh, rindices, max_depth-1);


        TrainNode::branch(rule, ltree, rtree, pred, loss)
    }
}


impl<'a> WeakLearner for RegressionTree<'a> {
    type Hypothesis = RegressionTreeRegressor;
    fn produce(&self, sample: &Sample, predictions: &[f64])
        -> Self::Hypothesis
    {
        let gh = self.loss_type.gradient_and_hessian(
            sample.target(),
            predictions,
        );


        let indices = (0..self.n_sample).filter(|&i| {
                gh[i].grad != 0.0 || gh[i].hess != 0.0
            })
            .collect::<Vec<usize>>();


        let tree = self.full_tree(sample, &gh, indices, self.max_depth);


        let root = Node::from(
            Rc::try_unwrap(tree)
                .expect("Root node has reference counter >= 1")
                .into_inner()
        );

        RegressionTreeRegressor::from(root)
    }
}



impl fmt::Display for RegressionTree<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "\
            ----------\n\
            # Decision Tree Weak Learner\n\n\
            - Max depth: {}\n\
            - Loss function: {}\n\
            - Bins:\
            ",
            self.max_depth,
            self.loss_type,
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
