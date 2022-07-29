//! Defines the inner representation 
//! of the Decision Tree class.
use polars::prelude::*;
use crate::Classifier;


use super::split_rule::*;


use std::rc::Rc;
use std::cell::RefCell;

use std::ops::Add;


/// Train/Test error on a node.
#[derive(Copy, Clone, Debug)]
pub(super) struct NodeError {
    pub(super) train: f64,
    pub(super) test:  f64,
}


impl Add for NodeError {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            train: self.train + other.train,
            test:  self.test  + other.test,
        }
    }
}


impl From<(f64, f64)> for NodeError {
    #[inline]
    fn from((train, test): (f64, f64)) -> Self {
        Self { train, test }
    }
}


/// Train/Test error on a subtree.
#[derive(Copy, Clone, Debug)]
pub(super) struct TreeError {
    pub(super) train: f64,
    pub(super) test:  f64,
}


impl Add for TreeError {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            train: self.train + other.train,
            test:  self.test + other.test,
        }
    }
}


impl From<NodeError> for TreeError {
    #[inline]
    fn from(node_err: NodeError) -> Self {
        Self {
            train: node_err.train,
            test:  node_err.test,
        }
    }
}


/// Enumeration of `TrainBranchNode` and `TrainLeafNode`.
#[derive(Debug)]
pub enum TrainNode {
    /// A node that have two childrens.
    Branch(TrainBranchNode),


    /// A node that have no child.
    Leaf(TrainLeafNode),
}


/// Represents the branch nodes of decision tree.
/// Each `TrainBranchNode` must have two childrens
#[derive(Debug)]
pub struct TrainBranchNode {
    pub(super) rule: Splitter,
    pub(super) left:  Rc<RefCell<TrainNode>>,
    pub(super) right: Rc<RefCell<TrainNode>>,

    // Common members
    pub(super) prediction: i64,
    pub(self) total_weight: f64, 
    pub(self) node_err: NodeError,
    pub(self) tree_err: TreeError,
    pub(self) leaves:   usize,
}


impl TrainBranchNode {
    /// Returns the `TrainBranchNode` from the given components.
    /// Note that this function does not assign the impurity.
    #[inline]
    pub(super) fn from_raw(rule: Splitter,
                           left:  Rc<RefCell<TrainNode>>,
                           right: Rc<RefCell<TrainNode>>,
                           prediction: i64,
                           total_weight: f64,
                           node_err: NodeError)
        -> Self
    {
        let tree_err = left.borrow().tree_error()
            + right.borrow().tree_error();
        let leaves = left.borrow().leaves() + right.borrow().leaves();


        Self {
            rule,
            left,
            right,

            prediction,
            total_weight,
            node_err,
            tree_err,
            leaves
        }
    }


    /// Convert `self` to the components that are used for
    /// the construction of a leaf.
    #[inline]
    pub(self) fn into_leaf_component(self)
        -> (i64, f64, NodeError)
    {
        (self.prediction, self.total_weight, self.node_err)
    }
}


/// Represents the leaf nodes of decision tree.
#[derive(Debug)]
pub struct TrainLeafNode {
    pub(super) prediction: i64,
    pub(self) total_weight: f64,
    pub(self) node_err: NodeError,
}


impl TrainLeafNode {
    /// Returns a `TrainLeafNode` that predicts the label
    /// given to this function.
    /// Note that this function does not assign the impurity.
    #[inline]
    pub(super) fn from_raw(prediction: i64,
                           total_weight: f64,
                           node_err: NodeError)
        -> Self
    {
        Self {
            prediction,
            total_weight,
            node_err,
        }
    }
}


impl From<TrainBranchNode> for TrainLeafNode {
    #[inline]
    fn from(branch: TrainBranchNode) -> TrainLeafNode {
        let (p, total_weight, node_err) = branch.into_leaf_component();
        TrainLeafNode::from_raw(p, total_weight, node_err)
    }
}


impl TrainNode {
    /// Construct a leaf node from the given arguments.
    #[inline]
    pub(super) fn leaf(prediction: i64,
                       total_weight: f64,
                       node_err: NodeError)
        -> Self
    {
        let leaf = TrainLeafNode::from_raw(
            prediction, total_weight, node_err
        );
        TrainNode::Leaf(leaf)
    }


    /// Construct a branch node from the arguments.
    #[inline]
    pub(super) fn branch(rule: Splitter,
                         left: Rc<RefCell<TrainNode>>,
                         right: Rc<RefCell<TrainNode>>,
                         prediction: i64,
                         total_weight: f64,
                         node_err: NodeError)
        -> Self
    {
        let node = TrainBranchNode::from_raw(
            rule, left, right, prediction, total_weight, node_err,
        );


        TrainNode::Branch(node)
    }


    #[inline]
    pub(super) fn train_node_error(&self) -> f64 {
        match self {
            TrainNode::Branch(ref branch) => {
                branch.node_err.train * branch.total_weight
            }
            TrainNode::Leaf(ref leaf) => {
                leaf.node_err.train * leaf.total_weight
            }
        }
    }


    #[inline]
    pub(super) fn node_error(&self) -> NodeError {
        match self {
            TrainNode::Branch(ref branch) => branch.node_err,
            TrainNode::Leaf(ref leaf) => leaf.node_err,
        }
    }


    /// TODO fix tree error
    #[inline]
    pub(super) fn tree_error(&self) -> TreeError {
        match self {
            TrainNode::Branch(ref branch) => branch.tree_err,
            TrainNode::Leaf(ref leaf) => leaf.node_err.into(),
        }
    }


    pub(super) fn set_tree_error_train(&mut self) -> f64 {
        match self {
            TrainNode::Branch(branch) => {
                let l = branch.left.borrow_mut()
                    .set_tree_error_train();
                let r = branch.right.borrow_mut()
                    .set_tree_error_train();
                branch.tree_err.train = l + r;
                branch.tree_err.train
            },
            TrainNode::Leaf(leaf) => {
                leaf.total_weight * leaf.node_err.train
            },
        }
    }


    #[inline]
    pub(super) fn alpha(&self) -> f64 {
        match self {
            TrainNode::Branch(_) => {
                let node_err = self.node_error();
                let tree_err = self.tree_error();
                let leaves   = self.leaves() as f64;

                // DEBUG
                assert!(leaves > 1.0);
                assert!(node_err.train.is_finite());
                assert!(tree_err.train.is_finite());
                (node_err.train - tree_err.train) / (leaves - 1.0)
            },
            TrainNode::Leaf(_) => f64::MAX
        }
    }


    /// Returns the number of leaves of this sub-tree.
    #[inline]
    pub(super) fn leaves(&self) -> usize {
        match self {
            TrainNode::Branch(ref node) => node.leaves,
            TrainNode::Leaf(_) => 1_usize
        }
    }


    /// Execute preprocessing before the pruning.
    /// This method removes the leaves that do not affect
    /// the training error.
    #[inline]
    pub(super) fn pre_process(&mut self) {

        if let TrainNode::Branch(ref mut branch) = self {
            let t = branch.node_err.train * branch.total_weight;
            let l = branch.left.borrow().train_node_error();
            let r = branch.right.borrow().train_node_error();


            if t == l + r {
                self.prune();
            } else {
                branch.left.borrow_mut().pre_process();
                branch.right.borrow_mut().pre_process();
            }
        }
    }


    /// After `pre_process()`, the total number of leaves changes.
    /// This method modifies the leaves for all nodes.
    #[inline]
    pub(super) fn reassign_leaves(&mut self) -> usize {
        match self {
            TrainNode::Branch(branch) => {
                let l = branch.left.borrow_mut()
                    .reassign_leaves();
                let r = branch.right.borrow_mut()
                    .reassign_leaves();
                branch.leaves = l + r;
                branch.leaves
            },
            TrainNode::Leaf(_) => 1_usize,
        }
    }


    #[inline]
    pub(super) fn prune(&mut self) {
        if let TrainNode::Branch(ref mut branch) = self {
            *self = TrainNode::leaf(
                branch.prediction,
                branch.total_weight,
                branch.node_err,
            );
        }
    }
}


impl Classifier for TrainLeafNode {
    #[inline]
    fn predict(&self, _data: &DataFrame, _row: usize) -> i64 {
        self.prediction
    }
}


impl Classifier for TrainBranchNode {
    #[inline]
    fn predict(&self, data: &DataFrame, row: usize) -> i64 {
        match self.rule.split(data, row) {
            LR::Left => self.left.borrow().predict(data, row),
            LR::Right => self.right.borrow().predict(data, row)
        }
    }
}


impl Classifier for TrainNode {
    #[inline]
    fn predict(&self, data: &DataFrame, row: usize) -> i64 {
        match self {
            TrainNode::Branch(ref node) => node.predict(data, row),
            TrainNode::Leaf(ref node) => node.predict(data, row)
        }
    }
}

