//! Defines the inner representation 
//! of the Decision Tree class.
use crate::Data;
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
            test:  self.test + other.test,
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
pub enum TrainNode<O, L> {
    /// A node that have two childrens.
    Branch(TrainBranchNode<O, L>),


    /// A node that have no child.
    Leaf(TrainLeafNode<L>),
}


/// Represents the branch nodes of decision tree.
/// Each `TrainBranchNode` must have two childrens
#[derive(Debug)]
pub struct TrainBranchNode<O, L> {
    pub(super) split_rule: SplitRule<O>,
    pub(super) left:  Rc<RefCell<TrainNode<O, L>>>,
    pub(super) right: Rc<RefCell<TrainNode<O, L>>>,

    // Common members
    pub(super) prediction: L,
    pub(self) node_err: NodeError,
    pub(self) tree_err: TreeError,
    pub(self) leaves:   usize,
}


impl<O, L> TrainBranchNode<O, L> {
    /// Returns the `TrainBranchNode` from the given components.
    /// Note that this function does not assign the impurity.
    #[inline]
    pub(super) fn from_raw(split_rule: SplitRule<O>,
                           left:  Rc<RefCell<TrainNode<O, L>>>,
                           right: Rc<RefCell<TrainNode<O, L>>>,
                           prediction: L,
                           node_err: NodeError)
        -> Self
    {
        let tree_err = left.borrow().tree_error()
            + right.borrow().tree_error();
        let leaves = left.borrow().leaves() + right.borrow().leaves();


        if !tree_err.train.is_finite() {
            println!("tree error: {}", tree_err.train);
            let l = left.borrow().leaves() == 1;
            let r = right.borrow().leaves() == 1;
            println!(
                "Left: {}, right: {}",
                if l { "leaf" } else { "branch" },
                if r { "leaf" } else { "branch" },
            );

            println!(
                "left.tree_err.train: {}, right.tree_err.train: {}",
                left.borrow().tree_error().train,
                right.borrow().tree_error().train,
            );

            assert!(tree_err.train.is_finite());

        }


        Self {
            split_rule,
            left,
            right,

            prediction,
            node_err,
            tree_err,
            leaves
        }
    }


    /// Convert `self` to the components that are used for
    /// the construction of a leaf.
    #[inline]
    pub(self) fn into_leaf_component(self)
        -> (L, NodeError)
    {
        (self.prediction, self.node_err)
    }
}


/// Represents the leaf nodes of decision tree.
#[derive(Debug)]
pub struct TrainLeafNode<L> {
    pub(super) prediction: L,
    pub(self) node_err: NodeError,
}


impl<L> TrainLeafNode<L> {
    /// Returns a `TrainLeafNode` that predicts the label
    /// given to this function.
    /// Note that this function does not assign the impurity.
    #[inline]
    pub(super) fn from_raw(prediction: L,
                           node_err: NodeError)
        -> Self
    {
        Self {
            prediction,
            node_err,
        }
    }
}


impl<O, L> From<TrainBranchNode<O, L>> for TrainLeafNode<L> {
    #[inline]
    fn from(branch: TrainBranchNode<O, L>) -> TrainLeafNode<L> {
        let (p, node_err) = branch.into_leaf_component();
        TrainLeafNode::from_raw(p, node_err)
    }
}


impl<O, L> TrainNode<O, L> {
    /// Construct a leaf node from the given arguments.
    #[inline]
    pub(super) fn leaf(prediction: L, node_err: NodeError)
        -> Self
    {
        let leaf = TrainLeafNode::from_raw(
            prediction, node_err
        );
        TrainNode::Leaf(leaf)
    }


    /// Construct a branch node from the arguments.
    #[inline]
    pub(super) fn branch(rule: SplitRule<O>,
                         left:  Rc<RefCell<TrainNode<O, L>>>,
                         right: Rc<RefCell<TrainNode<O, L>>>,
                         prediction: L,
                         node_err: NodeError)
        -> Self
    {
        let node = TrainBranchNode::from_raw(
            rule,
            left,
            right,
            prediction,
            node_err,
        );


        TrainNode::Branch(node)
    }


    #[inline]
    pub(super) fn node_error(&self) -> NodeError {
        match self {
            TrainNode::Branch(ref branch) => branch.node_err,
            TrainNode::Leaf(ref leaf) => leaf.node_err,
        }
    }


    #[inline]
    pub(super) fn tree_error(&self) -> TreeError {
        match self {
            TrainNode::Branch(ref branch) => branch.tree_err,
            TrainNode::Leaf(ref leaf) => leaf.node_err.into(),
        }
    }


    #[inline]
    pub(super) fn alpha(&self) -> f64 {
        match self {
            TrainNode::Branch(_) => {
                let node_err = self.node_error();
                let tree_err = self.tree_error();
                let leaves   = self.leaves() as f64;
                assert!(node_err.train.is_finite());
                assert!(tree_err.train.is_finite());
                (node_err.train - tree_err.train) / (leaves - 1.0)
            },
            TrainNode::Leaf(_) => {
                f64::INFINITY
            }
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
    pub(super) fn pre_process(&mut self)
        where L: Clone
    {

        if let TrainNode::Branch(ref mut branch) = self {
            let left  = branch.left.borrow().node_error();
            let right = branch.right.borrow().node_error();


            if branch.node_err.train == left.train + right.train {
                *self = TrainNode::leaf(
                    branch.prediction.clone(),
                    branch.node_err,
                );
            } else {
                branch.left.borrow_mut().pre_process();
                branch.right.borrow_mut().pre_process();
            }
        }
    }


    #[inline]
    pub(super) fn prune(&mut self)
        where L: Clone
    {
        if let TrainNode::Branch(ref mut branch) = self {
            *self = TrainNode::leaf(
                branch.prediction.clone(),
                branch.node_err,
            );
        }
    }
}


impl<D, L> Classifier<D, L> for TrainLeafNode<L>
    where D: Data,
          L: PartialEq + Clone,
{
    #[inline]
    fn predict(&self, _data: &D) -> L {
        self.prediction.clone()
    }
}


impl<D, L, O> Classifier<D, L> for TrainBranchNode<O, L>
    where D: Data<Output = O>,
          L: PartialEq + Clone,
          O: PartialOrd,
{
    #[inline]
    fn predict(&self, data: &D) -> L {
        match self.split_rule.split(data) {
            LR::Left  => self.left.borrow().predict(data),
            LR::Right => self.right.borrow().predict(data)
        }
    }
}


impl<D, L, O> Classifier<D, L> for TrainNode<O, L>
    where D: Data<Output = O>,
          L: PartialEq + Clone,
          O: PartialOrd,
{
    #[inline]
    fn predict(&self, data: &D) -> L {
        match self {
            TrainNode::Branch(ref node) => node.predict(data),
            TrainNode::Leaf(ref node)   => node.predict(data)
        }
    }
}

