//! Defines the inner representation 
//! of the Decision Tree class.
use crate::Data;
use crate::Classifier;


use super::split_rule::*;


use serde::{Serialize, Deserialize};
use std::rc::Rc;

use std::cmp::Ordering;
use std::ops::{Mul, Add};


/// Error on a node.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub(crate) struct NodeError(f64);


impl From<f64> for NodeError {
    #[inline(always)]
    fn from(node_err: f64) -> Self {
        NodeError(node_err)
    }
}


/// Error on a sub-tree.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub(crate) struct TreeError(f64);


impl From<f64> for TreeError {
    #[inline(always)]
    fn from(tree_err: f64) -> Self {
        TreeError(tree_err)
    }
}


impl Add for TreeError {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self::Output {
        Self(self.0 + other.0)
    }
}


/// Impurity
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Impurity(f64);


impl From<f64> for Impurity {
    #[inline(always)]
    fn from(impurity: f64) -> Self {
        Impurity(impurity)
    }
}


impl PartialEq for Impurity {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}


impl PartialOrd for Impurity {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}


impl Mul for Impurity {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self::Output {
        Self(self.0 * other.0)
    }
}


impl Add for Impurity {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self::Output {
        Self(self.0 + other.0)
    }
}


// TODO
//      Add other criterions.
//      E.g., Gini criterion, Twoing criterion (page 38 of CART)
/// Maximization objectives.
/// * `Criterion::Gini` is the gini-index,
/// * `Criterion::Entropy` is the entropy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Criterion {
    /// Binary entropy function.
    Entropy,
}


/// Enumeration of `BranchNode` and `LeafNode`.
#[derive(Debug, Serialize, Deserialize)]
pub enum Node<S, L> {
    /// A node that have two childrens.
    Branch(BranchNode<S, L>),
    /// A node that have no child.
    Leaf(LeafNode<L>),
}


/// Represents the branch nodes of decision tree.
/// Each `BranchNode` must have two childrens
#[derive(Debug, Serialize, Deserialize)]
pub struct BranchNode<S, L> {
    pub(self) split_rule: S,
    pub(self) left_node:  Rc<Node<S, L>>,
    pub(self) right_node: Rc<Node<S, L>>,

    // Common members
    pub(self) prediction: L,
    pub(self) node_err:   NodeError,
    pub(self) tree_err:   TreeError,
    pub(self) impurity:   Impurity,
    pub(self) leaves:     usize,
}


impl<S, L> BranchNode<S, L> {
    /// Returns the `BranchNode` from the given components.
    /// Note that this function does not assign the impurity.
    #[inline]
    pub(crate) fn from_raw(split_rule: S,
                           left_node:  Rc<Node<S, L>>,
                           right_node: Rc<Node<S, L>>,
                           prediction: L,
                           node_err:   NodeError,
                           impurity:   Impurity)
        -> Self
    {
        let tree_err = left_node.tree_error() + right_node.tree_error();
        let tree_err = TreeError::from(tree_err);
        let leaves = left_node.leaves() + right_node.leaves();


        Self {
            split_rule,
            left_node,
            right_node,

            prediction,
            node_err,
            tree_err,
            impurity,
            leaves
        }
    }


    /// Convert `self` to the components that are used for
    /// the construction of a leaf.
    #[inline]
    pub(self) fn into_leaf_component(self)
        -> (L, NodeError, Impurity)
    {
        (self.prediction, self.node_err, self.impurity)
    }
}


/// Represents the leaf nodes of decision tree.
#[derive(Debug, Serialize, Deserialize)]
pub struct LeafNode<L> {
    pub(self) prediction: L,
    pub(self) node_err:   NodeError,
    pub(self) impurity:   Impurity,
}


impl<L> LeafNode<L> {
    /// Returns a `LeafNode` that predicts the label
    /// given to this function.
    /// Note that this function does not assign the impurity.
    #[inline]
    pub(crate) fn from_raw(prediction: L,
                           node_err:   NodeError,
                           impurity:   Impurity)
        -> Self
    {
        Self {
            prediction,
            node_err,
            impurity,
        }
    }
}


impl<S, L> From<BranchNode<S, L>> for LeafNode<L> {
    #[inline]
    fn from(branch: BranchNode<S, L>) -> LeafNode<L> {
        let (p, node_err, impurity) = branch.into_leaf_component();
        LeafNode::from_raw(p, node_err, impurity)
    }
}


impl<S, L> Node<S, L> {
    /// Construct a leaf node from the given arguments.
    #[inline]
    pub(crate) fn leaf(prediction: L,
                       node_err:   NodeError,
                       impurity:   Impurity)
        -> Self
    {
        let leaf = LeafNode::from_raw(prediction, node_err, impurity);
        Node::Leaf(leaf)
    }


    /// Construct a branch node from the arguments.
    #[inline]
    pub(crate) fn branch(rule: S,
                         left:  Rc<Node<S, L>>,
                         right: Rc<Node<S, L>>,
                         prediction: L,
                         node_err:  NodeError,
                         impurity:  Impurity)
        -> Self
    {
        let node = BranchNode::from_raw(
            rule,
            left,
            right,
            prediction,
            node_err,
            impurity
        );


        Node::Branch(node)
    }


    #[inline]
    pub(crate) fn node_error(&self) -> f64 {
        match self {
            Node::Branch(ref branch) => branch.node_err.0,
            Node::Leaf(ref leaf) => leaf.node_err.0,
        }
    }


    #[inline]
    pub(crate) fn tree_error(&self) -> f64 {
        match self {
            Node::Branch(ref branch) => branch.tree_err.0,
            Node::Leaf(ref leaf) => leaf.node_err.0,
        }
    }


    /// Returns the number of leaves of this sub-tree.
    #[inline]
    pub(crate) fn leaves(&self) -> usize {
        match self {
            Node::Branch(ref node) => node.leaves,
            Node::Leaf(_) => 1_usize
        }
    }



    /// Execute preprocessing before the pruning.
    /// This method removes the leaves that do not affect
    /// the training error.
    #[inline]
    pub(super) fn pre_process(&mut self)
        where L: Clone
    {

        if let Node::Branch(ref mut branch) = self {
            let left  = branch.left_node.node_error();
            let right = branch.right_node.node_error();


            if branch.node_err.0 == left + right {
                *self = Node::leaf(
                    branch.prediction.clone(),
                    branch.node_err,
                    branch.impurity
                );
            } else {
                // branch.left_node.pre_process();
                // branch.right_node.pre_process();
                Rc::get_mut(&mut branch.left_node)
                    .unwrap()
                    .pre_process();
                Rc::get_mut(&mut branch.right_node)
                    .unwrap()
                    .pre_process();
            }
        }
    }
}


impl<D, L> Classifier<D, L> for LeafNode<L>
    where D: Data,
          L: PartialEq + Clone,
{
    #[inline]
    fn predict(&self, _data: &D) -> L {
        self.prediction.clone()
    }
}


impl<S, D, L, O> Classifier<D, L> for BranchNode<S, L>
    where S: SplitRule<D>,
    // where S: SplitRule<Input = D>,
          D: Data<Output = O>,
          L: PartialEq + Clone,
          O: PartialOrd,
{
    #[inline]
    fn predict(&self, data: &D) -> L {
        match self.split_rule.split(data) {
            LR::Left  => self.left_node.predict(data),
            LR::Right => self.right_node.predict(data)
        }
    }
}


impl<S, D, L, O> Classifier<D, L> for Node<S, L>
    // where S: SplitRule<Input = D>,
    where S: SplitRule<D>,
          D: Data<Output = O>,
          L: PartialEq + Clone,
          O: PartialOrd,
{
    #[inline]
    fn predict(&self, data: &D) -> L {
        match self {
            Node::Branch(ref node) => node.predict(data),
            Node::Leaf(ref node)   => node.predict(data)
        }
    }
}

