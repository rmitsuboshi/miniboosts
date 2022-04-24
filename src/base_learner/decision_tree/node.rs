//! Defines the inner representation 
//! of the Decision Tree class.
use crate::Data;
use crate::Classifier;


use super::split_rule::*;


use serde::{Serialize, Deserialize};


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
    pub(self) left_node:  Box<Node<S, L>>,
    pub(self) right_node: Box<Node<S, L>>,

    // Common members
    pub(self) impurity:   f64,
    pub(self) leaves:     usize,
}


impl<S, L> BranchNode<S, L> {
    /// Returns the `BranchNode` from the given components.
    /// Note that this function does not assign the impurity.
    #[inline]
    pub(crate) fn from_raw(split_rule: S,
                           left_node:  Box<Node<S, L>>,
                           right_node: Box<Node<S, L>>,
                           impurity:   f64)
        -> Self
    {
        let leaves = left_node.leaves() + right_node.leaves();


        Self {
            split_rule,
            left_node,
            right_node,

            impurity,
            leaves
        }
    }
}


/// Represents the leaf nodes of decision tree.
#[derive(Debug, Serialize, Deserialize)]
pub struct LeafNode<L> {
    pub(self) prediction: L,
    pub(self) impurity:   f64,
    pub(self) leaves:     usize,
}


impl<L> LeafNode<L> {
    /// Returns a `LeafNode` that predicts the label
    /// given to this function.
    /// Note that this function does not assign the impurity.
    #[inline]
    pub(crate) fn from_raw(prediction: L, impurity: f64) -> Self {
        let leaves = 1_usize;
        Self {
            prediction,

            impurity,
            leaves
        }
    }
}


impl<S, L> Node<S, L> {
    /// Construct a leaf node that predicts `label`.
    pub(crate) fn leaf(label: L, impurity: f64) -> Self {
        let node = LeafNode::from_raw(label, impurity);

        Node::Leaf(node)
    }


    /// Construct a branch node from the arguments.
    pub(crate) fn branch(rule: S,
                         left:  Box<Node<S, L>>,
                         right: Box<Node<S, L>>,
                         impurity:   f64)
        -> Self
    {
        let node = BranchNode::from_raw(
            rule,
            left,
            right,
            impurity
        );


        Node::Branch(node)
    }


    /// Returns the number of leaves of this sub-tree.
    pub(crate) fn leaves(&self) -> usize {
        match self {
            Node::Branch(ref node) => node.leaves,
            Node::Leaf(ref node)   => node.leaves
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

