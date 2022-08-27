//! Defines the inner representation 
//! of the Decision Tree class.
use polars::prelude::*;
use crate::Classifier;


use super::split_rule::*;
use super::train_node::*;


use serde::{Serialize, Deserialize};

use std::rc::Rc;


/// Enumeration of `BranchNode` and `LeafNode`.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum Node {
    /// A node that have two childrens.
    Branch(BranchNode),


    /// A node that have no child.
    Leaf(LeafNode),
}


/// Represents the branch nodes of decision tree.
/// Each `BranchNode` must have two childrens
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct BranchNode {
    pub(self) rule: Splitter,
    pub(self) left: Box<Node>,
    pub(self) right: Box<Node>,
}


impl BranchNode {
    /// Returns the `BranchNode` from the given components.
    /// Note that this function does not assign the impurity.
    #[inline]
    pub(super) fn from_raw(rule: Splitter, left: Box<Node>, right: Box<Node>)
        -> Self
    {
        Self {
            rule,
            left,
            right,
        }
    }
}


/// Represents the leaf nodes of decision tree.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct LeafNode {
    pub(self) prediction: f64,
}


impl LeafNode {
    /// Returns a `LeafNode` that predicts the label
    /// given to this function.
    /// Note that this function does not assign the impurity.
    #[inline]
    pub(crate) fn from_raw(prediction: f64) -> Self {
        Self { prediction }
    }
}


impl From<TrainBranchNode> for BranchNode {
    #[inline]
    fn from(branch: TrainBranchNode) -> Self {

        let left = match Rc::try_unwrap(branch.left) {
            Ok(l) => l.into_inner().into(),
            Err(_) => panic!("Strong count is greater than 1")
        };
        let right = match Rc::try_unwrap(branch.right) {
            Ok(r) => r.into_inner().into(),
            Err(_) => panic!("Strong count is greater than 1")
        };

        Self::from_raw(
            branch.rule,
            Box::new(left),
            Box::new(right),
        )
    }
}


impl From<TrainLeafNode> for LeafNode {
    #[inline]
    fn from(leaf: TrainLeafNode) -> Self {
        Self::from_raw(leaf.prediction)
    }
}


impl From<TrainNode> for Node {
    #[inline]
    fn from(train_node: TrainNode) -> Self {
        match train_node {
            TrainNode::Branch(node) => {
                Node::Branch(node.into())
            },
            TrainNode::Leaf(node) => {
                Node::Leaf(node.into())
            }
        }
    }
}


impl Classifier for LeafNode {
    #[inline]
    fn confidence(&self, _data: &DataFrame, _row: usize) -> f64 {
        self.prediction
    }
}


impl Classifier for BranchNode {
    #[inline]
    fn confidence(&self, data: &DataFrame, row: usize) -> f64 {
        match self.rule.split(data, row) {
            LR::Left => self.left.confidence(data, row),
            LR::Right => self.right.confidence(data, row)
        }
    }
}


impl Classifier for Node {
    #[inline]
    fn confidence(&self, data: &DataFrame, row: usize) -> f64 {
        match self {
            Node::Branch(ref node) => node.confidence(data, row),
            Node::Leaf(ref node) => node.confidence(data, row)
        }
    }
}

