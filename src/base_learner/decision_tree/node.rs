//! Defines the inner representation 
//! of the Decision Tree class.
use polars::prelude::*;
use crate::Classifier;


use super::split_rule::*;
use super::train_node::*;


use serde::{Serialize, Deserialize};

use std::rc::Rc;
use std::cmp::Ordering;
use std::ops::{Mul, Add};



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
    pub(self) split_rule: SplitRule,
    pub(self) left: Box<Node>,
    pub(self) right: Box<Node>,
}


impl BranchNode {
    /// Returns the `BranchNode` from the given components.
    /// Note that this function does not assign the impurity.
    #[inline]
    pub(crate) fn from_raw(split_rule: SplitRule,
                           left: Box<Node>,
                           right: Box<Node>)
        -> Self
    {
        Self {
            split_rule,
            left,
            right,
        }
    }
}


/// Represents the leaf nodes of decision tree.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct LeafNode {
    pub(self) prediction: i64,
}


impl LeafNode {
    /// Returns a `LeafNode` that predicts the label
    /// given to this function.
    /// Note that this function does not assign the impurity.
    #[inline]
    pub(crate) fn from_raw(prediction: i64) -> Self {
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
            branch.split_rule,
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
    fn predict(&self, _data: &DataFrame, _row: usize) -> i64 {
        self.prediction
    }
}


impl Classifier for BranchNode {
    #[inline]
    fn predict(&self, data: &DataFrame, row: usize) -> i64 {
        match self.split_rule.split(data, row) {
            LR::Left => self.left.predict(data, row),
            LR::Right => self.right.predict(data, row)
        }
    }
}


impl Classifier for Node {
    #[inline]
    fn predict(&self, data: &DataFrame, row: usize) -> i64 {
        match self {
            Node::Branch(ref node) => node.predict(data, row),
            Node::Leaf(ref node) => node.predict(data, row)
        }
    }
}

