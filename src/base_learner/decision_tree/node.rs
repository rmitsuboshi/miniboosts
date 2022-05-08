//! Defines the inner representation 
//! of the Decision Tree class.
use crate::Data;
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
#[derive(Debug, Serialize, Deserialize)]
pub enum Node<O, L> {
    /// A node that have two childrens.
    Branch(BranchNode<O, L>),


    /// A node that have no child.
    Leaf(LeafNode<L>),
}


/// Represents the branch nodes of decision tree.
/// Each `BranchNode` must have two childrens
#[derive(Debug, Serialize, Deserialize)]
pub struct BranchNode<O, L> {
    pub(self) split_rule: SplitRule<O>,
    pub(self) left_node:  Box<Node<O, L>>,
    pub(self) right_node: Box<Node<O, L>>,
}


impl<O, L> BranchNode<O, L> {
    /// Returns the `BranchNode` from the given components.
    /// Note that this function does not assign the impurity.
    #[inline]
    pub(crate) fn from_raw(split_rule: SplitRule<O>,
                           left_node:  Box<Node<O, L>>,
                           right_node: Box<Node<O, L>>)
        -> Self
    {
        Self {
            split_rule,
            left_node,
            right_node,
        }
    }
}


/// Represents the leaf nodes of decision tree.
#[derive(Debug, Serialize, Deserialize)]
pub struct LeafNode<L> {
    pub(self) prediction: L,
}


impl<L> LeafNode<L> {
    /// Returns a `LeafNode` that predicts the label
    /// given to this function.
    /// Note that this function does not assign the impurity.
    #[inline]
    pub(crate) fn from_raw(prediction: L) -> Self {
        Self { prediction }
    }
}


impl<O, L> From<TrainBranchNode<O, L>> for BranchNode<O, L> {
    #[inline]
    fn from(branch: TrainBranchNode<O, L>) -> Self {

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


impl<L> From<TrainLeafNode<L>> for LeafNode<L> {
    #[inline]
    fn from(leaf: TrainLeafNode<L>) -> Self {
        Self::from_raw(leaf.prediction)
    }
}


impl<O, L> From<TrainNode<O, L>> for Node<O, L> {
    #[inline]
    fn from(train_node: TrainNode<O, L>) -> Self {
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


impl<D, L> Classifier<D, L> for LeafNode<L>
    where D: Data,
          L: PartialEq + Clone,
{
    #[inline]
    fn predict(&self, _data: &D) -> L {
        self.prediction.clone()
    }
}


impl<D, L, O> Classifier<D, L> for BranchNode<O, L>
    where D: Data<Output = O>,
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


impl<D, L, O> Classifier<D, L> for Node<O, L>
    where D: Data<Output = O>,
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

