//! Defines the inner representation 
//! of the Decision Tree class.
use crate::{Data, Label};
use crate::Classifier;


use super::split_rule::*;


use serde::{Serialize, Deserialize};


/// Maximization objectives.
/// * `Criterion::Gini` is the gini-index,
/// * `Criterion::Entropy` is the entropy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Criterion {
    /// Gini-index.
    Gini,
    /// Binary entropy function.
    Entropy,
}


/// Enumeration of `BranchNode` and `LeafNode`.
#[derive(Debug, Serialize, Deserialize)]
pub enum Node<S> {
    /// A node that have two childrens.
    Branch(BranchNode<S>),
    /// A node that have no child.
    Leaf(LeafNode),
}


/// Represents the branch nodes of decision tree.
/// Each `BranchNode` must have two childrens
#[derive(Debug, Serialize, Deserialize)]
pub struct BranchNode<S> {
    pub(self) split_rule: S,
    pub(self) left_node:  Box<Node<S>>,
    pub(self) right_node: Box<Node<S>>,

    // Common members
    pub(self) impurity:   f64,
    pub(self) leaves:     usize,
}


impl<S> BranchNode<S> {
    /// Returns the `BranchNode` from the given components.
    /// Note that this function does not assign the impurity.
    #[inline]
    pub(crate) fn from_raw(split_rule: S,
                           left_node:  Box<Node<S>>,
                           right_node: Box<Node<S>>,
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
pub struct LeafNode {
    pub(self) prediction: Label,
    pub(self) impurity:   f64,
    pub(self) leaves:     usize,
}


impl LeafNode {
    /// Returns a `LeafNode` that predicts the label
    /// given to this function.
    /// Note that this function does not assign the impurity.
    #[inline]
    pub(crate) fn from_raw(prediction: Label, impurity: f64) -> Self {
        let leaves = 1_usize;
        Self {
            prediction,

            impurity,
            leaves
        }
    }
}


impl<S> Node<S> {
    /// Construct a leaf node that predicts `label`.
    pub(crate) fn leaf(label: Label, impurity: f64) -> Self {
        let node = LeafNode::from_raw(label, impurity);

        Node::Leaf(node)
    }


    pub(crate) fn branch(rule: S,
                         left:  Box<Node<S>>,
                         right: Box<Node<S>>,
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


    pub(crate) fn leaves(&self) -> usize {
        match self {
            Node::Branch(ref node) => node.leaves,
            Node::Leaf(ref node)   => node.leaves
        }
    }
}


impl<D> Classifier<D> for LeafNode
    where D: Data,
{
    #[inline]
    fn predict(&self, _data: &D) -> Label {
        self.prediction
    }
}


impl<S, D, O> Classifier<D> for BranchNode<S>
    where S: SplitRule<Input = D>,
          D: Data<Output = O>,
          O: PartialOrd,
{
    #[inline]
    fn predict(&self, data: &D) -> Label {
        match self.split_rule.split(data) {
            LR::Left  => self.left_node.predict(data),
            LR::Right => self.right_node.predict(data)
        }
    }
}


impl<S, D, O> Classifier<D> for Node<S>
    where S: SplitRule<Input = D>,
          D: Data<Output = O>,
          O: PartialOrd,
{
    #[inline]
    fn predict(&self, data: &D) -> Label {
        match self {
            Node::Branch(ref node) => node.predict(data),
            Node::Leaf(ref node)   => node.predict(data)
        }
    }
}

// impl<S, O, D> Classifier<D> for Node<S>
//     where S: SplitRule<Input = D>,
//           D: Data<Output = O>,
//           O: PartialOrd,
// {
//     #[inline]
//     fn predict(&self, data: &D) -> Label {
//         match self.split_rule {
//             None => {
//                 self.prediction
//             },
//             Some(ref rule) => {
//                 match rule.split(data) {
//                     LR::Left  => {
//                         self.left_node
//                             .as_ref()
//                             .unwrap()
//                             .predict(data)
//                     },
//                     LR::Right => {
//                         self.right_node
//                             .as_ref()
//                             .unwrap()
//                             .predict(data)
//                     }
//                 }
//             }
//         }
//     }
// }



