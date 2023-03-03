//! Defines the inner representation 
//! of the Decision Tree class.
use crate::{Classifier, Sample};


use crate::weak_learner::common::{
    type_and_struct::*,
    split_rule::*,
};
use super::train_node::*;


use serde::{Serialize, Deserialize};

use std::rc::Rc;


/// Enumeration of `BranchNode` and `LeafNode`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Node {
    /// A node that have two childrens.
    Branch(BranchNode),


    /// A node that have no child.
    Leaf(LeafNode),
}


/// Represents the branch nodes of decision tree.
/// Each `BranchNode` must have two childrens
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BranchNode {
    pub(super) rule: Splitter,
    pub(super) left: Box<Node>,
    pub(super) right: Box<Node>,
}


impl BranchNode {
    /// Returns the `BranchNode` from the given components.
    /// Note that this function does not assign the impurity.
    #[inline]
    pub(super) fn from_raw(
        rule: Splitter,
        left: Box<Node>,
        right: Box<Node>
    ) -> Self
    {
        Self { rule, left, right, }
    }
}


/// Represents the leaf nodes of decision tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LeafNode {
    pub(super) confidence: Confidence<f64>,
}


impl LeafNode {
    /// Returns a `LeafNode` that predicts the label
    /// given to this function.
    /// Note that this function does not assign the impurity.
    #[inline]
    pub(crate) fn from_raw(confidence: Confidence<f64>) -> Self {
        Self { confidence }
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
        Self::from_raw(leaf.confidence)
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
    fn confidence(&self, _sample: &Sample, _row: usize) -> f64 {
        self.confidence.0
    }
}


impl Classifier for BranchNode {
    #[inline]
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        match self.rule.split(sample, row) {
            LR::Left => self.left.confidence(sample, row),
            LR::Right => self.right.confidence(sample, row)
        }
    }
}


impl Classifier for Node {
    #[inline]
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        match self {
            Node::Branch(ref node) => node.confidence(sample, row),
            Node::Leaf(ref node) => node.confidence(sample, row)
        }
    }
}


impl Node {
    pub(super) fn to_dot_info(&self, id: usize) -> (Vec<String>, usize) {
        match self {
            Node::Branch(b) => {
                let b_info = format!(
                    "\tnode_{id} [ label = \"{feat} < {thr:.2} ?\" ];\n",
                    feat = b.rule.feature,
                    thr = b.rule.threshold.0
                );

                let (l_info, next_id) = b.left.to_dot_info(id + 1);
                let (mut r_info, ret_id) = b.right.to_dot_info(next_id);

                let mut info = l_info;
                info.push(b_info);
                info.append(&mut r_info);

                let l_edge = format!(
                    "\tnode_{id} -- node_{l_id} [ label = \"Yes\" ];\n",
                    l_id = id + 1
                );
                let r_edge = format!(
                    "\tnode_{id} -- node_{r_id} [ label = \"No\" ];\n",
                    r_id = next_id
                );

                info.push(l_edge);
                info.push(r_edge);

                (info, ret_id)
            },
            Node::Leaf(l) => {
                let info = format!(
                    "\tnode_{id} [ \
                     label = \"{p}\", \
                     shape = box, \
                     ];\n",
                    p = l.confidence.0
                );

                (vec![info], id + 1)
            }
        }
    }
}
