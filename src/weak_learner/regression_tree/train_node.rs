//! Defines the inner representation 
//! of the Decision Tree class.
use crate::Regressor;


use crate::weak_learner::common::{
    type_and_struct::*,
    split_rule::*,
};
use crate::Sample;


use std::rc::Rc;
use std::cell::RefCell;

use std::fmt;


/// Enumeration of `TrainBranchNode` and `TrainLeafNode`.
pub enum TrainNode {
    /// A node that have two childrens.
    Branch(TrainBranchNode),


    /// A node that have no child.
    Leaf(TrainLeafNode),
}


/// Represents the branch nodes of decision tree.
/// Each `TrainBranchNode` must have two childrens
pub struct TrainBranchNode {
    // Splitting rule
    pub(super) rule: Splitter,


    // Left child
    pub(super) left: Rc<RefCell<TrainNode>>,


    // Right child
    pub(super) right: Rc<RefCell<TrainNode>>,


    // A label that have most weight on this node.
    pub(super) prediction: Prediction<f64>,


    // Training error as a leaf
    pub(self) loss_as_leaf: LossValue,


    pub(self) leaves: usize,
}


/// Represents the leaf nodes of decision tree.
pub struct TrainLeafNode {
    pub(super) prediction: Prediction<f64>,
    pub(self) loss_as_leaf: LossValue,
}


impl From<TrainBranchNode> for TrainLeafNode {
    #[inline]
    fn from(branch: TrainBranchNode) -> Self {
        Self {
            prediction: branch.prediction,
            loss_as_leaf: branch.loss_as_leaf,
        }
    }
}


impl TrainNode {
    /// Construct a leaf node from the given arguments.
    #[inline]
    pub(super) fn leaf(
        prediction: Prediction<f64>,
        loss_as_leaf: LossValue,
    ) -> Rc<RefCell<Self>>
    {
        let leaf = TrainLeafNode {
            prediction,
            loss_as_leaf,
        };


        Rc::new(RefCell::new(TrainNode::Leaf(leaf)))
    }


    /// Construct a branch node from the arguments.
    #[inline]
    pub(super) fn branch(
        rule: Splitter,
        left: Rc<RefCell<TrainNode>>,
        right: Rc<RefCell<TrainNode>>,
        prediction: Prediction<f64>,
        loss_as_leaf: LossValue,
    ) -> Rc<RefCell<Self>>
    {
        let leaves = left.borrow().leaves() + right.borrow().leaves();
        let node = TrainBranchNode {
            rule,
            left,
            right,

            prediction,
            loss_as_leaf,


            leaves,
        };

        Rc::new(RefCell::new(TrainNode::Branch(node)))
    }


    /// Returns the number of leaves of this sub-tree.
    #[inline]
    pub(super) fn leaves(&self) -> usize {
        match self {
            TrainNode::Branch(ref node) => node.leaves,
            TrainNode::Leaf(_) => 1_usize
        }
    }
}


impl Regressor for TrainLeafNode {
    #[inline]
    fn predict(&self, _sample: &Sample, _row: usize) -> f64 {
        self.prediction.0
    }
}


impl Regressor for TrainBranchNode {
    #[inline]
    fn predict(&self, sample: &Sample, row: usize) -> f64 {
        match self.rule.split(sample, row) {
            LR::Left => self.left.borrow().predict(sample, row),
            LR::Right => self.right.borrow().predict(sample, row)
        }
    }
}


impl Regressor for TrainNode {
    #[inline]
    fn predict(&self, sample: &Sample, row: usize) -> f64 {
        match self {
            TrainNode::Branch(ref node) => node.predict(sample, row),
            TrainNode::Leaf(ref node) => node.predict(sample, row)
        }
    }
}


// ------------------------------------------------------------
// Some debug code

impl fmt::Debug for TrainBranchNode {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrainBranchNode")
            .field("threshold", &self.rule)
            .field("leaves", &self.leaves)
            .field("r(t)", &self.loss_as_leaf.0)
            .field("left", &self.left)
            .field("right", &self.right)
            .finish()
    }
}


impl fmt::Debug for TrainLeafNode {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrainLeafNode")
            .field("prediction", &self.prediction.0)
            .field("r(t)", &self.loss_as_leaf.0)
            .finish()
    }
}


impl fmt::Debug for TrainNode {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrainNode::Branch(branch) => {
                write!(f, "{:?}", branch)
            },
            TrainNode::Leaf(leaf) => {
                write!(f, "{:?}", leaf)
            },
        }
    }
}
