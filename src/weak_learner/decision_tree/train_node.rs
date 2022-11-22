//! Defines the inner representation 
//! of the Decision Tree class.
use polars::prelude::*;
use crate::Classifier;


use super::split_rule::*;


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
    pub(super) prediction: f64,


    // Total mass on this node.
    pub(self) total_weight: f64,


    // Training error as a leaf
    pub(self) loss_as_leaf: f64,


    pub(self) leaves: usize,
}


impl TrainBranchNode {
    /// Returns the node misclassification cost of this node.
    #[inline]
    pub(self) fn node_misclassification_cost(&self) -> f64 {
        let r = self.loss_as_leaf;
        let p = self.total_weight;

        r * p
    }
}


/// Represents the leaf nodes of decision tree.
pub struct TrainLeafNode {
    pub(super) prediction: f64,
    pub(self) total_weight: f64,
    pub(self) loss_as_leaf: f64,
}


impl TrainLeafNode {
    /// Returns the node misclassification cost of this node.
    #[inline]
    pub(self) fn node_misclassification_cost(&self) -> f64 {
        let r = self.loss_as_leaf;
        let p = self.total_weight;

        r * p
    }
}


impl From<TrainBranchNode> for TrainLeafNode {
    #[inline]
    fn from(branch: TrainBranchNode) -> Self {
        Self {
            prediction: branch.prediction,
            total_weight: branch.total_weight,
            loss_as_leaf: branch.loss_as_leaf,
        }
    }
}


impl TrainNode {
    /// Construct a leaf node from the given arguments.
    #[inline]
    pub(super) fn leaf(prediction: i64,
                       total_weight: f64,
                       loss_as_leaf: f64)
        -> Rc<RefCell<Self>>
    {
        let prediction = prediction as f64;
        let leaf = TrainLeafNode {
            prediction,
            total_weight,
            loss_as_leaf,
        };


        Rc::new(RefCell::new(TrainNode::Leaf(leaf)))
    }


    /// Construct a branch node from the arguments.
    #[inline]
    pub(super) fn branch(rule: Splitter,
                         left: Rc<RefCell<TrainNode>>,
                         right: Rc<RefCell<TrainNode>>,
                         prediction: i64,
                         total_weight: f64,
                         loss_as_leaf: f64)
        -> Rc<RefCell<Self>>
    {
        let prediction = prediction as f64;
        let leaves = left.borrow().leaves() + right.borrow().leaves();
        let node = TrainBranchNode {
            rule,
            left,
            right,

            prediction,
            total_weight,
            loss_as_leaf,


            leaves,
        };

        Rc::new(RefCell::new(TrainNode::Branch(node)))
    }


    #[inline]
    pub(super) fn is_leaf(&self) -> bool {
        match self {
            TrainNode::Branch(_) => false,
            TrainNode::Leaf(_) => true,
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


    /// Returns the node misclassification cost of this node.
    #[inline]
    pub(super) fn node_misclassification_cost(&self) -> f64 {
        match self {
            TrainNode::Branch(ref branch)
                => branch.node_misclassification_cost(),
            TrainNode::Leaf(ref leaf)
                => leaf.node_misclassification_cost(),
        }
    }


    /// Execute preprocessing before the pruning.
    /// This method removes the leaves that do not affect
    /// the training error.
    #[inline]
    pub(super) fn remove_redundant_nodes(&mut self) {

        if let TrainNode::Branch(ref mut branch) = self {

            // If the left node is not a leaf,
            // move to the leaf node.
            if !branch.left.borrow().is_leaf() {
                branch.left.borrow_mut().remove_redundant_nodes();
                return;
            }
            // If the right node is not a leaf,
            // move to the leaf node.
            if !branch.right.borrow().is_leaf() {
                branch.right.borrow_mut().remove_redundant_nodes();
                return;
            }


            let t = branch.node_misclassification_cost();
            let l = branch.left.borrow().node_misclassification_cost();
            let r = branch.right.borrow().node_misclassification_cost();


            if t == l + r {
                if let TrainNode::Branch(branch) = self {
                    *self = TrainNode::Leaf(
                        TrainLeafNode {
                            prediction: branch.prediction,
                            total_weight: branch.total_weight,
                            loss_as_leaf: branch.loss_as_leaf,
                        }
                    );
                }
            } else {
                branch.left.borrow_mut().remove_redundant_nodes();
                branch.right.borrow_mut().remove_redundant_nodes();
            }
        }
    }
}


impl Classifier for TrainLeafNode {
    #[inline]
    fn confidence(&self, _data: &DataFrame, _row: usize) -> f64 {
        self.prediction
    }
}


impl Classifier for TrainBranchNode {
    #[inline]
    fn confidence(&self, data: &DataFrame, row: usize) -> f64 {
        match self.rule.split(data, row) {
            LR::Left => self.left.borrow().confidence(data, row),
            LR::Right => self.right.borrow().confidence(data, row)
        }
    }
}


impl Classifier for TrainNode {
    #[inline]
    fn confidence(&self, data: &DataFrame, row: usize) -> f64 {
        match self {
            TrainNode::Branch(ref node) => node.confidence(data, row),
            TrainNode::Leaf(ref node) => node.confidence(data, row)
        }
    }
}



// pub(super) struct TrainNodeGuard<'a> {
//     guard: Ref<'a, TrainNode>
// }
// 
// 
// impl<'b> Deref for TrainNodeGuard<'b> {
//     type Target = TrainNode;
// 
//     fn deref(&self) -> &Self::Target {
//         &self.guard
//     }
// }
// 
// 
// pub(super) fn get_ref(node: &Rc<RefCell<TrainNode>>) -> TrainNodeGuard
// {
//     TrainNodeGuard {
//         guard: node.borrow()
//     }
// }


impl fmt::Debug for TrainBranchNode {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrainBranchNode")
            .field("threshold", &self.rule)
            .field("leaves", &self.leaves)
            .field("p(t)", &self.total_weight)
            .field("r(t)", &self.loss_as_leaf)
            .field("left", &self.left)
            .field("right", &self.right)
            .finish()
    }
}


impl fmt::Debug for TrainLeafNode {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrainLeafNode")
            .field("prediction", &self.prediction)
            .field("p(t)", &self.total_weight)
            .field("r(t)", &self.loss_as_leaf)
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
