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


    // The parent node
    pub(super) parent: Option<Rc<RefCell<TrainNode>>>,


    // Left child
    pub(super) left: Rc<RefCell<TrainNode>>,


    // Right child
    pub(super) right: Rc<RefCell<TrainNode>>,


    // A label that have most weight on this node.
    pub(super) prediction: i64,


    // Total mass on this node.
    pub(self) train_total_weight: f64,


    // Training error as a leaf
    pub(self) train_error_as_leaf: f64,


    // Training error as a tree.
    // This value is the sum of `train_error_as_leaf` of childrens
    // from this node.
    pub(self) train_error_as_tree: f64,



    // Test error as a leaf, tree, respectively.
    pub(self) test_total_weight: f64,
    pub(self) test_error_as_leaf: f64,


    pub(self) leaves: usize,
}


impl TrainBranchNode {
    /// Returns the node misclassification cost of this node.
    #[inline]
    pub(self) fn train_node_misclassification_cost(&self) -> f64 {
        let r = self.train_error_as_leaf;
        let p = self.train_total_weight;

        r * p
    }
}


/// Represents the leaf nodes of decision tree.
pub struct TrainLeafNode {
    pub(super) parent: Option<Rc<RefCell<TrainNode>>>,
    pub(super) prediction: i64,
    pub(self) train_total_weight: f64,
    pub(self) train_error_as_leaf: f64,
    pub(self) test_total_weight: f64,
    pub(self) test_error_as_leaf: f64,
}


impl TrainLeafNode {
    /// Returns the node misclassification cost of this node.
    #[inline]
    pub(self) fn train_node_misclassification_cost(&self) -> f64 {
        let r = self.train_error_as_leaf;
        let p = self.train_total_weight;

        r * p
    }


    /// Returns the node misclassification cost of this node.
    #[inline]
    pub(self) fn test_node_misclassification_cost(&self) -> f64 {
        let r = self.test_error_as_leaf;
        let p = self.test_total_weight;

        r * p
    }
}


impl From<TrainBranchNode> for TrainLeafNode {
    #[inline]
    fn from(branch: TrainBranchNode) -> Self {
        Self {
            parent: branch.parent,
            prediction: branch.prediction,
            train_total_weight: branch.train_total_weight,
            train_error_as_leaf: branch.train_error_as_leaf,
            test_total_weight: branch.test_total_weight,
            test_error_as_leaf: branch.test_error_as_leaf
        }
    }
}


impl TrainNode {
    /// Construct a leaf node from the given arguments.
    #[inline]
    pub(super) fn leaf(prediction: i64,
                       train_total_weight: f64,
                       train_error_as_leaf: f64,
                       test_total_weight: f64,
                       test_error_as_leaf: f64)
        -> Rc<RefCell<Self>>
    {
        let leaf = TrainLeafNode {
            parent: None,
            prediction,
            train_total_weight,
            train_error_as_leaf,
            test_total_weight,
            test_error_as_leaf,
        };


        Rc::new(RefCell::new(TrainNode::Leaf(leaf)))
    }


    /// Construct a branch node from the arguments.
    #[inline]
    pub(super) fn branch(rule: Splitter,
                         left: Rc<RefCell<TrainNode>>,
                         right: Rc<RefCell<TrainNode>>,
                         prediction: i64,
                         train_total_weight: f64,
                         train_error_as_leaf: f64,
                         test_total_weight: f64,
                         test_error_as_leaf: f64)
        -> Rc<RefCell<Self>>
    {
        let leaves = left.borrow().leaves() + right.borrow().leaves();
        let node = TrainBranchNode {
            rule,
            parent: None,
            left,
            right,

            // impurity,
            prediction,
            train_total_weight,
            train_error_as_leaf,
            train_error_as_tree: f64::MAX,

            test_total_weight,
            test_error_as_leaf,

            leaves,
        };

        let node = Rc::new(RefCell::new(TrainNode::Branch(node)));


        if let TrainNode::Branch(branch) = & *node.borrow() {
            {
                let p = Rc::clone(&node);
                branch.left.borrow_mut().set_parent(p);
            }
            {
                let p = Rc::clone(&node);
                branch.right.borrow_mut().set_parent(p);
            }
        }


        node
    }


    #[inline]
    pub(self) fn set_parent(&mut self, parent: Rc<RefCell<TrainNode>>) {
        match self {
            TrainNode::Branch(b) => {
                b.parent = Some(parent);
            },
            TrainNode::Leaf(l) => {
                l.parent = Some(parent);
            }
        }
    }


    #[inline]
    pub(super) fn remove_parent(&mut self) {
        match self {
            TrainNode::Branch(b) => {
                b.parent = None;
                b.left.borrow_mut().remove_parent();
                b.right.borrow_mut().remove_parent();
            },
            TrainNode::Leaf(l) => {
                l.parent = None;
            }
        }
    }


    #[inline]
    pub(super) fn is_leaf(&self) -> bool {
        match self {
            TrainNode::Branch(_) => false,
            TrainNode::Leaf(_) => true,
        }
    }


    /// Returns the value that represents the `weak link`.
    #[inline]
    pub(super) fn alpha(&self) -> f64 {
        match self {
            TrainNode::Branch(branch) => {
                let error_as_leaf = self.train_node_misclassification_cost();
                let error_as_tree = branch.train_error_as_tree;
                let leaves = self.leaves() as f64;

                // DEBUG
                assert_eq!(self.leaves(), 2);
                assert!(error_as_leaf.is_finite());
                assert!(error_as_tree.is_finite());

                assert!(error_as_leaf >= 0.0);
                assert!(error_as_tree >= 0.0);
                assert!(error_as_leaf >= error_as_tree);
                // END OF DEBUG


                (error_as_leaf - error_as_tree) / (leaves - 1.0)
            },
            TrainNode::Leaf(_) => f64::MAX
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
    pub(super) fn train_node_misclassification_cost(&self) -> f64 {
        match self {
            TrainNode::Branch(ref branch)
                => branch.train_node_misclassification_cost(),
            TrainNode::Leaf(ref leaf)
                => leaf.train_node_misclassification_cost(),
        }
    }


    /// Returns the tree misclassification cost of this node.
    #[inline]
    pub(super) fn test_tree_misclassification_cost(&self) -> f64 {
        match self {
            TrainNode::Branch(ref branch) => {
                let l = branch.left.borrow()
                    .test_tree_misclassification_cost();
                let r = branch.right.borrow()
                    .test_tree_misclassification_cost();
                l + r
            },
            TrainNode::Leaf(ref leaf)
                => leaf.test_node_misclassification_cost(),
        }
    }


    /// Execute preprocessing before the pruning.
    /// This method removes the leaves that do not affect
    /// the training error.
    #[inline]
    pub(super) fn pre_process(&mut self) {

        if let TrainNode::Branch(ref mut branch) = self {

            // If the left node is not a leaf,
            // move to the leaf node.
            if !branch.left.borrow().is_leaf() {
                branch.left.borrow_mut().pre_process();
                return;
            }
            // If the right node is not a leaf,
            // move to the leaf node.
            if !branch.right.borrow().is_leaf() {
                branch.right.borrow_mut().pre_process();
                return;
            }


            let t = branch.train_node_misclassification_cost();
            let l = branch.left.borrow().train_node_misclassification_cost();
            let r = branch.right.borrow().train_node_misclassification_cost();


            // DEBUG
            assert!(t >= l + r);
            // END OF DEBUG


            if t == l + r {
                self.prune();
            } else {
                branch.left.borrow_mut().pre_process();
                branch.right.borrow_mut().pre_process();
            }
        }
    }


    /// Convert `self` to the leaf node.
    /// If `self` is already a leaf node, do nothing.
    #[inline]
    pub(super) fn prune(&mut self) -> Option<Rc<RefCell<TrainNode>>> {
        let mut parent = None;
        if let TrainNode::Branch(ref mut branch) = self {

            // Take the parent node for return
            if let Some(p) = &branch.parent {
                p.borrow_mut().reduce_leaves_by_2();
                parent = Some(Rc::clone(p));
            }


            // Remove the parent node to reduce the reference count of `Rc`.
            branch.left.borrow_mut()
                .remove_parent();
            branch.right.borrow_mut()
                .remove_parent();

            *self = TrainNode::Leaf(TrainLeafNode {
                parent: parent.clone(),
                prediction: branch.prediction,
                train_total_weight: branch.train_total_weight,
                train_error_as_leaf: branch.train_error_as_leaf,
                test_total_weight: branch.test_total_weight,
                test_error_as_leaf: branch.test_error_as_leaf,
            });
        }
        parent
    }


    #[inline]
    fn reduce_leaves_by_2(&mut self) {
        if let TrainNode::Branch(b) = self {
            b.leaves -= 2;
            if let Some(p) = &b.parent {
                p.borrow_mut().reduce_leaves_by_2();
            }
        }
    }
}


impl Classifier for TrainLeafNode {
    #[inline]
    fn predict(&self, _data: &DataFrame, _row: usize) -> i64 {
        self.prediction
    }
}


impl Classifier for TrainBranchNode {
    #[inline]
    fn predict(&self, data: &DataFrame, row: usize) -> i64 {
        match self.rule.split(data, row) {
            LR::Left => self.left.borrow().predict(data, row),
            LR::Right => self.right.borrow().predict(data, row)
        }
    }
}


impl Classifier for TrainNode {
    #[inline]
    fn predict(&self, data: &DataFrame, row: usize) -> i64 {
        match self {
            TrainNode::Branch(ref node) => node.predict(data, row),
            TrainNode::Leaf(ref node) => node.predict(data, row)
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
            .field("p(t)", &self.train_total_weight)
            .field("r(t)", &self.train_error_as_leaf)
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
            .field("p(t)", &self.train_total_weight)
            .field("r(t)", &self.train_error_as_leaf)
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
