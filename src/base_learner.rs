//! Provides some base learning algorithms.
pub mod core;
// 
/// Defines the decision tree class.
pub mod decision_tree;

pub use self::core::BaseLearner;
pub use self::decision_tree::{
    Criterion,
    DTree,
    DTreeClassifier,
};
