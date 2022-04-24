//! Provides some base learning algorithms.
pub mod core;

/// Defines the decision stump class.
pub mod decision_stump;

// /// Defines the decision tree class.
// pub mod decision_tree;

pub use self::core::BaseLearner;
pub use self::decision_stump::{DStump, DStumpClassifier};
// pub use self::decision_tree::{DTree, DTreeClassifier};
// pub use self::decision_tree::{SplitRule, StumpSplit};
