//! Provides some base learning algorithms.
pub mod core;

/// Defines the decision stump class.
pub mod decision_stump;
// pub mod dtree;
// pub mod ltf;

pub use self::core::BaseLearner;
pub use self::decision_stump::{DStump, DStumpClassifier};
