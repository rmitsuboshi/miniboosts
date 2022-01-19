//! Provides some base learning algorithms.
pub mod core;
pub mod dstump;
// pub mod dtree;
// pub mod ltf;

pub use self::core::BaseLearner;
pub use self::dstump::{DStump, DStumpClassifier};
