//! Built-in weak learners.
//!
//! Every booster in this crate is generic over [`crate::WeakLearner`], so
//! any type implementing that trait can be used. The crate ships one
//! learner, the weighted [`decision_tree`]; implement `WeakLearner` for
//! your own hypothesis class to compare boosters with other learners.

pub mod decision_tree;
