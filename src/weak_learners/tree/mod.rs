//! Tree weak learners, which share the binning of feature values
//! ([`crate::core::binning`]) and the split rules ([`crate::core::tree`]):
//! - [`decision_tree`]: a weighted classification tree;
//! - [`regression_tree`]: a weighted least-squares regression tree for the
//!   boosters that fit real-valued responses.

pub mod decision_tree;
pub mod regression_tree;
