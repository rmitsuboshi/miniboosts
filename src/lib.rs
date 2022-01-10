#![warn(missing_docs)]

//! 
//! A crate that provides some boosting algorithms.
//! All the boosting algorithm in this crate has theoretical iteration bound
//! until finding a combined hypothesis.
//! 
//! This crate includes two types of boosting algorithms.
//! 
//! - Empirical risk minimizing (ERM) boosting
//!     The boosting algorithms of this type minimizes the empirical loss
//!     over the training examples.
//!     In this crate,
//!     `AdaBoost` and `AdaBoostV` are correspond to this type.
//! 
//! 
//! - Margin maximizing boosting
//!     The boosting algorithms of this type maximizes the $\ell_1$-margin.
//!     In other words, minimizes the maximum weighted loss over
//!     the training examples.
//!     The resulting combined classifier has nicer generalization error bound
//!     than the ERM one.
//!     In this crate,
//!     `LPBoost`, `ERLPBoost`, `TotalBoost`, and `SoftBoost` are
//!     correspond to this type.

pub mod data_type;
pub mod data_reader;
pub mod booster;
pub mod base_learner;


// Export functions that reads file with some format.
pub use data_reader::{read_csv, read_libsvm};

pub use booster::Booster;
pub use booster::{AdaBoost, AdaBoostV};
pub use booster::{LPBoost, ERLPBoost, SoftBoost, TotalBoost};


pub use base_learner::BaseLearner;
pub use base_learner::DStump;
