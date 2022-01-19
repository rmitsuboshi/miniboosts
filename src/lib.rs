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
pub mod classifier;
pub mod booster;
pub mod base_learner;

// Export struct `Sample`.
pub use data_type::{Sample, Data, Label, DType};


// Export functions that reads file with some format.
pub use data_reader::{read_csv, read_libsvm};

// Export the `Classifier` trait.
pub use classifier::{Classifier, CombinedClassifier};

// Export the `Booster` trait.
pub use booster::Booster;

// Export the boosting algorithms that minimizes the empirical loss.
pub use booster::AdaBoost;

// Export the boosting algorithms that maximizes the hard margin.
pub use booster::{AdaBoostV, TotalBoost};

// Export the boosting algorithms that maximizes the soft margin.
pub use booster::{LPBoost, ERLPBoost, SoftBoost, CERLPBoost};


// Export the `BaseLearner` trait.
pub use base_learner::BaseLearner;

// Export the instances of the `BaseLearner` trait.
pub use base_learner::DStump;

// Export the `Classifier` trait.
// pub use base_learner::Classifier;

// Export the instances of the `Classifier` trait.
// The `CombinedClassifier` is the output of the `Boosting::run(..)`.
// pub use base_learner::{DStumpClassifier, CombinedClassifier};
pub use base_learner::DStumpClassifier;


