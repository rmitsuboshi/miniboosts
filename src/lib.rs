#![warn(missing_docs)]

//! 
//! A crate that provides some boosting algorithms.
//! All the boosting algorithm in this crate, 
//! except `LPBoost`, has theoretical iteration bound 
//! until finding a combined hypothesis. 
//! 
//! This crate includes three types of boosting algorithms. 
//! 
//! * Empirical risk minimizing (ERM) boosting
//!     - `AdaBoost`.
//! 
//! 
//! * Hard margin maximizing boosting
//!     - `AdaBoostV`,
//!     - `TotalBoost`.
//! 
//! 
//! * Soft margin maximizing boosting
//!     - `LPBoost`,
//!     - `SoftBoost`,
//!     - `ERLPBoost`,
//!     - `CERLPBoost`.
//! 

pub mod classifier;
pub mod booster;
pub mod base_learner;
pub mod prelude;


// Export the `Classifier` trait.
pub use classifier::{Classifier, CombinedClassifier};


// Export the `Booster` trait.
pub use booster::Booster;


// Export the boosting algorithms that minimizes the empirical loss.
pub use booster::AdaBoost;


// // Export the boosting algorithms that maximizes the hard margin.
pub use booster::{
    AdaBoostV,
    TotalBoost
};


// Export the boosting algorithms that maximizes the soft margin.
pub use booster::{
    LPBoost,
    ERLPBoost,
    SoftBoost,
    CERLPBoost
};


// Export the `BaseLearner` trait.
pub use base_learner::BaseLearner;


// Export the instances of the `BaseLearner` trait.
pub use base_learner::{
    DTree,
    Criterion,
};
// 
// 
// // Export the instances of the `Classifier` trait.
// // The `CombinedClassifier` is the output of the `Boosting::run(..)`.
pub use base_learner::DTreeClassifier;


