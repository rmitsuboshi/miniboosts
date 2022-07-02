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
//! 
//! 
//! # Example
//! 
//! ```rust,no_run
//! 
//! // `run()` and is the method of Booster trait
//! use lycaon::Booster;
//! 
//! // `predict(&data)` is the method of Classifier trait
//! use lycaon::Classifier;
//! 
//! // In this example, we use AdaBoost as the booster
//! use lycaon::AdaBoost;
//! 
//! // In this example,
//! // we use Decision stump (Decision Tree of depth 1) as the base learner
//! use lycaon::DStump;
//! 
//! // This function reads a file with LIBSVM format
//! use lycaon::data_reader::read_csv;
//! 
//! 
//! use lycaon::Sample;
//! 
//! 
//! fn main() {
//!     // Set file name
//!     let file = "/path/to/input/file.csv";
//! 
//!     // Read file
//!     let sample = read_csv(file).unwrap();
//! 
//!     // Initialize Booster
//!     let mut adaboost = AdaBoost::init(&sample);
//! 
//!     // Initialize Base Learner
//!     let dstump = DStump::init(&sample);
//! 
//!     // Set tolerance parameter
//!     let tolerance = 0.1;
//! 
//!     // Run boosting algorithm
//!     let f = adaboost.run(&dstump, &sample, tolerance);
//! 
//! 
//!     // These assertion may fail if the dataset are not linearly separable.
//!     for (dat, lab) in sample.iter() {
//!         // Check the predictions
//!         assert_eq!(f.predict(dat), *lab);
//!     }
//! }
//! ```
//! 
//! If you use the soft margin maximizing boostings like LPBoost,
//! you can set the capping parameter as follows:
//! 
//! ```rust,no_run
//! use lycaon::LPBoost;
//! use lycaon::read_csv;
//! use lycaon::Sample;
//! 
//! let file = "/path/to/input/file.csv";
//! let sample: Sample<Vec<f64>, f64> = read_csv(file).unwrap();
//! let capping_param = 0.2 * sample.len() as f64;
//! let booster = LPBoost::init(&sample)
//!     .capping(capping_param);
//! ```
//! 
//! Currently, the inner optimization problems are solved via 
//! Gurobi optimizer.
//! If you have the license of Gurobi,
//! you can use all boosters.
//! Otherwise, you can use the following boosting algorithms:
//! 
//! * AdaBoost
//! * AdaBoostV
//! * CERLPBoost
//! 
//! I'm working on for implementing the LP & QP solver.
//! Please looking forward to the future updates.

// pub mod data_type;
// pub mod data_reader;
pub mod classifier;
pub mod booster;
pub mod base_learner;

// Export struct `Sample`.
// pub use data_type::{Sample, Data, DataBounds};


// Export functions that reads file with some format.
// pub use data_reader::{read_csv, read_libsvm};


// Export the `Classifier` trait.
pub use classifier::{Classifier, CombinedClassifier};


// Export the `Booster` trait.
pub use booster::Booster;


// Export the boosting algorithms that minimizes the empirical loss.
pub use booster::AdaBoost;
// 
// 
// // // Export the boosting algorithms that maximizes the hard margin.
// pub use booster::{AdaBoostV, TotalBoost};
// // 
// // 
// // Export the boosting algorithms that maximizes the soft margin.
// pub use booster::{LPBoost, ERLPBoost, SoftBoost, CERLPBoost};
// // 
// // 
// Export the `BaseLearner` trait.
pub use base_learner::BaseLearner;
// // 
// // 
// Export the instances of the `BaseLearner` trait.
pub use base_learner::DTree;
// pub use base_learner::DStump;
// 
// 
// // Export the instances of the `Classifier` trait.
// // The `CombinedClassifier` is the output of the `Boosting::run(..)`.
// pub use base_learner::DStumpClassifier;
// pub use base_learner::DTreeClassifier;


