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
//!     - [`AdaBoost`](AdaBoost).
//! 
//! 
//! * Hard margin maximizing boosting
//!     - [`AdaBoostV`](AdaBoostV),
//!     - [`TotalBoost`](TotalBoost).
//! 
//! 
//! * Soft margin maximizing boosting
//!     - [`LPBoost`](LPBoost),
//!     - [`SoftBoost`](SoftBoost),
//!     - [`SmoothBoost`](SmoothBoost),
//!     - [`ERLPBoost`](ERLPBoost),
//!     - [`CERLPBoost`](CERLPBoost),
//!     - [`MLPBoost`](MLPBoost).
//! 
//! This crate also includes some Weak Learners.
//! * Classification
//!     - [`DTree`](DTree)
//! * Regression
//!     - [`RTree`](RTree). Note that the current implement is not efficient.
//! 
//! # Example
//! The following code shows a small example 
//! for running [`LPBoost`](LPBoost).  
//! See also:
//! - [`LPBoost::nu`]
//! - [`LPBoost::tolerance`]
//! - [`DTree`]
//! - [`DTreeClassifier`]
//! - [`CombinedHypothesis<F>`]
//! - [`DTree::max_depth`]
//! - [`DTree::criterion`]
//! - [`DataFrame`]
//! - [`Series`]
//! - [`DataFrame::shape`]
//! - [`CsvReader`]
//! 
//! [`LPBoost::nu`]: LPBoost::nu
//! [`LPBoost::tolerance`]: LPBoost::tolerance
//! [`DTree`]: crate::weak_learner::DTree
//! [`DTreeClassifier`]: crate::weak_learner::DTreeClassifier
//! [`CombinedHypothesis<F>`]: crate::hypothesis::CombinedHypothesis
//! [`DTree::max_depth`]: crate::weak_learner::DTree::max_depth
//! [`DTree::criterion`]: crate::weak_learner::DTree::criterion
//! [`DataFrame`]: polars::prelude::DataFrame
//! [`Series`]: polars::prelude::Series
//! [`DataFrame::shape`]: polars::prelude::DataFrame::shape
//! [`CsvReader`]: polars::prelude::CsvReader
//! 
//! 
//! ```no_run
//! // Import `polars` to read CSV file as `DataFrame`.
//! use polars::prelude::*;
//! // Import `miniboost` default features.
//! use miniboosts::prelude::*;
//! 
//! // Read the training data from the CSV file.
//! let mut data = CsvReader::from_path(path_to_csv_file)
//!     .unwrap()
//!     .has_header(true)
//!     .finish()
//!     .unwrap();
//! 
//! // Split the column corresponding to labels.
//! let target = data.drop_in_place(class_column_name).unwrap();
//! 
//! // Get the number of training examples.
//! let n_sample = data.shape().0 as f64;
//! 
//! // Initialize `LPBoost` and set the tolerance parameter as `0.01`.
//! // This means `booster` returns a hypothesis whose training error is
//! // less than `0.01` if the traing examples are linearly separable.
//! // Note that the default tolerance parameter is set as `1 / n_sample`,
//! // where `n_sample = data.shape().0` is 
//! // the number of training examples in `data`.
//! // Further, at the end of this chain,
//! // LPBoost calls `LPBoost::nu` to set the capping parameter 
//! // as `0.1 * n_sample`, which means that, 
//! // at most, `0.1 * n_sample` examples are regarded as outliers.
//! let booster = LPBoost::init(&data, &target)
//!     .tolerance(0.01)
//!     .nu(0.1 * n_sample);
//! 
//! // Set the weak learner with setting parameters.
//! let weak_learner = DecisionTree::init(&data, &target)
//!     .max_depth(2)
//!     .criterion(Criterion::Edge);
//! 
//! // Run `LPBoost` and obtain the resulting hypothesis `f`.
//! let f: CombinedHypothesis<DTreeClassifier> = booster.run(&weak_learner);
//! 
//! // Get the predictions on the training set.
//! let predictions: Vec<i64> = f.predict_all(&data);
//! 
//! // Calculate the training loss.
//! let training_loss = target.i64()
//!     .unwrap()
//!     .into_iter()
//!     .zip(predictions)
//!     .map(|(true_label, prediction) {
//!         let true_label = true_label.unwrap();
//!         if true_label == prediction { 0.0 } else { 1.0 }
//!     })
//!     .sum::<f64>()
//!     / n_sample;
//! 
//!
//! println!("Training Loss is: {training_loss}");
//! ```
pub mod sample;
pub mod common;
pub mod hypothesis;
pub mod booster;
pub mod weak_learner;
pub mod prelude;

pub mod research;


// Export the struct that represents batch sample
pub use sample::{
    Sample,
    Feature,
};


// Export some traits and the combined hypothesis struct.
pub use hypothesis::{
    Classifier,
    Regressor,
    CombinedHypothesis,
};




// Export the `Booster` trait.
pub use booster::{
    Booster,
    State,
};


// Export the boosting algorithms that minimizes the empirical loss.
pub use booster::{
    AdaBoost,
};


// Export the boosting algorithms that maximizes the hard margin.
pub use booster::{
    AdaBoostV,
    // SparsiBoost,
};


// Export the boosting algorithms that maximizes the soft margin.
pub use booster::{
    SmoothBoost,
    CERLPBoost,
};


#[cfg(feature="extended")]
pub use booster::{
    TotalBoost,
    LPBoost,
    ERLPBoost,
    SoftBoost,
    MLPBoost,
};


// Export the boosting algorithms for regression
pub use booster::{
    GBM,
};


// Export the `WeakLearner` trait.
pub use weak_learner::WeakLearner;


// Export the instances of the `WeakLearner` trait.
pub use weak_learner::{
    DTree,
    Criterion,

    // WLUnion,

    // GaussianNB,
    NeuralNetwork,
    Activation,
    NNLoss,
};


// Export the instances of the `Classifier` trait.
// The `CombinedClassifier` is the output of the `Boosting::run(..)`.
pub use weak_learner::{
    DTreeClassifier,

    NNHypothesis,
    // NBayesClassifier,
};

pub use weak_learner::{
    RTree,
    RTreeRegressor,
    LossType,
};

/// Some useful functions / traits
pub use common::{
    loss_functions::{
        GBMLoss,
        LossFunction,
    },
};


pub use research::{
    Logger,
};
