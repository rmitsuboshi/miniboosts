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
//!     - [`AdaBoost`],
//!     - [`GraphSepBoost`].
//! 
//! 
//! * Hard margin maximizing boosting
//!     - [`AdaBoostV`],
//!     - [`TotalBoost`](crate::booster::TotalBoost).
//! 
//! 
//! * Soft margin maximizing boosting
//!     - [`LPBoost`](crate::booster::LPBoost),
//!     - [`SoftBoost`](crate::booster::SoftBoost),
//!     - [`SmoothBoost`],
//!     - [`ERLPBoost`](crate::booster::ERLPBoost),
//!     - [`CERLPBoost`],
//!     - [`MLPBoost`](crate::booster::MLPBoost).
//! 
//!
//! This crate also includes some Weak Learners.
//! * Classification
//!     - [`DecisionTree`],
//!     - [`NeuralNetwork`],
//!     - [`GaussianNB`],
//!     - [`BadBaseLearner`] (The bad base learner for LPBoost).
//! * Regression
//!     - [`RegressionTree`]. Note that the current implement is not efficient.
//! 
//! # Example
//! The following code shows a small example for running [`LPBoost`].  
//! See also:
//! - [`LPBoost::nu`]
//! - [`LPBoost::tolerance`]
//! - [`DecisionTree`]
//! - [`DecisionTreeClassifier`]
//! 
//! [`LPBoost::nu`]: LPBoost::nu
//! [`LPBoost::tolerance`]: LPBoost::tolerance
//! [`DecisionTree`]: crate::weak_learner::DecisionTree
//! [`DecisionTreeClassifier`]: crate::weak_learner::DecisionTreeClassifier
//! [`NeuralNetwork`]: crate::weak_learner::NeuralNetwork
//! [`CombinedHypothesis<F>`]: crate::hypothesis::CombinedHypothesis
//! [`GaussianNB`]: crate::weak_learner::GaussianNB
//! [`BadBaseLearner`]: crate::weak_learner::BadBaseLearner
//! 
//! ```no_run
//! use miniboosts::prelude::*;
//! 
//! // Read the training sample from the CSV file.
//! // We use the column named `class` as the label.
//! let path = "path/to/dataset.csv";
//! let has_header = true;
//! let sample = Sample::from_csv(path, has_header)
//!     .unwrap()
//!     .set_target("class");
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
//! let booster = LPBoost::init(&sample)
//!     .tolerance(0.01)
//!     .nu(0.1 * n_sample);
//! 
//! // Set the weak learner with setting parameters.
//! let weak_learner = DecisionTreeBuilder::new(&sample)
//!     .max_depth(2)
//!     .criterion(Criterion::Entropy)
//!     .build();
//! 
//! // Run `LPBoost` and obtain the resulting hypothesis `f`.
//! let f = booster.run(&weak_learner);
//! 
//! // Get the predictions on the training set.
//! let predictions = f.predict_all(&data);
//! 
//! // Calculate the training loss.
//! let target = sample.target();
//! let training_loss = target.into_iter()
//!     .zip(predictions)
//!     .map(|(&y, fx)| if y as i64 == fx { 0.0 } else { 1.0 })
//!     .sum::<f64>()
//!     / n_sample;
//!
//! println!("Training Loss is: {training_loss}");
//! ```
mod sample;
mod common;
mod hypothesis;
mod booster;
mod weak_learner;

pub mod prelude;
pub mod research;
// pub mod pywriter;


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
    NaiveAggregation,
};




// Export the `Booster` trait.
pub use booster::Booster;

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


// Export the boosting algorithms that maximizes the soft margin.
// (These boosting algorithms use Gurobi)
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


// Export other boosting algorithms
pub use booster::GraphSepBoost;


// Export the `WeakLearner` trait.
pub use weak_learner::WeakLearner;


// Export the instances of the `WeakLearner` trait.
pub use weak_learner::{
    DecisionTree,
    DecisionTreeBuilder,
    Criterion,

    // WLUnion,

    GaussianNB,
    NeuralNetwork,
    Activation,
    NNLoss,

    BadBaseLearner,
    BadBaseLearnerBuilder,
};


// Export the instances of the `Classifier` trait.
// The `CombinedClassifier` is the output of the `Boosting::run(..)`.
pub use weak_learner::{
    DecisionTreeClassifier,

    NNHypothesis,
    NNClassifier,
    NNRegressor,

    BadClassifier,
    NBayesClassifier,
};

pub use weak_learner::{
    RegressionTree,
    RegressionTreeBuilder,
    RegressionTreeRegressor,
    LossType,
};

/// Some useful functions / traits
pub use common::{
    frank_wolfe::{
        FWType,
    },
    loss_functions::{
        GBMLoss,
        LossFunction,
    },
};


pub use research::{
    Logger,
    LoggerBuilder,
    objective_functions::{
        SoftMarginObjective,
        HardMarginObjective,
        ExponentialLoss,
    },
};
