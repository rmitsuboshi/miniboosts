//! The files in `weak_learner/` directory defines
//! `WeakLearner` trait and weak learners.

// Provides WeakLearner trait.
pub mod core;

// /// Union of Weak learners.
// pub mod union;

pub(crate) mod common;

// Defines Decision Tree.
mod decision_tree;


// Defines Regression Tree.
mod regression_tree;

// Defines Neural network.
mod neural_network;


// Defines a weak learner for worst-case LPBoost.
mod bad_learner;


// Defines Naive Bayes.
mod naive_bayes;

pub use self::core::WeakLearner;

pub use self::decision_tree::{
    Criterion,
    DecisionTree,
    DecisionTreeBuilder,
    DecisionTreeClassifier,
};

pub use self::naive_bayes::{
    GaussianNB,
    NBayesClassifier,
};


pub use self::regression_tree::{
    RegressionTree,
    RegressionTreeBuilder,
    RegressionTreeRegressor,
};


pub use self::neural_network::{
    NeuralNetwork,
    NNHypothesis,
    NNClassifier,
    NNRegressor,
    Activation,
    NNLoss,
};


pub use self::bad_learner::{
    BadClassifier,
    BadBaseLearnerBuilder,
    BadBaseLearner,
};


// pub use self::union::WLUnion;

pub(crate) use common::type_and_struct;
