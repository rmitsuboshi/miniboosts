//! The files in `weak_learner/` directory defines
//! `WeakLearner` trait and weak learners.

/// Provides WeakLearner trait.
pub mod core;

// /// Union of Weak learners.
// pub mod union;

pub(crate) mod common;

/// Defines Decision Tree.
pub mod decision_tree;


/// Defines Regression Tree.
pub mod regression_tree;

/// Defines Neural network.
pub mod neural_network;


/// Defines a weak learner for worst-case LPBoost.
pub mod bad_learner;


/// Defines Naive Bayes.
pub mod naive_bayes;

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
    LossType,
    RegressionTree,
    RegressionTreeBuilder,
    RegressionTreeRegressor,
};


pub use self::neural_network::{
    NeuralNetwork,
    NNHypothesis,
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
