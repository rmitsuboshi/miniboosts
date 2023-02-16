//! The files in `weak_learner/` directory defines
//! `WeakLearner` trait and weak learners.

/// Provides WeakLearner trait.
pub mod core;

/// Union of Weak learners.
pub mod union;

pub(crate) mod common;

/// Defines Decision Tree.
pub mod decision_tree;


/// Defines Regression Tree.
pub mod regression_tree;


/// Defines Naive Bayes.
pub mod naive_bayes;

pub use self::core::WeakLearner;

pub use self::decision_tree::{
    Criterion,
    DTree,
    DTreeClassifier,
};

pub use self::naive_bayes::{
    GaussianNB,
    NBayesClassifier,
};


pub use self::regression_tree::{
    LossType,
    RTree,
    RTreeRegressor,
};


pub use self::union::WLUnion;

pub(crate) use common::type_and_struct;
