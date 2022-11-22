//! Provides some base learning algorithms.
pub mod core;

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
    Loss,
    RTree,
    RTreeRegressor,
};
