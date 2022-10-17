//! Provides some base learning algorithms.
pub mod core;

/// Defines Decision Tree.
pub mod decision_tree;


/// Defines Naive Bayes.
pub mod naive_bayes;

pub use self::core::BaseLearner;
pub use self::decision_tree::{
    Criterion,
    DTree,
    DTreeClassifier,
};
pub use self::naive_bayes::{
    GaussianNB,
    NBayesClassifier,
};
