//! Built-in weak learners.
//!
//! Every booster in this crate is generic over [`crate::WeakLearner`], so
//! any type implementing that trait can be used. The crate ships
//! - the weighted [`decision_tree`];
//! - [`finite_class`]: an exact edge maximizer over a finite hypothesis
//!   class, given as [`Classifier`](crate::Classifier)s or as a table of
//!   confidences;
//! - [`rbf`]: a finite class of Gaussian radial basis functions centered at
//!   training examples;
//! - [`linear`]: bounded linear functions (exact edge maximization) and
//!   weighted logistic regression (halfspaces or `tanh` rules);
//! - [`naive_bayes`]: weighted Gaussian naive Bayes;
//! - [`regression_tree`]: a weighted least-squares regression tree for the
//!   boosters that fit real-valued responses
//!   ([`GradientBoosting`](crate::GradientBoosting),
//!   [`LogitBoost`](crate::LogitBoost)).
//!
//! Implement `WeakLearner` for your own hypothesis class to compare
//! boosters with other learners.

pub mod decision_tree;
pub mod finite_class;
pub mod linear;
pub mod naive_bayes;
pub mod rbf;
pub mod regression_tree;
