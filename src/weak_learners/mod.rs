//! Built-in weak learners.
//!
//! Every booster in this crate is generic over [`crate::WeakLearner`], so
//! any type implementing that trait can be used. The crate ships
//! - [`tree`]: the weighted [`decision_tree`](tree::decision_tree) and
//!   [`regression_tree`](tree::regression_tree) (a weighted least-squares
//!   regression tree for the boosters that fit real-valued responses,
//!   [`GradientBoosting`](crate::GradientBoosting) and
//!   [`LogitBoost`](crate::LogitBoost));
//! - [`finite_class`]: an exact edge maximizer over a finite hypothesis
//!   class, given as [`Classifier`](crate::Classifier)s or as a table of
//!   confidences;
//! - [`rbf`]: a finite class of Gaussian radial basis functions centered at
//!   training examples;
//! - [`linear`]: bounded linear functions (exact edge maximization) and
//!   weighted logistic regression (halfspaces or `tanh` rules);
//! - [`naive_bayes`]: weighted Gaussian naive Bayes;
//! - [`neural_network`]: a network with one hidden layer, trained with
//!   Adam or ADOPT;
//! - [`union`]: the candidate with the largest edge among other weak
//!   learners, possibly with different hypothesis types;
//! - [`boosted`]: a booster together with its weak learner, run on rows
//!   resampled from the distribution.
//!
//! Implement `WeakLearner` for your own hypothesis class to compare
//! boosters with other learners.

pub mod boosted;
pub mod finite_class;
pub mod linear;
pub mod naive_bayes;
pub mod neural_network;
pub mod rbf;
pub mod tree;
pub mod union;
