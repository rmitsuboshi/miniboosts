//! A weighted binary decision-tree weak learner (CART-style).
//!
//! - [`DecisionTreeBuilder`] configures the learner (maximal depth,
//!   splitting criterion, number of bins) and builds a [`DecisionTree`].
//! - [`DecisionTree`] implements [`crate::WeakLearner`]: given a booster's
//!   distribution over the examples, it grows a tree on the weighted sample.
//! - [`DecisionTreeClassifier`] is the resulting hypothesis; its leaves
//!   output confidence values in `[-1, 1]`
//!   (`(W₊ - W₋) / (W₊ + W₋)` of the leaf's training weights).
//! - [`SplitBy`] selects the splitting criterion: entropy, Gini index,
//!   twoing rule, or edge.
//!
//! Labels must be `-1` and `+1`. See [`DecisionTree`] for the algorithm,
//! limitations, and the reference (Breiman, Friedman, Olshen & Stone,
//! *Classification and Regression Trees*, 1984).

pub(crate) mod builder;
pub(crate) mod classifier;
pub(crate) mod dtree;
pub mod node;
pub mod split_by;

pub use builder::{DEFAULT_MAX_DEPTH, DEFAULT_NBIN, DecisionTreeBuilder};
pub use classifier::DecisionTreeClassifier;
pub use dtree::{DecisionTree, LeafConfidence};
pub use split_by::SplitBy;
