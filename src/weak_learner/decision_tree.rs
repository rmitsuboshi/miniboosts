/// Defines the decision tree base learner.
pub mod dtree;
/// Defines the classifier produced by `DTree`.
pub mod dtree_classifier;

/// Defines binning.
pub mod bin;

/// Defines a builder for decision-tree weak learner.
pub mod builder;

/// Defines the inner representations of `DTreeClassifier`.
mod node;
mod criterion;
mod train_node;


pub use dtree_classifier::DTreeClassifier;
pub use dtree::DTree;
pub use criterion::Criterion;
pub use builder::DTreeBuilder;
