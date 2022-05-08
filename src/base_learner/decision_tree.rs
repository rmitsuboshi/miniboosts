/// Defines the decision tree base learner.
pub mod dtree;
/// Defines the classifier produced by `DTree`.
pub mod dtree_classifier;

/// Defines the inner representations of `DTreeClassifier`.
mod node;
mod train_node;

mod split_rule;
// mod measure;


pub use dtree_classifier::DTreeClassifier;
pub use dtree::DTree;
pub use split_rule::{SplitRule, StumpSplit};
