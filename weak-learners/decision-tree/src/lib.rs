//! A simple decision tree algorithm.

pub(crate) mod classifier;
pub mod node;
pub(crate) mod builder;
pub mod split_by;
pub(crate) mod dtree;

pub use builder::DecisionTreeBuilder;
pub use classifier::DecisionTreeClassifier;
pub use dtree::DecisionTree;
pub use split_by::SplitBy;

