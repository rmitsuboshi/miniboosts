//! A simple decision tree algorithm.

pub(crate) mod builder;
pub(crate) mod classifier;
pub(crate) mod dtree;
pub mod node;
pub mod split_by;

pub use builder::DecisionTreeBuilder;
pub use classifier::DecisionTreeClassifier;
pub use dtree::DecisionTree;
pub use split_by::SplitBy;
