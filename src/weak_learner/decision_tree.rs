// Defines the decision tree base learner.
mod decision_tree_algorithm;
// Defines the classifier produced by `DecisionTree`.
mod decision_tree_classifier;

// Defines a builder for decision-tree weak learner.
mod builder;

pub(crate) mod bin;

// Defines the inner representations of `DecisionTreeClassifier`.
mod node;
mod criterion;
mod train_node;


pub use decision_tree_classifier::DecisionTreeClassifier;
pub use decision_tree_algorithm::DecisionTree;
pub use criterion::Criterion;
pub use builder::DecisionTreeBuilder;
