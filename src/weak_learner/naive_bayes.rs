/// Defines Naive Bayes classifier.
pub mod nbayes;
/// Defines Naive Bayes Classifiers returned by `NBayes`.
mod nbayes_classifier;

/// Defines probability density/mass functions.
mod probability;

pub use nbayes::GaussianNB;
pub use nbayes_classifier::NBayesClassifier;
