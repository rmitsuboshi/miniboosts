/// Defines Naive Bayes classifier.
mod nbayes;
/// Defines Naive Bayes Classifiers returned by `NBayes`.
mod nbayes_classifier;

/// Defines probability density/mass functions.
mod probability;

pub use nbayes::GaussianNB;
pub use nbayes_classifier::NBayesClassifier;
