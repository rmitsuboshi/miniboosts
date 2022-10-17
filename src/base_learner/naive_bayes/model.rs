use serde::{
    Serialize,
    Deserialize,
};

/// The probability models for Naive bayes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ProbabilityModel {
    /// Gaussian Naive Bayes model.
    Gaussian,
}
