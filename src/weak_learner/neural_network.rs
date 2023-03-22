//! Two-layered neural network module.

/// Defines a neural network trainer
pub mod nn_weak_learner;
/// Defines a neural network hypothesis
pub mod nn_hypothesis;
/// Defines some loss functions
pub mod nn_loss;
/// Defines activation functions
pub mod activation;
pub(crate) mod layer;

pub use nn_weak_learner::NeuralNetwork;
pub use nn_loss::NNLoss;
pub use activation::Activation;
pub use nn_hypothesis::NNHypothesis;
