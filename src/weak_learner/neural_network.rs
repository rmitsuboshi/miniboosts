//! Two-layered neural network module.

/// Defines a neural network trainer
pub mod neural_network;
/// Defines a neural network hypothesis
pub mod nn_hypothesis;
/// Defines some loss functions
pub mod nn_loss;
/// Defines activation functions
pub mod activation;
pub(crate) mod layer;

pub use neural_network::NeuralNetwork;
pub use nn_loss::NNLoss;
pub use activation::Activation;
pub use nn_hypothesis::NNHypothesis;
