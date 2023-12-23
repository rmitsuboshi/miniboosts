//! Two-layered neural network module.  

// Defines a neural network trainer
mod nn_weak_learner;
// Defines a neural network hypothesis
mod nn_hypothesis;
// Defines some loss functions
mod nn_loss;
// Defines activation functions
mod activation;
pub(crate) mod layer;

pub use nn_weak_learner::NeuralNetwork;
pub use nn_loss::NNLoss;
pub use activation::Activation;
pub use nn_hypothesis::{
    NNHypothesis,
    NNClassifier,
    NNRegressor,
};
