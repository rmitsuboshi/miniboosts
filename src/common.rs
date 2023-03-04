//! Defines some common functions used in this library.

/// Defines loss function trait and its instances.
pub mod loss_functions;

/// Defines objective functions and its traits.
pub mod objective_functions;

pub(crate) mod checker;

pub use objective_functions::ObjectiveFunction;
