//! Defines some common functions used in this library.

/// Defines loss function trait and its instances.
pub mod loss_functions;

/// Defines objective functions and its traits.
pub mod objective_functions;

/// Defines some useful functions such as edge calculation.
pub mod utils;

/// Defines the Frank-Wolfe algorithms.
pub mod frank_wolfe;

/// Defines some checker functions.
pub(crate) mod checker;

/// Defines machine learning tasks.
pub(crate) mod task;

pub use objective_functions::ObjectiveFunction;
pub use task::*;
