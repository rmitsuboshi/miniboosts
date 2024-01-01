//! Defines some common functions used in this library.

/// Defines loss function trait and its instances.
pub mod loss_functions;

/// Defines some useful functions such as edge calculation.
pub mod utils;

/// Defines the Frank-Wolfe algorithms.
pub mod frank_wolfe;

/// Defines some checker functions.
pub(crate) mod checker;

/// Defines machine learning tasks.
pub(crate) mod task;
