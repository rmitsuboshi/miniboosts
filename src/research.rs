//! This directory provides some features for research  
//! Measure the followings of boosting algorithm per iteration
//! - Running time
//! - Objective value
//! - Training loss
//! - Test loss


/// Defines a trait for logging.
pub mod logger;

pub use logger::{
    Logger,
    Research,
};

/// Defines objective functions and its traits.
pub mod objective_functions;
pub use objective_functions::ObjectiveFunction;
