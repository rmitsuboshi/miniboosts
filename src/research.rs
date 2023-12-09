//! This directory provides some features for research  
//! Measure the followings of boosting algorithm per iteration
//! - Running time
//! - Objective value
//! - Training loss
//! - Test loss


/// Defines a trait for logging.
pub mod logger;
/// Defines the logger builder.
pub mod logger_builder;

pub use logger::{
    Logger,
    Research,
};

pub use logger_builder::LoggerBuilder;

/// Defines objective functions and its traits.
pub mod objective_functions;
pub use objective_functions::ObjectiveFunction;
