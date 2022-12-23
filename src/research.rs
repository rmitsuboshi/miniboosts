//! This directory provides some features for research  
//! Measure the followings of boosting algorithm per iteration
//! - Running time
//! - Objective value
//! - Training loss
//! - Test loss

/// Provides an algorithm that runs a boosting algorithm with logging.
pub mod boost_logger;

/// Defines a trait for logging.
pub mod logger;

/// Defines loss functions (e.g., zero-one loss, squared loss).
pub mod loss_functions;

/// Defines objective functions.
pub mod objective_functions;

pub use logger::Logger;
pub use objective_functions::{
    soft_margin_objective,
};


pub use boost_logger::{
    with_log,
};

pub use loss_functions::{
    zero_one_loss,
    squared_loss,
    absolute_loss,
};
