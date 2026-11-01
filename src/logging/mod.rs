//!
//! [`Logger`] runs a booster one step at a time and, after every step,
//! evaluates the current ensemble (via [`CurrentHypothesis`]) and writes one
//! CSV row with the objective value on the training sample, the training and
//! test loss, and the cumulative boosting time. Build a logger with
//! [`LoggerBuilder`]; supply the objective as a [`LoggingObjective`] (for
//! example [`LoggingSoftMarginObjective`]) and the loss as a closure
//! `Fn(&Sample, &S) -> f64`.
//!
//! The crate-level documentation describes the CSV columns and exactly what
//! the `Time` column measures. In short: it is the cumulative time spent in
//! [`Booster::boost`](crate::Booster::boost), in milliseconds; evaluation,
//! printing, and file I/O are excluded, and the time limit is checked only
//! between steps. The test sample is used for reporting only.
//!
//! [`Sample`]: crate::Sample
/// [`LoggerBuilder`], which assembles a [`Logger`].
pub mod builder;
/// [`Logger`], which runs a booster step by step and writes a CSV log.
pub mod logger;
/// Objective values recorded in the log ([`LoggingObjective`]).
pub mod objective;

pub use logger::{CurrentHypothesis, Logger};

pub use builder::LoggerBuilder;

pub use objective::{LoggingObjective, LoggingSoftMarginObjective};
