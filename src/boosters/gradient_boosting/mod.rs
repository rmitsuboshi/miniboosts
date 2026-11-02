//! Gradient boosting by Friedman (2001), named [`GradientBoosting`].
//!
//! See [`GradientBoosting`] for the algorithm, guarantees, and an example.
pub mod algorithm;
pub use algorithm::{GbmLoss, GradientBoosting};
