//! RobustBoost by Freund (2009), named [`RobustBoost`].
//!
//! See [`RobustBoost`] for the algorithm, guarantees, and an example.
pub mod algorithm;
mod solver;
pub use algorithm::RobustBoost;
