//! `AdaBoost*_ν` by Rätsch and Warmuth (2005), named [`AdaBoostV`].
//!
//! See [`AdaBoostV`] for the algorithm, guarantees, and an example.
pub mod algorithm;
pub use algorithm::AdaBoostV;
