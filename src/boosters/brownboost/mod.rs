//! BrownBoost by Freund (2001), named [`BrownBoost`].
//!
//! See [`BrownBoost`] for the algorithm, guarantees, and an example.
pub mod algorithm;
mod solver;
pub use algorithm::BrownBoost;
