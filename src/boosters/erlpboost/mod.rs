//! ERLPBoost: entropy-regularized LPBoost
//! (Warmuth, Glocer, and Vishwanathan, 2008).
//!
//! Each round solves an entropy-regularized version of LPBoost's edge
//! minimization over the hypotheses collected so far, which yields an
//! `O(ln(m/ν)/ε²)` iteration bound. See [`ErlpBoost`] for the algorithm, its
//! guarantee, limitations, and an example.
mod algorithm;
pub use algorithm::ErlpBoost;
