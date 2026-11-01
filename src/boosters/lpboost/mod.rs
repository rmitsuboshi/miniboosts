//! LPBoost: soft-margin boosting by column generation
//! (Demiriz, Bennett, and Shawe-Taylor, 2002).
//!
//! Each round adds the weak learner's hypothesis as a column of the
//! soft-margin linear program, re-solves it, and uses the optimal dual
//! solution as the next distribution over examples. See [`LpBoost`] for the
//! algorithm, its guarantee, limitations, and an example.
pub mod algorithm;
pub use algorithm::LpBoost;
