//! MLPBoost: boosting as Frank-Wolfe with an LPBoost secondary update
//! (Mitsuboshi, Hatano, and Takimoto, "Boosting as Frank-Wolfe").
//!
//! Each round computes a Frank-Wolfe candidate and an LPBoost candidate for
//! the weights on hypotheses and keeps whichever has the better
//! entropy-regularized soft-margin objective. See [`MlpBoost`] for the
//! algorithm, its guarantee, limitations, and an example.
pub mod algorithm;
pub use algorithm::MlpBoost;
