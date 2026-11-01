//! Corrective ERLPBoost: Frank-Wolfe soft-margin boosting
//! (Shalev-Shwartz and Singer, 2010).
//!
//! Each round moves the weights on hypotheses toward the new hypothesis by a
//! Frank-Wolfe step on the entropy-regularized soft-margin objective, without
//! solving a linear or convex program. See [`CorrectiveErlpBoost`] for the
//! algorithm, its guarantee, limitations, and an example, and
//! [`CorrErlpFwObjective`] for the objective passed to the Frank-Wolfe solver.
pub mod algorithm;
pub use algorithm::{CorrErlpFwObjective, CorrectiveErlpBoost};
