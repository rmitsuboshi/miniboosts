//! TotalBoost (Warmuth, Liao, and Rätsch, 2006).
//!
//! [`TotalBoost`] is a totally corrective booster that approximately
//! maximizes the hard margin. It is implemented as
//! [`SoftBoost`](crate::boosters::softboost::SoftBoost) with `nu = 1`
//! (no capping), so it shares SoftBoost's stopping rules, iteration budget
//! `⌈2 ln(m) / tolerance²⌉`, and early stop on entropy-solver failures (see
//! [`TotalBoost::numerical_stop_reason`](crate::boosters::totalboost::TotalBoost::numerical_stop_reason)).

pub mod algorithm;
pub use algorithm::TotalBoost;
