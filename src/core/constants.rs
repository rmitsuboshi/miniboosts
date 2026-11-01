//! Default parameters and numerical tolerances shared across the crate.
//!
//! The tolerances here are implementation constants, not accuracy guarantees.
//! Equal tolerance values do not imply equal accuracy across algorithms; see
//! each booster's documentation for what its `tolerance` parameter means.

/// Interval-width stopping threshold for the bisection line search in
/// [`crate::optimization`] (Frank–Wolfe step-size search).
pub const BINARY_SEARCH_TOLERANCE: f64 = 1e-10;
/// Default `tolerance` used by several boosters (for example
/// [`crate::LpBoost`] and [`crate::MadaBoost`]) when the user does not set one.
/// Its meaning is algorithm-specific: an iteration budget for some boosters,
/// an optimization stopping criterion for others.
pub const DEFAULT_TOLERANCE: f64 = 1e-3;
/// Default capping parameter `nu` for soft-margin boosters. With `nu = 1`
/// the per-example cap `1 / nu` is inactive, i.e., the hard-margin setting.
pub const DEFAULT_CAPPING: f64 = 1f64;
/// Slack allowed when checking upper bounds such as `d[i] <= 1 / nu`
/// (see [`crate::core::checkers::capped_simplex_condition`]).
pub const NUMERIC_TOLERANCE: f64 = 1e-5;
/// Allowed absolute deviation of `sum(d)` from `1` when checking that a
/// vector lies on the probability simplex
/// (see [`crate::core::checkers::capped_simplex_condition`]).
pub const SIMPLEX_TOLERANCE: f64 = 1e-5;
/// Small additive perturbation: widens a constant feature's range in
/// [`crate::core::binning::Bins::cut`] and keeps logarithms/reciprocals of
/// zero weights finite in some solvers.
pub const PERTURBATION: f64 = 1e-10;
/// Objective-decrease threshold used as a stopping criterion by the
/// sequential quadratic programming solver in [`crate::optimization`].
pub const SQP_TOLERANCE: f64 = 1e-10;

/// Initial capacity reserved for each [`crate::core::Feature`]'s value buffer.
/// It is a capacity hint only and does not limit the number of values.
pub const BUFFER_SIZE: usize = 256;
/// Maximum number of bins printed in full by the `Display` implementation of
/// [`crate::core::binning::Bins`]; longer lists are abbreviated with `...`.
pub const PRINT_WIDTH_BINNING: usize = 3;
