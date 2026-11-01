//!
//! These are the building blocks used by the totally corrective and
//! Frank-Wolfe boosters ([`LpBoost`](crate::LpBoost),
//! [`ErlpBoost`](crate::ErlpBoost), [`MlpBoost`](crate::MlpBoost),
//! [`CorrectiveErlpBoost`](crate::CorrectiveErlpBoost),
//! [`SoftBoost`](crate::SoftBoost)). They are public for experimentation; most
//! users never call them directly.
//!
//! Notation: `m` examples `(x_i, y_i)` with `y_i ∈ {-1, +1}`, hypotheses
//! `h_1, ..., h_T`, the capping parameter `ν ∈ [1, m]`, and the capped simplex
//! ```txt
//! Δ_{m,ν} = { d ∈ R^m : Σ_i d_i = 1, 0 ≤ d_i ≤ 1/ν }.
//! ```
//! The edge of `h_j` under `d` is `Σ_i d_i y_i h_j(x_i)`, where `h_j(x_i)` is
//! [`Classifier::confidence`](crate::Classifier::confidence).
//!
//! - [`soft_margin_optimization()`] and [`ColumnGeneration`]: the soft-margin
//!   linear program (LPBoost primal) over weights on hypotheses.
//! - [`edge_minimization()`]: the dual linear program, minimizing the maximum
//!   edge over `Δ_{m,ν}`.
//! - [`RowGeneration`] with a [`RowGenerationObjective`] such as
//!   [`EntropyRegularizedMaxEdge`]: the entropy-regularized restricted problem
//!   solved by ERLPBoost.
//! - [`FrankWolfe`] with an [`ObjectiveFunction`]: step-size rules for
//!   Frank-Wolfe updates used by MLPBoost and Corrective ERLPBoost.
//!
//! Linear, quadratic, and conic problems are solved with the
//! [Clarabel](https://docs.rs/clarabel) interior-point solver. Solver failures
//! are reported by **panicking**, not by returning an error; each item
//! documents which solver statuses and residual checks it accepts.
//!
//! # Examples
//! Solve the soft-margin LP and its dual for three fixed hypotheses, then
//! evaluate the resulting ensemble with [`SoftMarginObjective`]. Each
//! hypothesis misclassifies one of examples 2, 3, 4, so the best smallest
//! margin is `1/3`, attained by equal weights.
//! ```
//! use miniboosts::{Classifier, Sample};
//! use miniboosts::{
//!     ObjectiveFunction, SoftMarginObjective, edge_minimization, soft_margin_optimization,
//! };
//!
//! /// A hypothesis with fixed margins `y_i h(x_i)`, one per example.
//! struct Fixed([f64; 4]);
//!
//! impl Classifier for Fixed {
//!     fn confidence(&self, sample: &Sample, row: usize) -> f64 {
//!         self.0[row] * sample.target()[row]
//!     }
//! }
//!
//! let sample = Sample::dummy(4);
//! let hypotheses = [
//!     Fixed([1.0, 1.0, 1.0, -1.0]),
//!     Fixed([1.0, 1.0, -1.0, 1.0]),
//!     Fixed([1.0, -1.0, 1.0, 1.0]),
//! ];
//! let nu = 1.0;
//!
//! // Primal: weights on hypotheses.
//! let (rho, w) = soft_margin_optimization(nu, &sample, &hypotheses);
//! // Dual: a distribution on examples. Both LPs have the same value.
//! let (gamma, d) = edge_minimization(nu, &sample, &hypotheses);
//! // Interior-point solutions; 1e-6 is well above the observed 1e-8 error.
//! assert!((rho - 1.0 / 3.0).abs() < 1e-6 && (gamma - rho).abs() < 1e-6);
//! assert!((d.iter().sum::<f64>() - 1.0).abs() < 1e-6);
//!
//! // The soft-margin objective of the combined hypothesis Σ_j w_j h_j,
//! // in negated-margin coordinates, recovers the LP value.
//! let theta: Vec<f64> = (0..4)
//!     .map(|i| -w.iter().zip(&hypotheses).map(|(wj, h)| wj * h.0[i]).sum::<f64>())
//!     .collect();
//! let value = SoftMarginObjective::new(nu).objective_value(&theta);
//! assert!((value - rho).abs() < 1e-6);
//! ```
pub mod edge_minimization;
mod entropy_certificate;
mod entropy_solver;
pub mod frank_wolfe;
pub mod objective_function;
pub mod soft_margin_optimization;

pub use edge_minimization::{
    EntropyRegularizedMaxEdge, RowGeneration, RowGenerationObjective, edge_minimization,
};
pub use frank_wolfe::{FrankWolfe, FwUpdateRule, StepSize};
pub use objective_function::{
    Entropy, ErlpSoftMarginObjective, ObjectiveFunction, SoftMarginObjective,
};
pub use soft_margin_optimization::{ColumnGeneration, soft_margin_optimization};
