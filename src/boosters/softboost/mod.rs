//! SoftBoost (Warmuth, Glocer, and Rätsch, 2007).
//!
//! [`SoftBoost`] is a totally corrective booster that approximately maximizes
//! the soft margin with capping parameter `nu`. Each round projects the
//! uniform distribution, in relative entropy, onto capped distributions under
//! which every collected hypothesis has a small edge. It stops after at most
//! `⌈2 ln(m / nu) / tolerance²⌉` rounds, or earlier when the projection is
//! infeasible, has a zero entry, or fails numerically (see
//! [`SoftBoost::numerical_stop_reason`](crate::boosters::softboost::SoftBoost::numerical_stop_reason)).

pub mod algorithm;
mod solver;

pub use algorithm::SoftBoost;
