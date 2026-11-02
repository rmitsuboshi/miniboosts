//! Combined hypotheses returned by the boosters.
//!
//! - [`WeightedMajority`]: an owned weighted vote `Σ_t w_t h_t(x)`, returned
//!   by every booster except [`GraphSeparationBoosting`](crate::boosters::graph_separation_boosting::GraphSeparationBoosting).
//! - [`RefWeightedMajority`]: the same weighted vote over borrowed slices.
//! - [`AdditiveModel`]: an additive model `c_0 + Σ_t c_t f_t(x)` of real-valued
//!   hypotheses, returned by [`GradientBoosting`](crate::GradientBoosting)
//!   and [`LogitBoost`](crate::LogitBoost).
//! - [`MajorityVote`]: an unweighted vote of predicted labels, returned by
//!   [`BoostByMajority`](crate::BoostByMajority).
//! - [`NaiveAggregation`]: the lookup rule returned by
//!   [`GraphSeparationBoosting`](crate::boosters::graph_separation_boosting::GraphSeparationBoosting).

pub mod additive_model;
pub mod majority_vote;
pub mod naive_aggregation;
pub mod weighted_majority;

pub use additive_model::{AdditiveModel, ScoreLink};
pub use majority_vote::MajorityVote;
pub use naive_aggregation::NaiveAggregation;
pub use weighted_majority::{RefWeightedMajority, WeightedMajority};
