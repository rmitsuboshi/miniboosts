//! Combined hypotheses returned by the boosters.
//!
//! - [`WeightedMajority`]: an owned weighted vote `Σ_t w_t h_t(x)`, returned
//!   by every booster except [`GraphSeparationBoosting`](crate::boosters::graph_separation_boosting::GraphSeparationBoosting).
//! - [`RefWeightedMajority`]: the same weighted vote over borrowed slices.
//! - [`NaiveAggregation`]: the lookup rule returned by
//!   [`GraphSeparationBoosting`](crate::boosters::graph_separation_boosting::GraphSeparationBoosting).

pub mod naive_aggregation;
pub mod weighted_majority;

pub use naive_aggregation::NaiveAggregation;
pub use weighted_majority::{RefWeightedMajority, WeightedMajority};
