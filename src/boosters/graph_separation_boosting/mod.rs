//! Graph Separation Boosting (Alon, Gonen, Hazan, and Moran).
//!
//! [`GraphSeparationBoosting`] collects weak hypotheses until every pair of
//! oppositely labeled training examples is separated by at least one of
//! them, then aggregates them with the lookup rule of
//! [`NaiveAggregation`](crate::hypotheses::NaiveAggregation) instead of a
//! weighted vote. It has no tolerance or iteration-budget parameter.

pub mod algorithm;
pub use algorithm::GraphSeparationBoosting;
