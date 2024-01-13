//! The core library for `Hypothesis` traits.

pub(crate) mod hypothesis_traits;
pub(crate) mod weighted_majority;
pub(crate) mod naive_aggregation;


pub use hypothesis_traits::{
    Classifier,
    Regressor,
};

pub use weighted_majority::WeightedMajority;
pub use naive_aggregation::NaiveAggregation;


