//! LPBoost module.
pub mod lpboost_algorithm;

#[cfg(not(feature="gurobi"))]
mod lp_model;

#[cfg(feature="gurobi")]
mod gurobi_lp_model;

pub use lpboost_algorithm::LPBoost;
