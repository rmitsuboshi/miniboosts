//! ERLPBoost module.

pub mod erlpboost_algorithm;

#[cfg(not(feature="gurobi"))]
mod qp_model;

#[cfg(feature="gurobi")]
mod gurobi_qp_model;

pub use erlpboost_algorithm::ERLPBoost;

