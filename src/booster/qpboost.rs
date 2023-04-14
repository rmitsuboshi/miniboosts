//! QPBoost module.
//! In each round, QPBoost minimizes the quadratic approximation
//! of ERLPBoost objective, while ERLPBoost minimizes it repeatedly
//! until convergence.

pub mod qpboost_algorithm;
mod qp_model;


pub use qpboost_algorithm::QPBoost;

