//! QPBoost module.
//! In each round, QPBoost minimizes the quadratic approximation
//! of ERLPBoost objective, while ERLPBoost minimizes it repeatedly
//! until convergence.

pub mod qpb;
mod qp_model;
mod utils;


pub use qpb::QPBoost;

