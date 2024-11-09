//! `SoftBoost`.
//! This algorithm is based on this paper: 
//! [Boosting Algorithms for Maximizing the Soft Margin](https://papers.nips.cc/paper/2007/hash/cfbce4c1d7c425baf21d6b6f2babe6be-Abstract.html) 
//! by Gunnar Rätsch, Manfred K. Warmuth, and Laren A. Glocer.
pub mod softboost_algorithm;

#[cfg(not(feature="gurobi"))]
mod qp_model;

#[cfg(feature="gurobi")]
mod gurobi_qp_model;

pub use softboost_algorithm::SoftBoost;
