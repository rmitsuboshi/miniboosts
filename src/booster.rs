//! Provides some boosting algorithms.

mod core;

// ------------------------------------------------
// Classification
mod smoothboost;
mod adaboost;
mod adaboostv;
mod cerlpboost;
mod gradient_boost;
mod graph_separation_boosting;
mod madaboost;
// mod branching_program;
mod lpboost;

#[cfg(feature="gurobi")]
mod totalboost;

#[cfg(feature="gurobi")]
mod erlpboost;
#[cfg(feature="gurobi")]
mod softboost;
#[cfg(feature="gurobi")]
mod mlpboost;



/// Booster trait
pub use self::core::Booster;

// ------------------------------------------------
// Regression

// ------------------------------------------------
// Classification

// Empirical Risk Minimization
pub use self::adaboost::AdaBoost;
pub use self::madaboost::MadaBoost;


// Hard Margin Maximization
pub use self::adaboostv::AdaBoostV;
#[cfg(feature="gurobi")]
pub use self::totalboost::TotalBoost;


// Soft Margin Maximization
pub use self::lpboost::LPBoost;
#[cfg(feature="gurobi")]
pub use self::erlpboost::ERLPBoost;
#[cfg(feature="gurobi")]
pub use self::softboost::SoftBoost;

pub use self::smoothboost::SmoothBoost;

pub use self::cerlpboost::CERLPBoost;

#[cfg(feature="gurobi")]
pub use self::mlpboost::MLPBoost;


pub use self::gradient_boost::GBM;
pub use self::graph_separation_boosting::GraphSepBoost;

// // ------------------------------------------------
// // Regression
// pub use self::soft_lae::SLBoost;
// pub use self::leveragings::{
//     SquareLevR,
// };
