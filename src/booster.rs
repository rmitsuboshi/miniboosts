//! Provides some boosting algorithms.

pub mod core;

// ------------------------------------------------
// Classification
pub mod smoothboost;
pub mod adaboost;
pub mod adaboostv;

#[cfg(feature="extended")]
pub mod totalboost;

pub mod cerlpboost;
#[cfg(feature="extended")]
pub mod lpboost;
#[cfg(feature="extended")]
pub mod erlpboost;
#[cfg(feature="extended")]
pub mod softboost;
#[cfg(feature="extended")]
pub mod mlpboost;

pub mod gradient_boost;
pub mod graph_separation_boosting;


/// Booster trait
pub use self::core::Booster;

// ------------------------------------------------
// Regression

// ------------------------------------------------
// Classification

/// Empirical Risk Minimization
pub use self::adaboost::AdaBoost;


/// Hard Margin Maximization
pub use self::adaboostv::AdaBoostV;
#[cfg(feature="extended")]
pub use self::totalboost::TotalBoost;


/// Soft Margin Maximization
#[cfg(feature="extended")]
pub use self::lpboost::LPBoost;
#[cfg(feature="extended")]
pub use self::erlpboost::ERLPBoost;
#[cfg(feature="extended")]
pub use self::softboost::SoftBoost;

pub use self::smoothboost::SmoothBoost;

pub use self::cerlpboost::CERLPBoost;

#[cfg(feature="extended")]
pub use self::mlpboost::MLPBoost;


pub use self::gradient_boost::GBM;
pub use self::graph_separation_boosting::GraphSepBoost;

// // ------------------------------------------------
// // Regression
// pub use self::soft_lae::SLBoost;
// pub use self::leveragings::{
//     SquareLevR,
// };
