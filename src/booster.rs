//! Provides some boosting algorithms.

mod core;

// ------------------------------------------------
// Classification
mod smoothboost;
mod adaboost;
mod adaboostv;
mod madaboost;
// mod adaboostl;

#[cfg(feature="extended")]
mod totalboost;

mod cerlpboost;
#[cfg(feature="extended")]
mod lpboost;
#[cfg(feature="extended")]
mod erlpboost;
#[cfg(feature="extended")]
mod softboost;
#[cfg(feature="extended")]
mod mlpboost;
// #[cfg(feature="extended")]
// pub mod perturbed_lpboost;

mod gradient_boost;
mod graph_separation_boosting;


/// Booster trait
pub use self::core::Booster;

// ------------------------------------------------
// Regression

// ------------------------------------------------
// Classification

// Empirical Risk Minimization
pub use self::adaboost::AdaBoost;
pub use self::madaboost::MadaBoost;
// pub use self::adaboostl::AdaBoostL;


// Hard Margin Maximization
pub use self::adaboostv::AdaBoostV;
#[cfg(feature="extended")]
pub use self::totalboost::TotalBoost;


// Soft Margin Maximization
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
