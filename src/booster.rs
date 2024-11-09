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
mod mlpboost;
mod erlpboost;
mod softboost;
mod totalboost;



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
pub use self::totalboost::TotalBoost;


// Soft Margin Maximization
pub use self::lpboost::LPBoost;
pub use self::mlpboost::MLPBoost;
pub use self::erlpboost::ERLPBoost;
pub use self::cerlpboost::CERLPBoost;
pub use self::softboost::SoftBoost;

pub use self::smoothboost::SmoothBoost;




pub use self::gradient_boost::GBM;
pub use self::graph_separation_boosting::GraphSepBoost;

// // ------------------------------------------------
// // Regression
// pub use self::soft_lae::SLBoost;
// pub use self::leveragings::{
//     SquareLevR,
// };
