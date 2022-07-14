//! Provides some boosting algorithms.

pub mod core;
pub mod adaboost;
pub mod adaboostv;
pub mod lpboost;
pub mod cerlpboost;
pub mod erlpboost;
pub mod softboost;
pub mod totalboost;

/// 
/// Export the Boosters

/// Booster trait
pub use self::core::Booster;

/// Empirical Risk Minimization
pub use self::adaboost::AdaBoost;


/// Hard Margin Maximization
pub use self::adaboostv::AdaBoostV;
pub use self::totalboost::TotalBoost;


/// Soft Margin Maximization
pub use self::lpboost::LPBoost;
pub use self::erlpboost::ERLPBoost;
pub use self::softboost::SoftBoost;
pub use self::cerlpboost::CERLPBoost;
