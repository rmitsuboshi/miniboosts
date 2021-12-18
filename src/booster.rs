pub mod core;
pub mod adaboost;
pub mod adaboostv;
pub mod lpboost;
pub mod erlpboost;
pub mod softboost;
pub mod totalboost;

/// 
/// Export the Boosters

/// Booster trait
pub use self::core::Booster;

/// Empirical Risk Minimization
pub use adaboost::AdaBoost;

/// Hard Margin Maximization
pub use totalboost::TotalBoost;

/// Soft Margin Maximization
pub use lpboost::LPBoost;
pub use erlpboost::ERLPBoost;
pub use softboost::SoftBoost;
