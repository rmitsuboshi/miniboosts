//! This module re-exports frequently used structs and traits.

pub use miniboosts_core::{
    SampleReader,
    Booster,
    WeakLearner,
    Classifier,
};

pub use adaboost::AdaBoost;
pub use adaboostv::AdaBoostV;
pub use corrective_erlpboost::CorrectiveErlpBoost;
pub use erlpboost::ErlpBoost;
pub use graph_separation_boosting::GraphSeparationBoosting;
pub use lpboost::LpBoost;
pub use madaboost::MadaBoost;
pub use mlpboost::MlpBoost;
pub use smoothboost::SmoothBoost;
pub use softboost::SoftBoost;
pub use totalboost::TotalBoost;
pub use decision_tree::{
    DecisionTreeBuilder,
    SplitBy,
};

