#![warn(missing_docs)]
#![doc = include_str!("../README.md")]
#![doc = include_str!("../docs/logging.md")]

pub mod boosters;
pub mod core;
pub mod hypotheses;
pub mod logging;
pub mod optimization;
pub mod weak_learners;

pub mod prelude;

pub use crate::core::{Booster, Classifier, Regressor, Sample, SampleReader, WeakLearner};
pub use crate::hypotheses::{NaiveAggregation, WeightedMajority};
pub use crate::logging::*;
pub use crate::optimization::*;

pub use crate::boosters::adaboost::AdaBoost;
pub use crate::boosters::adaboostv::AdaBoostV;
pub use crate::boosters::corrective_erlpboost::CorrectiveErlpBoost;
pub use crate::boosters::erlpboost::ErlpBoost;
pub use crate::boosters::graph_separation_boosting::GraphSeparationBoosting;
pub use crate::boosters::lpboost::LpBoost;
pub use crate::boosters::madaboost::MadaBoost;
pub use crate::boosters::mlpboost::MlpBoost;
pub use crate::boosters::smoothboost::SmoothBoost;
pub use crate::boosters::softboost::SoftBoost;
pub use crate::boosters::sparsiboost::SparsiBoost;
pub use crate::boosters::totalboost::TotalBoost;

pub use crate::weak_learners::decision_tree::{
    DecisionTree, DecisionTreeBuilder, DecisionTreeClassifier, SplitBy, node, split_by,
};
