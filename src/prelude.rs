//! This module re-exports frequently used structs and traits.
//!
//! `use miniboosts::prelude::*;` imports
//! - the traits [`Booster`], [`WeakLearner`], [`Classifier`], and the file
//!   reader [`SampleReader`];
//! - every booster ([`AdaBoost`], [`AdaBoostV`], [`BrownBoost`],
//!   [`CorrectiveErlpBoost`], [`ErlpBoost`], [`GraphSeparationBoosting`],
//!   [`LpBoost`], [`MadaBoost`], [`MlpBoost`], [`SmoothBoost`],
//!   [`SoftBoost`], [`SparsiBoost`], [`TotalBoost`]);
//! - the weak learners' entry points: the decision tree's
//!   [`DecisionTreeBuilder`] and [`SplitBy`], [`FiniteClass`],
//!   [`RbfClassBuilder`], [`BoundedLinear`], [`LogisticRegression`] with
//!   [`LinearOutput`], and [`NaiveBayes`].
//!
//! It does **not** import [`Sample`](crate::Sample),
//! [`Regressor`](crate::Regressor), the combined hypotheses
//! ([`WeightedMajority`](crate::WeightedMajority),
//! [`NaiveAggregation`](crate::NaiveAggregation)), or the logging types;
//! import those from the crate root.
//!
//! # Examples
//! ```
//! use std::io::BufReader;
//! use miniboosts::prelude::*;
//! use miniboosts::Sample; // not part of the prelude
//!
//! let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
//! let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
//!     .unwrap()
//!     .set_target("class");
//!
//! // `DecisionTreeBuilder`, `SplitBy`, `AdaBoost`, and the traits providing
//! // `run` (`Booster`) and `predict_all` (`Classifier`) all come from the
//! // prelude.
//! let tree = DecisionTreeBuilder::new(&sample)
//!     .max_depth(1)
//!     .split_by(SplitBy::Entropy)
//!     .build();
//! let f = AdaBoost::init(&sample).tolerance(0.1).run(&tree);
//! assert_eq!(f.predict_all(&sample), vec![-1, -1, -1, 1, 1, 1]);
//!
//! // Any other booster is used the same way, e.g. LPBoost (capping
//! // parameter `nu = 1`, i.e. hard margin).
//! let f = LpBoost::init(&sample).nu(1.0).run(&tree);
//! assert_eq!(f.predict_all(&sample), vec![-1, -1, -1, 1, 1, 1]);
//! ```

pub use crate::core::{Booster, Classifier, SampleReader, WeakLearner};

pub use crate::boosters::adaboost::AdaBoost;
pub use crate::boosters::adaboostv::AdaBoostV;
pub use crate::boosters::brownboost::BrownBoost;
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
pub use crate::weak_learners::decision_tree::{DecisionTreeBuilder, SplitBy};
pub use crate::weak_learners::finite_class::FiniteClass;
pub use crate::weak_learners::linear::{BoundedLinear, LinearOutput, LogisticRegression};
pub use crate::weak_learners::naive_bayes::NaiveBayes;
pub use crate::weak_learners::rbf::RbfClassBuilder;
