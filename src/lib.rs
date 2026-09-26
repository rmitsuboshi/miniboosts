#![warn(missing_docs)]
#![doc = include_str!("../README.md")]
#![doc = include_str!("../logging/src/README.md")]

pub mod prelude;

pub use hypotheses::{NaiveAggregation, WeightedMajority};
pub use logging::*;
pub use miniboosts_core::{Booster, Classifier, Regressor, Sample, SampleReader, WeakLearner};
pub use optimization::*;

/// Exponential-loss boosting by Freund and Schapire.
/// `tolerance` sets the iteration budget, not a measured-error stopping test.
/// See [Boosting: Foundations and Algorithms](https://direct.mit.edu/books/oa-monograph/5342/BoostingFoundations-and-Algorithms).
pub use adaboost::AdaBoost;

/// Hard-margin boosting by Rätsch and Warmuth.
/// `tolerance` controls the margin update and iteration budget.
/// See [Efficient Margin Maximizing with Boosting](https://www.jmlr.org/papers/v6/ratsch05a.html).
pub use adaboostv::AdaBoostV;

/// Soft-margin column generation; `tolerance` controls the optimization gap.
/// A guarantee against all hypotheses requires an appropriate weak-learning
/// oracle and accurately solved LPs, not merely a small training loss.
/// See [Linear Programming Boosting via Column Generation](https://link.springer.com/content/pdf/10.1023/A:1012470815092.pdf).
pub use lpboost::LpBoost;

/// Entropy-regularized soft-margin boosting. The tolerance is split between
/// regularization accuracy and the optimization stopping criterion.
/// See [Entropy Regularized LPBoost](https://www.stat.purdue.edu/~vishy/papers/WarGloVis08.pdf).
pub use erlpboost::ErlpBoost;

/// Graph-separation boosting using the aggregation rule in Lemma 4.2 of
/// [Boosting Simple Learners](https://theoretics.episciences.org/10757).
pub use graph_separation_boosting::GraphSeparationBoosting;

/// Corrective entropy-regularized boosting using Frank-Wolfe updates.
/// `tolerance` controls regularization and the duality-gap stopping criterion.
/// See [On the equivalence of weak learnability and linear separability](https://link.springer.com/article/10.1007/s10994-010-5173-z).
pub use corrective_erlpboost::CorrectiveErlpBoost;

/// MadaBoost with capped-product example weights.
/// `tolerance` sets the iteration budget; it is not a measured-error threshold.
/// See [MadaBoost: A Modification of AdaBoost](https://www.learningtheory.org/colt2000/papers/DomingoWatanabe.pdf).
pub use madaboost::MadaBoost;

/// Soft-margin boosting combining Frank-Wolfe and LP candidate updates.
/// `tolerance` controls regularization and optimization accuracy.
/// See [Boosting as Frank-Wolfe](https://arxiv.org/abs/2209.10831).
pub use mlpboost::MlpBoost;

/// Smooth boosting requiring a weak-learning advantage `gamma` on each
/// requested distribution. `kappa` specifies the target error under that
/// assumption; choosing a tree depth does not establish the assumption.
/// See Figure 1 of [Smooth Boosting and Learning with Malicious Noise](https://link.springer.com/chapter/10.1007/3-540-44581-1_31).
pub use smoothboost::SmoothBoost;

/// Soft-margin boosting with entropy projection implemented through exponential cones. `tolerance` is an optimization parameter, not a bound on the
/// observed fraction of classification errors.
/// See [Boosting Algorithms for Maximizing the Soft Margin](https://proceedings.neurips.cc/paper/2007/file/cfbce4c1d7c425baf21d6b6f2babe6be-Paper.pdf).
pub use softboost::SoftBoost;

/// Hard-margin specialization of [`SoftBoost`] with `nu = 1`.
/// See [Totally Corrective Boosting Algorithms That Maximize the Margin](https://dl.acm.org/doi/10.1145/1143844.1143970).
pub use totalboost::TotalBoost;

pub use decision_tree::*;
