//! Boosting algorithms.
//!
//! Every booster implements [`Booster`](crate::core::Booster) and works with
//! any [`WeakLearner`](crate::core::WeakLearner). Each booster is built from a
//! training [`Sample`](crate::core::Sample) with `init`, configured with
//! builder methods, and trained with [`Booster::run`](crate::core::Booster::run).
//! All of them solve binary classification with labels `-1` and `+1`.
//!
//! # Overview
//!
//! In the table, `m` is the number of training examples, `ν` (`nu`) caps each
//! example weight at `1/ν` with `1 ≤ ν ≤ m`, and `ε` is `tolerance`. The
//! default `tolerance` is `1e-3` and the default `nu` is `1.0` (no capping).
//!
//! | Booster | Kind | Key parameters | Stopping rule |
//! | --- | --- | --- | --- |
//! | [`AdaBoost`](crate::boosters::adaboost::AdaBoost) | adaptive | `tolerance`, `force_quit_at` | budget `⌈ln(m) / ε²⌉` rounds, or a hypothesis with weighted edge `±1` |
//! | [`AdaBoostV`](crate::boosters::adaboostv::AdaBoostV) | adaptive, hard-margin | `tolerance`, `force_quit_at` | budget `⌊2 ln(m) / ε²⌋` rounds, or a hypothesis with weighted edge `±1` |
//! | [`ArcGv`](crate::boosters::arcgv::ArcGv) | hard-margin (game value) | `force_quit_at` (required) | `Δ_k = 0` (edge at most the current minimal margin), or the round limit |
//! | [`BoostByMajority`](crate::boosters::bbm::BoostByMajority) | non-adaptive, unweighted majority | `gamma` (required), `tolerance` (target training error `ε`) | `k = ⌈ln(1/ε) / (2γ²)⌉` rounds, or every vote decided |
//! | [`BrownBoost`](crate::boosters::brownboost::BrownBoost) | adaptive boost-by-majority (noise-tolerant) | `tolerance` (target training error `ε`), `edge_threshold`, `force_quit_at` | remaining time `s` (initially `c = (erfc⁻¹(ε))²`) reaches `0`; no iteration bound otherwise |
//! | [`RobustBoost`](crate::boosters::robustboost::RobustBoost) | BrownBoost with a non-convex, margin-goal potential | `tolerance` (`ε`), `theta`, `sigma_f`, `force_quit_at` | time `t` reaches `1`; no iteration bound otherwise |
//! | [`MadaBoost`](crate::boosters::madaboost::MadaBoost) | adaptive (bounded weights) | `tolerance`, `force_quit_at` | budget `⌈(m - 1) / ε²⌉` rounds, or a hypothesis with weighted edge `±1` |
//! | [`LpBoost`](crate::boosters::lpboost::LpBoost) | soft-margin, totally corrective | `nu`, `tolerance` | LP value `≥ γ̂ - ε` (min edge so far); no iteration bound |
//! | [`ErlpBoost`](crate::boosters::erlpboost::ErlpBoost) | soft-margin, totally corrective | `nu`, `tolerance` | entropy-regularized gap `≤ ε/2`, or a `⌈max(8/ε, 32 ln(m/ν) / ε²)⌉` budget |
//! | [`CorrectiveErlpBoost`](crate::boosters::corrective_erlpboost::CorrectiveErlpBoost) | soft-margin, corrective (Frank–Wolfe) | `nu`, `tolerance`, `update_rule` | duality gap `≤ ε/2`, `ν = m` (after one round), or a `⌈32 ln(m/ν) / ε²⌉` budget |
//! | [`MlpBoost`](crate::boosters::mlpboost::MlpBoost) | soft-margin, Frank–Wolfe with LP steps | `nu`, `tolerance`, `update_rule` | optimality gap `≤ ε/2`, `ν = m` (after one round), or a `⌈32 ln(m/ν) / ε²⌉` budget |
//! | [`SmoothBoost`](crate::boosters::smoothboost::SmoothBoost) | smooth | `kappa`, `gamma` | total example weight below `κ m`, or a `⌈2 / (κ γ² √(1 - γ))⌉` budget |
//! | [`SparsiBoost`](crate::boosters::sparsiboost::SparsiBoost) | hard-margin, AdaBoostV then sparsification | `n_hypotheses` (`T`), `tolerance` (gap `v`), `seed`; at least one of `T`, `v` | AdaBoostV for `⌈8 ln m / v²⌉` rounds if `v` is set, else `c T` with `c = ⌈lg m / lg(2 + m/T)⌉` (or a hypothesis with weighted edge `±1`); then halving until margins would move by more than `v/2`, or until at most `T` hypotheses remain (`T` takes precedence) |
//! | [`SoftBoost`](crate::boosters::softboost::SoftBoost) | soft-margin, totally corrective | `nu`, `tolerance` | infeasible or zero-entry entropy projection, solver failure, or a `⌈2 ln(m/ν) / ε²⌉` budget |
//! | [`TotalBoost`](crate::boosters::totalboost::TotalBoost) | hard-margin, totally corrective | `tolerance` | as SoftBoost with `ν = 1` |
//! | [`GraphSeparationBoosting`](crate::boosters::graph_separation_boosting::GraphSeparationBoosting) | sample-compression (consistent aggregation) | none | every oppositely labeled pair separated, or a round separates no new pair |
//! | [`GradientBoosting`](crate::boosters::gradient_boosting::GradientBoosting) | functional gradient descent (regression weak learner) | `loss`, `shrinkage`, `force_quit_at` (required) | the round limit |
//! | [`LogitBoost`](crate::boosters::logitboost::LogitBoost) | Newton steps on the log-likelihood (regression weak learner) | `max_response`, `force_quit_at` (required) | the round limit |
//!
//! Most boosters return a [`WeightedMajority`](crate::hypotheses::WeightedMajority).
//! [`GraphSeparationBoosting`](crate::boosters::graph_separation_boosting::GraphSeparationBoosting)
//! returns a [`NaiveAggregation`](crate::hypotheses::NaiveAggregation),
//! [`BoostByMajority`](crate::boosters::bbm::BoostByMajority) a
//! [`MajorityVote`](crate::hypotheses::MajorityVote), and
//! [`GradientBoosting`](crate::boosters::gradient_boosting::GradientBoosting) and
//! [`LogitBoost`](crate::boosters::logitboost::LogitBoost) an
//! [`AdditiveModel`](crate::hypotheses::AdditiveModel); the last two need a
//! weak learner that returns a [`Regressor`](crate::Regressor) and accepts
//! real-valued targets.
//!
//! `tolerance` is algorithm-specific: for the adaptive methods it sets an
//! iteration budget, and for the margin-maximizing methods it is an
//! optimization gap. In neither case is it a bound on measured
//! classification error, and equal values do not imply equal accuracy across
//! algorithms. The theoretical guarantees documented on each booster require
//! that booster's weak-learning (or oracle) assumptions and accurate
//! subproblem solutions; they are not checked at runtime. See each booster's
//! documentation for details and the paper it follows.
//!
//! # Examples
//! A fair comparison runs every booster on the same folds with the same weak
//! learner and the same metric, and states each booster's stopping rule
//! (see the README). Here, 3-fold cross-validation over a fixed permutation
//! gives every booster identical training and test sets; the decision stumps
//! are built from the training fold only, so test data never influence
//! training or binning.
//! ```
//! use std::io::BufReader;
//! use miniboosts::{FwUpdateRule, Sample, prelude::*};
//!
//! /// Fraction of `sample` whose label `f` predicts correctly.
//! fn accuracy<C: Classifier>(f: &C, sample: &Sample) -> f64 {
//!     let correct = f
//!         .predict_all(sample)
//!         .iter()
//!         .zip(sample.target())
//!         .filter(|&(&p, &y)| p as f64 == y)
//!         .count();
//!     correct as f64 / sample.shape().0 as f64
//! }
//!
//! // Labels follow the sign of `x + y`, so no single stump separates them.
//! let csv = b"x,y,class\n\
//!     -2,1,-1\n-1,-2,-1\n1,-3,-1\n-3,2,-1\n0,-1,-1\n-2,-1,-1\n\
//!     2,1,1\n1,2,1\n-1,3,1\n3,-1,1\n0,2,1\n2,2,1\n";
//! let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
//!     .unwrap()
//!     .set_target("class");
//!
//! // One fixed permutation (record it, or the seed that produced it);
//! // fold k tests on ix[4k..4k + 4], and each fold has both labels.
//! let ix = [0, 6, 4, 10, 1, 7, 2, 8, 3, 9, 5, 11];
//! let names = ["AdaBoost", "LpBoost", "MlpBoost", "SoftBoost", "SmoothBoost"];
//! let mut test_accuracy = vec![Vec::new(); names.len()];
//!
//! for k in 0..3 {
//!     let (train, test) = sample.split(&ix, 4 * k, 4 * k + 4);
//!     // The same weak learner for every booster.
//!     let learner = DecisionTreeBuilder::new(&train).max_depth(1).build();
//!
//!     // Stopping rules differ: AdaBoost runs a fixed number of rounds,
//!     // LpBoost/MlpBoost/SoftBoost stop on a soft-margin gap (`tolerance`),
//!     // and SmoothBoost stops once the total example weight is below κ m.
//!     // Equal tolerances therefore do not imply equal accuracy.
//!     let f = AdaBoost::init(&train).force_quit_at(50).run(&learner);
//!     test_accuracy[0].push(accuracy(&f, &test));
//!     let f = LpBoost::init(&train).nu(2.0).tolerance(0.1).run(&learner);
//!     test_accuracy[1].push(accuracy(&f, &test));
//!     let f = MlpBoost::init(&train)
//!         .nu(2.0)
//!         .tolerance(0.1)
//!         .update_rule(FwUpdateRule::LineSearch)
//!         .run(&learner);
//!     test_accuracy[2].push(accuracy(&f, &test));
//!     let mut softboost = SoftBoost::init(&train).nu(2.0).tolerance(0.1);
//!     let f = softboost.run(&learner);
//!     // Do not report a numerically failed run as a converged result.
//!     if let Some(reason) = softboost.numerical_stop_reason() {
//!         eprintln!("fold {k}: SoftBoost stopped early: {reason}");
//!     }
//!     test_accuracy[3].push(accuracy(&f, &test));
//!     let f = SmoothBoost::init(&train).kappa(0.25).gamma(0.2).run(&learner);
//!     test_accuracy[4].push(accuracy(&f, &test));
//! }
//!
//! for (name, accs) in names.iter().zip(&test_accuracy) {
//!     assert_eq!(accs.len(), 3);
//!     assert!(accs.iter().all(|a| (0.0..=1.0).contains(a)));
//!     let mean = accs.iter().sum::<f64>() / accs.len() as f64;
//!     println!("{name:>12}: mean test accuracy {mean:.3} over 3 folds");
//! }
//! ```

pub mod adaboost;
pub mod adaboostv;
pub mod arcgv;
pub mod bbm;
pub mod brownboost;
pub mod corrective_erlpboost;
pub mod erlpboost;
pub mod gradient_boosting;
pub mod graph_separation_boosting;
pub mod logitboost;
pub mod lpboost;
pub mod madaboost;
pub mod mlpboost;
pub mod robustboost;
pub mod smoothboost;
pub mod softboost;
pub mod sparsiboost;
pub mod totalboost;
