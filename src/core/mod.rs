//!
//! This module holds everything that boosters and weak learners share:
//!
//! - [`Sample`], [`Feature`], and
//!   [`SampleReader`]: column-oriented training
//!   data and file readers (CSV and SVMlight).
//! - [`Booster`] and
//!   [`WeakLearner`]: the two players of the
//!   boosting protocol.
//! - [`Classifier`] and
//!   [`Regressor`]: prediction traits for hypotheses.
//! - [`CurrentHypothesis`]: a snapshot of a
//!   booster's model during training, used by [`crate::Logger`].
//! - [`constants`]: default parameters and numerical
//!   tolerances.
//! - [`binning`], [`checkers`],
//!   [`helpers`], [`tree`]:
//!   numerical and tree-building utilities shared by the implementations.
//!
//! Binary classification throughout the crate assumes labels `-1` and `+1`
//! stored as `f64` in [`Sample::target`](crate::core::Sample::target).
//
// Links use full paths because this text is combined with the outer doc
// comment on `pub mod core` in `lib.rs` and resolved from the crate root.

pub mod booster;
pub mod constants;
pub mod hypothesis;
pub mod sample;
pub mod tools;
pub mod weak_learner;

pub use tools::{binning, checkers, helpers, tree};

pub use sample::{Feature, Sample, SampleReader};

pub use weak_learner::WeakLearner;

pub use booster::Booster;

pub use hypothesis::{Classifier, Regressor};

/// A snapshot of a booster's current model, used by experiment loggers.
///
/// [`crate::Logger`] calls [`CurrentHypothesis::current_hypothesis`] between
/// boosting rounds to evaluate training/test loss and the objective value
/// without finishing the run.
///
/// Implementations must preserve the final model's prediction semantics:
/// after the last round, the snapshot should predict exactly as the
/// [`Booster::Output`] returned by [`Booster::postprocess`] would.
/// Calling this method must not change the booster's state.
///
/// # Examples
/// Inspect AdaBoost's model after each round. The labels below are not
/// separable by one threshold, so AdaBoost runs every forced round.
/// ```
/// use std::io::BufReader;
/// use miniboosts::prelude::*;
/// use miniboosts::Sample;
/// use miniboosts::core::CurrentHypothesis;
///
/// let csv = b"x,class\n-2,-1\n-1,1\n1,1\n2,-1\n";
/// let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
///     .unwrap()
///     .set_target("class");
/// let tree = DecisionTreeBuilder::new(&sample).max_depth(1).build();
/// let mut booster = AdaBoost::init(&sample).force_quit_at(3);
///
/// booster.preprocess();
/// let mut iteration = 1;
/// while booster.boost(&tree, iteration).is_continue() {
///     let snapshot = booster.current_hypothesis();
///     assert_eq!(snapshot.hypotheses.len(), iteration);
///     // ‖w‖₁ = 1 and leaf confidences in [-1, 1] keep the
///     // combined confidence in [-1, 1] (up to rounding).
///     let conf = snapshot.confidence_all(&sample);
///     assert!(conf.iter().all(|c| c.abs() <= 1.0 + 1e-12));
///     iteration += 1;
/// }
/// assert_eq!(iteration, 4);
/// // After the last round, the snapshot predicts like the final model.
/// let snapshot = booster.current_hypothesis();
/// let f = booster.postprocess();
/// assert_eq!(snapshot.confidence_all(&sample), f.confidence_all(&sample));
/// ```
pub trait CurrentHypothesis {
    /// The type of the snapshot, typically the same combined hypothesis type
    /// as [`Booster::Output`].
    type Output;

    /// Returns the combined hypothesis built from the rounds completed so far.
    /// Before the first round, the crate's boosters return an empty
    /// ensemble, whose confidence is `0.0` on every example.
    fn current_hypothesis(&self) -> Self::Output;
}
