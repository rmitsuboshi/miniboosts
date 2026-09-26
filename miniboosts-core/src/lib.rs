pub mod booster;
pub mod constants;
pub mod hypothesis;
pub mod sample;
pub mod tools;
pub mod weak_learner;

pub use tools::{binning, checkers, helpers, tree};

/// A struct that returns [`Sample`].
/// Using this struct, one can read a CSV/SVMLIGHT format file to [`Sample`].
/// Other formats are not supported yet.
/// # Example
/// The following code is a simple example to read a CSV file.
/// ```no_run
/// use miniboosts_core::SampleReader;
/// let filename = "/path/to/csv/file.csv";
/// let sample = SampleReader::default()
///     .file(filename)
///     .has_header(true)
///     .target_feature("class")
///     .read()
///     .unwrap();
/// ```
pub use sample::{Feature, Sample, SampleReader};

pub use weak_learner::WeakLearner;

pub use booster::Booster;

pub use hypothesis::{Classifier, Regressor};

/// A snapshot of a booster's current model, used by experiment loggers.
/// Implementations must preserve the final model's prediction semantics.
pub trait CurrentHypothesis {
    type Output;
    fn current_hypothesis(&self) -> Self::Output;
}
