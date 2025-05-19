pub mod constants;
pub mod tools;
pub mod sample;
pub mod booster;
pub mod weak_learner;
pub mod hypothesis;

pub use tools::{
    binning,
    checkers,
    helpers,
    tree,
};

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
pub use sample::{
    SampleReader,
    Sample,
    Feature,
};

pub use weak_learner::{
    WeakLearner,
};

pub use booster::{
    Booster,
};

pub use hypothesis::{
    Classifier,
    Regressor,
};

