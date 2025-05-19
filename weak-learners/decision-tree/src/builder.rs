use miniboosts_core::{
    tree::*,
    binning::*,
    Sample,
};
use crate::DecisionTree;
use crate::split_by::*;
use std::collections::HashMap;

/// The number of bins set as default.
pub const DEFAULT_NBIN: usize = 255;
/// The maxmial depth set as default.
pub const DEFAULT_MAX_DEPTH: usize = 2;

/// A struct that builds `DecisionTree`.
/// `DecisionTreeBuilder` keeps parameters for constructing `DecisionTree`.
/// 
/// # Example
/// 
/// ```no_run
/// use decision_tree::{
///     DecisionTreeBuilder,
///     SplitBy,
/// };
/// use miniboosts_core::{
///     Sample,
///     SampleReader,
/// };
/// let file = "/path/to/data/file.csv";
/// let sample = SampleReader::default()
///     .file(file)
///     .has_header(true)
///     .target_feature("class")
///     .read()
///     .unwrap();
/// let weak_learner = DecisionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .split_by(SplitBy::Entropy)
///     .build();
/// ```
#[derive(Clone)]
pub struct DecisionTreeBuilder<'a> {
    sample: &'a Sample,
    n_bins: HashMap<&'a str, usize>,
    max_depth: Depth,
    split_by: SplitBy,
}

impl<'a> DecisionTreeBuilder<'a> {
    /// Construct a new instance of [`DecisionTreeBuilder`].
    pub fn new(sample: &'a Sample) -> Self {
        let n_bins = sample.features()
            .iter()
            .map(|feat| {
                let n_bin = feat.distinct_value_count()
                    .min(DEFAULT_NBIN);
                (feat.name(), n_bin)
            })
            .collect();
        let max_depth = Depth::from(DEFAULT_MAX_DEPTH);
        let split_by = SplitBy::Entropy;

        Self { sample, n_bins, max_depth, split_by, }
    }

    /// Specify the maximal depth of the tree.
    /// Default maximal depth is `2`.
    pub fn max_depth(mut self, depth: usize) -> Self {
        assert!(depth > 0, "Tree must have positive depth");
        self.max_depth = Depth::from(depth);

        self
    }

    /// Set the node splitting rule.
    /// Default value is `SplitBy::Entropy`.
    /// See [`SplitBy`] for other rules.
    #[inline]
    pub fn split_by(mut self, split_by: SplitBy) -> Self {
        self.split_by = split_by;
        self
    }

    /// Set the number of bins to a feature named `name`.
    /// By default, each feature is binned in `255` bins.
    pub fn set_nbins<T>(&mut self, name: T, n_bins: usize)
        where T: AsRef<str>
    {
        let name = name.as_ref();
        match self.n_bins.get_mut(name) {
            Some(val) => { *val = n_bins; },
            None => {
                panic!("The feature named `{name}` does not exist");
            },
        }
    }

    /// Build a `DecisionTree`.
    /// This method consumes `self`.
    pub fn build(self) -> DecisionTree<'a> {
        let bins = self.sample.features()
            .iter()
            .map(|feature| {
                let name = feature.name();
                let n_bins = *self.n_bins.get(name).unwrap();

                (name, Bins::cut(feature, n_bins))
            })
            .collect::<HashMap<_, _>>();

        let dtree = DecisionTree::new(
            bins, self.split_by, self.max_depth
        );

        dtree
    }
}

