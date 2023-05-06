use crate::{Sample, DTree};
use crate::weak_learner::common::type_and_struct::*;
use super::{
    bin::*,
    criterion::*,
};
use std::collections::HashMap;


/// The number of bins set as default.
pub const DEFAULT_NBIN: usize = 255;
pub const DEFAULT_MAX_DEPTH: usize = 2;


/// A struct that builds `DTree`.
/// `DTreeBuilder` constructs a weak-learner 
/// that produces decision-tree classifiers.
#[derive(Clone)]
pub struct DTreeBuilder<'a> {
    sample: &'a Sample,
    /// Number of bins per feature.
    n_bins: HashMap<&'a str, usize>,

    max_depth: Depth,
    criterion: Criterion,
}


impl<'a> DTreeBuilder<'a> {
    /// Construct a new instance of `DTreeBuilder`.
    /// By default, `DTreeBuilder` sets the parameters as follows;
    /// ```text
    /// n_bins: DEFAULT_NBIN == 255,
    /// max_depth: DEFAULT_MAX_DEPTH == 2,
    /// criterion: Criterion::Entropy,
    /// ```
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
        let criterion = Criterion::Entropy;

        Self { sample, n_bins, max_depth, criterion, }
    }


    /// Specify the maximal depth of the tree.
    /// Default maximal depth is `2`.
    pub fn max_depth(mut self, depth: usize) -> Self {
        assert!(depth > 0);
        self.max_depth = Depth::from(depth);

        self
    }


    /// Set criterion for node splitting.
    /// Default value is `Criterion::Entropy`.
    /// See [`Criterion`](Criterion).
    #[inline]
    pub fn criterion(mut self, criterion: Criterion) -> Self {
        self.criterion = criterion;
        self
    }


    /// Set the number of bins to a feature named `name`.
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


    /// Build a `DTree`.
    /// This method consumes `self`.
    pub fn build(self) -> DTree<'a> {
        let bins = self.sample.features()
            .iter()
            .map(|feature| {
                let name = feature.name();
                let n_bins = *self.n_bins.get(name).unwrap();

                (name, Bins::cut(feature, n_bins))
            })
            .collect::<HashMap<_, _>>();

        let dtree = DTree::from_components(
            bins, self.criterion, self.max_depth
        );


        dtree
    }
}
