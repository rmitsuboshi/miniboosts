use crate::{Sample, RegressionTree};
use super::bin::*;
use super::loss::*;

use std::collections::HashMap;


/// The number of bins set as default.
pub const DEFAULT_NBIN: usize = 255;
/// The maxmial depth set as default.
pub const DEFAULT_MAX_DEPTH: usize = 2;
/// Default L2-regularization parameter
pub const DEFAULT_LAMBDA_L2: f64 = 0.01;


/// A struct that builds `RegressionTree`.
/// `RegressionTreeBuilder` keeps parameters for constructing `RegressionTree`.
/// 
/// # Example
/// 
/// ```no_run
/// use miniboosts::prelude::*;
/// 
/// let weak_learner = RegressionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .loss(LossType::L1)
///     .lambda_l2(0.1)
///     .build();
/// ```
#[derive(Clone)]
pub struct RegressionTreeBuilder<'a> {
    sample: &'a Sample,
    /// Number of bins per feature.
    n_bins: HashMap<&'a str, usize>,


    max_depth: usize,


    /// L2 regularization for the leaf values.
    lambda_l2: f64,

    /// Loss function
    loss: LossType,
}


impl<'a> RegressionTreeBuilder<'a> {
    /// Construct a new instance of `RegressionTreeBuilder`.
    /// By default, 
    /// `RegressionTreeBuilder` sets the parameters as follows;
    /// ```text
    /// n_bins: DEFAULT_NBIN == 255,
    /// max_depth: DEFAULT_MAX_DEPTH == 2,
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
        let max_depth = DEFAULT_MAX_DEPTH;

        let lambda_l2 = DEFAULT_LAMBDA_L2;

        let loss = LossType::L2;

        Self { sample, n_bins, max_depth, loss, lambda_l2, }
    }


    /// Specify the loss type. Default is `LossType::L2`.
    pub fn loss(mut self, loss: LossType) -> Self {
        self.loss = loss;
        self
    }


    /// Set the L2-regularization parameter.
    pub fn lambda_l2(mut self, lambda_l2: f64) -> Self {
        self.lambda_l2 = lambda_l2;
        self
    }


    /// Specify the maximal depth of the tree.
    /// Default maximal depth is `2`.
    pub fn max_depth(mut self, depth: usize) -> Self {
        assert!(depth > 0);
        self.max_depth = depth;

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


    /// Build a `RegressionTree`.
    /// This method consumes `self`.
    pub fn build(self) -> RegressionTree<'a> {
        let bins = self.sample.features()
            .iter()
            .map(|feature| {
                let name = feature.name();
                let n_bins = *self.n_bins.get(name).unwrap();

                (name, Bins::cut(feature, n_bins))
            })
            .collect::<HashMap<_, _>>();


        let n_sample = self.sample.shape().0;
        let regression_tree = RegressionTree::from_components(
            bins, n_sample, self.max_depth, self.lambda_l2, self.loss,
        );


        regression_tree
    }
}
