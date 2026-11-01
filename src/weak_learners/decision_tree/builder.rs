//! [`DecisionTreeBuilder`]: configures and builds a [`DecisionTree`].
use crate::core::{Sample, binning::*, tree::*};
use crate::weak_learners::decision_tree::DecisionTree;
use crate::weak_learners::decision_tree::split_by::*;
use std::collections::HashMap;

/// The default maximal number of bins per feature. Each feature gets
/// `min(number of distinct values, DEFAULT_NBIN)` bins.
pub const DEFAULT_NBIN: usize = 255;
/// The default maximal depth of the tree.
pub const DEFAULT_MAX_DEPTH: usize = 2;

/// Configures and builds a [`DecisionTree`] weak learner.
///
/// The builder borrows the training sample, from which [`Self::build`]
/// computes the equal-width bins of every feature; the resulting
/// [`DecisionTree`] borrows the sample's feature names.
///
/// # Parameters
/// | Method | Default | Meaning |
/// |---|---|---|
/// | [`Self::max_depth`] | `2` | maximal number of split levels (`>= 1`) |
/// | [`Self::split_by`] | [`SplitBy::Entropy`] | splitting criterion |
/// | [`Self::set_nbins`] | `min(distinct values, 255)` | bins of one feature |
///
/// # Examples
/// Configure every parameter, build the weak learner, and train a booster.
/// ```
/// use std::io::BufReader;
/// use miniboosts::prelude::*;
/// use miniboosts::Sample;
///
/// let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
/// let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
///     .unwrap()
///     .set_target("class");
///
/// let mut builder = DecisionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .split_by(SplitBy::Gini);
/// // `set_nbins` takes `&mut self`, so it is called separately.
/// builder.set_nbins("x", 8);
/// let weak_learner = builder.build();
///
/// let f = AdaBoost::init(&sample).run(&weak_learner);
/// assert_eq!(f.predict_all(&sample), vec![-1, -1, -1, 1, 1, 1]);
/// ```
#[derive(Clone)]
pub struct DecisionTreeBuilder<'a> {
    sample: &'a Sample,
    n_bins: HashMap<&'a str, usize>,
    max_depth: Depth,
    split_by: SplitBy,
}

impl<'a> DecisionTreeBuilder<'a> {
    /// Creates a builder for `sample` with the default parameters.
    ///
    /// Time complexity: `O(m log m)` per feature (counting distinct values),
    /// where `m` is the number of examples.
    ///
    /// # Panics
    /// Panics if a feature contains NaN.
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::Sample;
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// use miniboosts::{DecisionTreeBuilder, WeakLearner};
    ///
    /// // Defaults: depth 2, entropy, min(distinct values, 255) bins.
    /// let tree = DecisionTreeBuilder::new(&sample).build();
    /// let info = tree.info().unwrap();
    /// assert!(info.contains(&("Max depth", "2".to_string())));
    /// assert!(info.contains(&("Split criterion", "Entropy".to_string())));
    /// // `x` takes 4 distinct values and `y` takes 3.
    /// assert!(info.contains(&("# of bins (max)", "4".to_string())));
    /// ```
    pub fn new(sample: &'a Sample) -> Self {
        let n_bins = sample
            .features()
            .iter()
            .map(|feat| {
                let n_bin = feat.distinct_value_count().min(DEFAULT_NBIN);
                (feat.name(), n_bin)
            })
            .collect();
        let max_depth = Depth::from(DEFAULT_MAX_DEPTH);
        let split_by = SplitBy::Entropy;

        Self {
            sample,
            n_bins,
            max_depth,
            split_by,
        }
    }

    /// Sets the maximal depth of the tree, i.e., the maximal number of splits
    /// on any root-to-leaf path. `1` gives a decision stump; a tree of depth
    /// `depth` has at most `2^depth` leaves. Default is `2`.
    ///
    /// # Panics
    /// Panics if `depth == 0`.
    ///
    /// # Examples
    /// Labels `+1` exactly on the interval `[0, 2)` need two thresholds:
    /// a stump cannot fit them, a depth-2 tree can.
    /// ```
    /// use miniboosts::{Classifier, DecisionTreeBuilder, Sample, WeakLearner};
    ///
    /// let rows = vec![vec![-2.0], vec![-1.0], vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    /// let y = [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0];
    /// let sample = Sample::from_dense(&rows, Some(&y)).unwrap();
    /// let uniform = vec![1.0 / 6.0; 6];
    ///
    /// let stump = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    /// let h = stump.produce(&sample, &uniform);
    /// assert_ne!(h.predict_all(&sample), vec![-1, -1, 1, 1, -1, -1]);
    ///
    /// let tree = DecisionTreeBuilder::new(&sample).max_depth(2).build();
    /// let h = tree.produce(&sample, &uniform);
    /// assert_eq!(h.predict_all(&sample), vec![-1, -1, 1, 1, -1, -1]);
    /// ```
    pub fn max_depth(mut self, depth: usize) -> Self {
        assert!(depth > 0, "Tree must have positive depth");
        self.max_depth = Depth::from(depth);

        self
    }

    /// Sets the node splitting criterion.
    /// Default is [`SplitBy::Entropy`].
    /// See [`SplitBy`] for the other criteria and their formulas.
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::Sample;
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// use miniboosts::{DecisionTreeBuilder, SplitBy, WeakLearner};
    ///
    /// let tree = DecisionTreeBuilder::new(&sample)
    ///     .split_by(SplitBy::Twoing)
    ///     .build();
    /// let info = tree.info().unwrap();
    /// assert!(info.contains(&("Split criterion", "Twoing Rule".to_string())));
    /// ```
    #[inline]
    pub fn split_by(mut self, split_by: SplitBy) -> Self {
        self.split_by = split_by;
        self
    }

    /// Sets the number of equal-width bins for the feature named `name`.
    /// More bins give more candidate thresholds at a higher cost.
    /// By default, a feature with `k` distinct values (an implicit sparse
    /// zero counts once) gets `min(k, 255)` bins.
    ///
    /// Unlike the other setters, this method takes `&mut self` and cannot be
    /// chained.
    ///
    /// # Panics
    /// Panics if no feature is named `name`. A value of `0` is accepted here
    /// but makes [`Self::build`] panic.
    ///
    /// # Examples
    /// With a single bin, the only candidate threshold is the upper end of
    /// that bin, which sends every example to one side, so no split is made
    /// and the tree is one leaf.
    /// ```
    /// use std::io::BufReader;
    /// use miniboosts::{Classifier, DecisionTreeBuilder, Sample, WeakLearner};
    ///
    /// let csv = b"x,class\n-2,-1\n-1,-1\n1,1\n2,1\n";
    /// let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    ///     .unwrap()
    ///     .set_target("class");
    /// let uniform = vec![0.25; 4];
    ///
    /// let mut builder = DecisionTreeBuilder::new(&sample).max_depth(1);
    /// builder.set_nbins("x", 1);
    /// let h = builder.build().produce(&sample, &uniform);
    /// // A single leaf with balanced weights has confidence 0
    /// // (exact: 0.5 - 0.5 with dyadic weights).
    /// assert_eq!(h.confidence_all(&sample), vec![0.0; 4]);
    ///
    /// // With the default bins (4 here) the stump separates the classes.
    /// let h = DecisionTreeBuilder::new(&sample)
    ///     .max_depth(1)
    ///     .build()
    ///     .produce(&sample, &uniform);
    /// assert_eq!(h.confidence_all(&sample), vec![-1.0, -1.0, 1.0, 1.0]);
    /// ```
    pub fn set_nbins<T>(&mut self, name: T, n_bins: usize)
    where
        T: AsRef<str>,
    {
        let name = name.as_ref();
        match self.n_bins.get_mut(name) {
            Some(val) => {
                *val = n_bins;
            }
            None => {
                panic!("The feature named `{name}` does not exist");
            }
        }
    }

    /// Builds the [`DecisionTree`], cutting every feature into its bins
    /// with [`Bins::cut`]. This method consumes `self`.
    ///
    /// # Panics
    /// Panics if some feature gets `0` bins (e.g., an empty feature), is
    /// empty, or contains non-finite values; see [`Bins::cut`].
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::Sample;
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// use miniboosts::{DecisionTreeBuilder, WeakLearner};
    ///
    /// let tree = DecisionTreeBuilder::new(&sample).max_depth(3).build();
    /// assert_eq!(tree.name(), "Decision Tree");
    /// // `DecisionTree` implements `Display`, listing its parameters and bins.
    /// assert!(tree.to_string().contains("Max depth: 3"));
    /// ```
    pub fn build(self) -> DecisionTree<'a> {
        let bins = self
            .sample
            .features()
            .iter()
            .map(|feature| {
                let name = feature.name();
                let n_bins = *self.n_bins.get(name).unwrap();

                (name, Bins::cut(feature, n_bins))
            })
            .collect::<HashMap<_, _>>();

        let dtree = DecisionTree::new(bins, self.split_by, self.max_depth);

        dtree
    }
}
