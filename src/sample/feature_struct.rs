use polars::prelude::*;
use std::ops::Index;
use std::slice::Iter;

use crate::common::{utils, checker};

const BUF_SIZE: usize = 256;
const MINIMAL_WEIGHT_SUM: f64 = 1e-100;

/// Dense representation of a feature.
#[derive(Debug,Clone)]
pub struct DenseFeature {
    /// Feature name
    pub name: String,
    /// Feature values.
    pub sample: Vec<f64>,
}


/// Sparse representation of a feature.
#[derive(Debug,Clone)]
pub struct SparseFeature {
    /// Feature name
    pub name: String,
    /// Pairs of sample index and feature value
    pub sample: Vec<(usize, f64)>,
    /// Number of examples.
    /// Note that `self.n_sample >= self.sample.len()`.
    pub(crate) n_sample: usize,
}


/// An enumeration of sparse/dense feature.
#[derive(Debug,Clone)]
pub enum Feature {
    /// Dense representation of a feature
    Dense(DenseFeature),
    /// Sparse representation of a feature
    Sparse(SparseFeature),
}


impl Feature {
    /// Construct a dense feature
    pub fn new_dense<T: ToString>(name: T) -> Self {
        Self::Dense(DenseFeature::new(name))
    }


    /// Construct a sparse feature
    pub fn new_sparse<T: ToString>(name: T) -> Self {
        Self::Sparse(SparseFeature::new(name))
    }


    pub(crate) fn is_sparse(&self) -> bool {
        match self {
            Self::Dense(_) => false,
            Self::Sparse(_) => true,
        }
    }


    // pub(crate) fn is_dense(&self) -> bool {
    //     !self.is_sparse()
    // }


    pub(crate) fn set_n_sample(&mut self, n_sample: usize) {
        match self {
            Self::Dense(_) => {},
            Self::Sparse(feat) => {
                feat.n_sample = n_sample;
            }
        }
    }


    pub(crate) fn append(&mut self, i: usize, f: f64) {
        match self {
            Self::Dense(feat) => { feat.append(f); },
            Self::Sparse(feat) => { feat.append((i, f)); }
        }
    }


    /// Get the feature name.
    pub fn name(&self) -> &str {
        match self {
            Self::Dense(feat) => feat.name(),
            Self::Sparse(feat) => feat.name(),
        }
    }


    pub(super) fn replace_name<S>(&mut self, name: S) -> String
        where S: ToString,
    {
        match self {
            Self::Dense(feat) => feat.replace_name(name),
            Self::Sparse(feat) => feat.replace_name(name),
        }
    }


    pub(crate) fn into_target(self) -> Vec<f64> {
        match self {
            Self::Dense(feat) => feat.into_target(),
            Self::Sparse(feat) => feat.into_target(),
        }
    }


    /// Returns the number of items in this feature.
    pub fn len(&self) -> usize {
        match self {
            Self::Dense(feat) => feat.len(),
            Self::Sparse(feat) => feat.len(),
        }
    }


    /// Returns `true` if the number of examples is equals to `0`.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Dense(feat) => feat.is_empty(),
            Self::Sparse(feat) => feat.is_empty(),
        }
    }


    pub(crate) fn distinct_value_count(&self) -> usize {
        match self {
            Self::Dense(feat) => feat.distinct_value_count(),
            Self::Sparse(feat) => feat.distinct_value_count(),
        }
    }


    /// Compute the weighted mean of the feature.
    pub(crate) fn weighted_mean<T>(&self, weight: T) -> f64
        where T: AsRef<[f64]>
    {
        let weight = weight.as_ref();
        checker::check_capped_simplex_condition(weight, 1.0);


        match self {
            Self::Dense(feat) => feat.weighted_mean(weight),
            Self::Sparse(feat) => feat.weighted_mean(weight),
        }
    }


    pub(crate) fn weighted_mean_and_variance<T>(&self, weight: T)
        -> (f64, f64)
        where T: AsRef<[f64]>
    {
        let weight = weight.as_ref();
        let mean = self.weighted_mean(weight);


        let variance = match self {
            Self::Dense(feat) => feat.weighted_variance(mean, weight),
            Self::Sparse(feat) => feat.weighted_variance(mean, weight),
        };
        (mean, variance)
    }


    /// Computes the mean for this feature whose label is `y.`
    pub(crate) fn weighted_mean_for_label<T>(
        &self,
        y: f64,
        target: T,
        weight: T
    ) -> f64
        where T: AsRef<[f64]>
    {
        let target = target.as_ref();
        let weight = weight.as_ref();
        match self {
            Self::Dense(feat)
                => feat.weighted_mean_for_label(y, target, weight),
            Self::Sparse(feat)
                => feat.weighted_mean_for_label(y, target, weight),
        }
    }


    /// Computes the mean and variance for this feature
    /// whose label is `y.`
    pub(crate) fn weighted_mean_and_variance_for_label<T>(
        &self,
        y: f64,
        target: T,
        weight: T
    ) -> (f64, f64)
        where T: AsRef<[f64]>
    {
        let target = target.as_ref();
        let weight = weight.as_ref();
        let mean = self.weighted_mean_for_label(y, target, weight);
        let var = match self {
            Self::Dense(feat)
                => feat.weighted_variance_for_label(mean, y, target, weight),
            Self::Sparse(feat)
                => feat.weighted_variance_for_label(mean, y, target, weight),
        };
        (mean, var)
    }
}


impl DenseFeature {
    /// Construct an empty dense feature with `name`.
    pub fn new<T: ToString>(name: T) -> Self {
        Self {
            name: name.to_string(),
            sample: Vec::with_capacity(BUF_SIZE),
        }
    }


    fn name(&self) -> &str {
        &self.name
    }


    pub(self) fn replace_name<S>(&mut self, name: S) -> String
        where S: ToString,
    {
        let name = name.to_string();
        std::mem::replace(&mut self.name, name)
    }


    /// Returns an iterator over feature values.
    pub fn iter(&self) -> Iter<'_, f64> {
        self.sample.iter()
    }


    /// Convert `polars::Series` into `DenseFeature`.
    pub fn from_series(series: &Series) -> Self {
        let name = series.name().to_string();

        let sample = series.f64()
            .expect("The series is not a dtype f64")
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .unwrap();

        Self { name, sample, }
    }


    fn into_target(self) -> Vec<f64> {
        self.sample
    }


    /// Append an example to this feature.
    pub fn append(&mut self, x: f64) {
        self.sample.push(x);
    }


    /// Returns the number of items in `self.sample`.
    pub fn len(&self) -> usize {
        self.sample.len()
    }


    /// Returns `true` if `self.len()` is equals to `0`.
    pub fn is_empty(&self) -> bool {
        self.sample.is_empty()
    }


    fn distinct_value_count(&self) -> usize {
        let values = self.sample[..].to_vec();
        inner_distinct_value_count(values)
    }


    pub(self) fn weighted_mean(&self, weight: &[f64]) -> f64 {
        self.sample.iter()
            .zip(weight)
            .map(|(f, w)| f * w)
            .sum::<f64>()
    }


    pub(self) fn weighted_variance(&self, mean: f64, weight: &[f64])
        -> f64
    {
        self.sample.iter()
            .zip(weight)
            .map(|(f, w)| w * (f - mean).powi(2))
            .sum::<f64>()
    }


    /// Compute the mean of the feature whose label is `y`.
    /// Each instance `i` is weighted by weighted by `weight[i]`
    pub(self) fn weighted_mean_for_label(
        &self,
        y: f64,
        target: &[f64],
        weight: &[f64],
    ) -> f64
    {
        let mut total_weight = utils::total_weight_for_label(y, target, weight);
        if total_weight == 0.0 { total_weight = MINIMAL_WEIGHT_SUM; }
        self.sample.iter()
            .zip(target)
            .zip(weight)
            .map(|((f, t), w)| if *t != y { 0.0 } else { f * w })
            .sum::<f64>()
            / total_weight
    }


    /// Compute the variance of the feature whose label is `y`.
    /// Each instance `i` is weighted by weighted by `weight[i]`
    pub(self) fn weighted_variance_for_label(
        &self,
        mean: f64,
        y: f64,
        target: &[f64],
        weight: &[f64],
    ) -> f64
    {
        let mut total_weight = utils::total_weight_for_label(y, target, weight);
        if total_weight == 0.0 { total_weight = MINIMAL_WEIGHT_SUM; }
        self.sample.iter()
            .zip(target)
            .zip(weight)
            .map(|((f, t), w)| {
                if *t != y { 0.0 } else { w * (f - mean).powi(2) }
            })
            .sum::<f64>()
            / total_weight
    }
}


impl SparseFeature {
    /// Construct an empty sparse feature with `name`.
    pub fn new<T: ToString>(name: T) -> Self {
        Self {
            name: name.to_string(),
            sample: Vec::with_capacity(BUF_SIZE),
            n_sample: 0_usize,
        }
    }


    /// Append an example to this feature.
    pub fn append(&mut self, (i, x): (usize, f64)) {
        self.sample.push((i, x));
    }


    pub(self) fn replace_name<S>(&mut self, name: S) -> String
        where S: ToString,
    {
        let name = name.to_string();
        std::mem::replace(&mut self.name, name)
    }


    fn name(&self) -> &str {
        &self.name
    }


    /// Returns an iterator over non-zero feature values.
    pub fn iter(&self) -> Iter<'_, (usize, f64)> {
        self.sample.iter()
    }


    fn into_target(self) -> Vec<f64> {
        let sample = self.sample;

        let mut target = vec![0.0_f64; self.n_sample];
        sample.into_iter()
            .for_each(|(i, x)| {
                target[i] = x;
            });

        target
    }


    /// Return the number of examples that have non-zero value.
    pub fn len(&self) -> usize {
        self.sample.len()
    }


    /// Returns `true` if this feature has zero values.
    pub fn has_zero(&self) -> bool {
        self.len() < self.n_sample
    }


    /// Returns the number of indices that have zero-value.
    pub fn zero_counts(&self) -> usize {
        self.n_sample - self.len()
    }


    /// Returns `true` if `self.len()` is equals to `0`.
    pub fn is_empty(&self) -> bool {
        self.sample.is_empty()
    }


    fn distinct_value_count(&self) -> usize {
    
        let values = self.sample[..]
            .iter()
            .map(|(_, v)| *v)
            .collect::<Vec<_>>();
        let mut uniq_value_count = inner_distinct_value_count(values);
        if self.has_zero() {
            uniq_value_count += 1;
        }
        uniq_value_count
    }


    pub(self) fn weighted_mean(&self, weight: &[f64]) -> f64 {
        self.sample.iter()
            .map(|&(i, f)| weight[i] * f)
            .sum::<f64>()
    }


    pub(self) fn weighted_mean_for_label(
        &self,
        y: f64,
        target: &[f64],
        weight: &[f64],
    ) -> f64
    {
        let mut total_weight = utils::total_weight_for_label(y, target, weight);
        if total_weight == 0.0 { total_weight = MINIMAL_WEIGHT_SUM; }
        self.sample.iter()
            .zip(target)
            .map(|((i, f), &t)|
                if t == y { 0.0 } else { weight[*i] * f }
            )
            .sum::<f64>()
            / total_weight
    }


    pub(self) fn weighted_variance(&self, mean: f64, weight: &[f64])
        -> f64
    {
        let mut variance = 0.0;
        let mut zero_weight = 0.0;
        let mut prev = 0;
        for &(i, f) in self.sample.iter() {
            zero_weight += weight[prev..i].iter().sum::<f64>();
            // while prev < i {
            //     zero_weight += weight[prev];
            //     prev += 1;
            // }
            prev = i + 1;
            variance += weight[i] * (f - mean).powi(2);
        }

        zero_weight += weight[prev..].iter().sum::<f64>();

        variance += zero_weight * mean.powi(2);
        variance
    }


    pub(self) fn weighted_variance_for_label(
        &self,
        mean: f64,
        y: f64,
        target: &[f64],
        weight: &[f64],
    ) -> f64
    {
        let mut total_weight = utils::total_weight_for_label(y, target, weight);
        if total_weight == 0.0 { total_weight = MINIMAL_WEIGHT_SUM; }

        let mut variance = 0.0;
        let mut zero_weight = 0.0;
        let mut prev = 0;
        for &(i, f) in self.sample.iter() {
            zero_weight += weight[prev..i].iter().sum::<f64>();
            prev = i + 1;
            if target[i] == y {
                variance += weight[i] * (f - mean).powi(2);
            }
        }

        zero_weight += weight[prev..].iter().sum::<f64>();

        variance += zero_weight * mean.powi(2);
        variance / total_weight
    }
}


impl Index<usize> for Feature {
    type Output = f64;
    fn index(&self, idx: usize) -> &Self::Output {
        match self {
            Self::Dense(feat)  => &feat[idx],
            Self::Sparse(feat) => &feat[idx],
        }
    }
}


/// Count the number of items in `src` that has the same value.
/// The given vector `src` is assumed to be sorted in ascending order.
fn inner_distinct_value_count(mut src: Vec<f64>) -> usize {
    src.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut iter = src.into_iter();
    let mut value = match iter.next() {
        Some(v) => v,
        None => { return 0; }
    };
    let mut uniq_value_count = 1;

    for v in iter {
        if v != value {
            value = v;
            uniq_value_count += 1;
        }
    }

    uniq_value_count
}


impl Index<usize> for DenseFeature {
    type Output = f64;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.sample[idx]
    }
}


impl Index<usize> for SparseFeature {
    type Output = f64;
    fn index(&self, idx: usize) -> &Self::Output {
        let out_of_range = self.sample.len();
        let pos = self.sample[..].binary_search_by(|(i, _)| i.cmp(&idx))
            .unwrap_or(out_of_range);

        self.sample.get(pos).map(|(_, x)| x).unwrap_or(&0.0)
    }
}
