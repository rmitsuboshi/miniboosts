use polars::prelude::*;
use std::ops::Index;

const BUF_SIZE: usize = 256;

/// Dense representation of a feature.
#[derive(Debug)]
pub struct DenseFeature {
    /// Feature name
    pub name: String,
    /// Feature values.
    pub sample: Vec<f64>,
}


/// Sparse representation of a feature.
#[derive(Debug)]
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
#[derive(Debug)]
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


    /// Get the feature name.
    pub fn name(&self) -> &str {
        match self {
            Self::Dense(feat) => feat.name(),
            Self::Sparse(feat) => feat.name(),
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
}


impl SparseFeature {
    /// Construct an empty dense feature with `name`.
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


    fn name(&self) -> &str {
        &self.name
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
