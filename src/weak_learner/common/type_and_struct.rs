use serde::{Serialize, Deserialize};
use std::{ops, cmp};
use std::collections::HashMap;

use crate::sample::feature::{SparseFeature, DenseFeature};
use crate::Feature;


#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
#[repr(transparent)]
pub(crate) struct Prediction<T>(pub(crate) T);


impl<T> From<T> for Prediction<T> {
    #[inline]
    fn from(prediction: T) -> Self {
        Self(prediction)
    }
}


impl<T> ops::Add<Self> for Prediction<T>
    where T: ops::Add<Output = T>,
{
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}


#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
#[repr(transparent)]
pub(crate) struct Confidence<T>(pub(crate) T);


impl<T> From<T> for Confidence<T> {
    #[inline]
    fn from(prediction: T) -> Self {
        Self(prediction)
    }
}


impl<T> ops::Add<Self> for Confidence<T>
    where T: ops::Add<Output = T>,
{
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}


#[derive(Clone, Copy, PartialEq)]
#[repr(transparent)]
pub(crate) struct LossValue(pub(crate) f64);


impl From<f64> for LossValue {
    #[inline]
    fn from(loss_value: f64) -> Self {
        Self(loss_value)
    }
}


impl ops::Add<Self> for LossValue {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl cmp::PartialEq<f64> for LossValue {
    #[inline]
    fn eq(&self, other: &f64) -> bool {
        self.0.eq(other)
    }
}


impl cmp::PartialOrd<Self> for LossValue {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}


/// Struct `Depth` defines the maximal depth of a tree.
/// This is just a wrapper for `usize`.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub(crate) struct Depth(usize);


impl From<usize> for Depth {
    fn from(depth: usize) -> Self {
        Self(depth)
    }
}


impl ops::Sub<usize> for Depth {
    type Output = Self;
    /// Define the subtraction of the `Depth` struct.
    /// The subtraction does not return a value less than or equals to 1.
    #[inline]
    fn sub(self, other: usize) -> Self::Output {
        if self.0 <= 1 {
            self
        } else {
            Self(self.0 - other)
        }
    }
}

impl cmp::PartialEq<usize> for Depth {
    #[inline]
    fn eq(&self, rhs: &usize) -> bool {
        self.0.eq(rhs)
    }
}


impl cmp::PartialOrd<usize> for Depth {
    #[inline]
    fn partial_cmp(&self, other: &usize) -> Option<cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}



#[derive(Clone, Copy, PartialEq)]
#[repr(transparent)]
pub(crate) struct FeatureValue(pub(crate) f64);


impl From<f64> for FeatureValue {
    fn from(feature: f64) -> Self {
        Self(feature)
    }
}


impl ops::Add<Self> for FeatureValue {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl cmp::PartialEq<f64> for FeatureValue {
    fn eq(&self, other: &f64) -> bool {
        self.0.eq(other)
    }
}


impl cmp::PartialOrd<Self> for FeatureValue {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}



/// Probability mass
#[derive(Clone, Copy, PartialEq)]
#[repr(transparent)]
pub(crate) struct Mass(pub(crate) f64);


impl From<f64> for Mass {
    fn from(mass: f64) -> Self {
        Self(mass)
    }
}


impl ops::Add<Self> for Mass {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl cmp::PartialEq<f64> for Mass {
    fn eq(&self, other: &f64) -> bool {
        self.0.eq(other)
    }
}


/// Target
#[derive(Clone, Copy, PartialEq)]
#[repr(transparent)]
pub(crate) struct Target(pub(crate) f64);


impl From<f64> for Target {
    #[inline]
    fn from(target: f64) -> Self {
        Self(target)
    }
}


impl ops::Add<Self> for Target {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl cmp::PartialEq<f64> for Target {
    #[inline]
    fn eq(&self, other: &f64) -> bool {
        self.0.eq(other)
    }
}


impl cmp::PartialOrd<Self> for Target {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}




#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
#[repr(transparent)]
pub(crate) struct Threshold(pub(crate) f64);


impl From<f64> for Threshold {
    #[inline]
    fn from(threshold: f64) -> Self {
        Self(threshold)
    }
}


impl ops::Add<Self> for Threshold {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl cmp::PartialEq<f64> for Threshold {
    #[inline]
    fn eq(&self, other: &f64) -> bool {
        self.0.eq(other)
    }
}

#[derive(Debug)]
pub(crate) struct WeightedFeature {
    pub(crate) feature_val: f64,
    pub(crate) label_to_weight: HashMap<i64, f64>,
    total_weight: f64,
}


impl WeightedFeature {
    pub(crate) fn new(
        feature_val: f64,
        label_to_weight: HashMap<i64, f64>,
    ) -> Self
    {
        let total_weight = label_to_weight.values().sum::<f64>();
        Self { feature_val, label_to_weight, total_weight, }
    }


    pub(crate) fn total_weight(&self) -> f64 {
        self.total_weight
    }
}


pub(crate) fn group_by_x(
    feature: &Feature,
    target: &[f64],
    indices: &[usize],
    dist: &[f64],
) -> Vec<WeightedFeature>
{
    let mut grouped = match feature {
        Feature::Dense(f) => group_by_x_dense(f, target, indices, dist),
        Feature::Sparse(f) => group_by_x_sparse(f, target, indices, dist),
    };
    grouped.shrink_to_fit();

    grouped
}


fn group_by_x_dense(
    feature: &DenseFeature,
    target: &[f64],
    indices: &[usize],
    dist: &[f64],
) -> Vec<WeightedFeature>
{
    if indices.is_empty() { return Vec::with_capacity(0); }

    let mut indices = indices.to_vec();
    indices.sort_by(|&i, &j|
        feature[i].partial_cmp(&feature[j]).unwrap()
    );

    let mut iter = indices.into_iter();

    let idx = iter.next().unwrap();
    let mut x = feature[idx];
    let mut label_to_weight = HashMap::new();
    let y = target[idx] as i64;
    let weight = label_to_weight.entry(y).or_insert(0.0);
    *weight += dist[idx];

    let mut items = Vec::new();
    while let Some(i) = iter.next() {
        let xi = feature[i];
        let di = dist[i];
        let yi = target[i] as i64;
        if x != xi {
            let f = WeightedFeature::new(x, label_to_weight);
            items.push(f);

            label_to_weight = HashMap::new();
            x = xi;
        }

        let weight = label_to_weight.entry(yi).or_insert(0.0);
        *weight += di;
    }
    items.push(WeightedFeature::new(x, label_to_weight));

    items
}


pub(crate) fn group_by_x_sparse(
    feature: &SparseFeature,
    target: &[f64],
    indices: &[usize],
    dist: &[f64],
) -> Vec<WeightedFeature>
{
    if indices.is_empty() { return Vec::with_capacity(0); }

    let mut zero_map = HashMap::new();


    let mut x_y_d = indices.into_iter()
        .filter_map(|&i| {
            let rx = feature.sample.binary_search_by(|(j, _)| j.cmp(&i));
            let d = dist[i];
            let y = target[i] as i64;
            match rx {
                Ok(ii) => {
                    let (_, x) = feature.sample[ii];
                    Some((x, y, d))
                },
                Err(_) => {
                    let val = zero_map.entry(y as i64).or_insert(0.0);
                    *val += d;
                    None
                }
            }
        })
        .collect::<Vec<(f64, i64, f64)>>();
    x_y_d.sort_by(|(x1, _, _), (x2, _, _)| x1.partial_cmp(&x2).unwrap());

    let mut x_y_d = x_y_d.into_iter();

    let (mut x, y, d) = x_y_d.next().unwrap();
    let mut label_to_weight = HashMap::new();
    label_to_weight.insert(y, d);

    let mut items = Vec::new();
    while let Some((xi, yi, di)) = x_y_d.next() {
        if x != xi {
            let f = WeightedFeature::new(x, label_to_weight);
            items.push(f);

            label_to_weight = HashMap::new();
            label_to_weight.insert(yi, di);
            x = xi;
        } else {
            let weight = label_to_weight.entry(yi).or_insert(0.0);
            *weight += di;
        }
    }
    items.push(WeightedFeature::new(x, label_to_weight));

    let res = items.binary_search_by(|x|
        x.feature_val.partial_cmp(&0.0).unwrap()
    );


    // If there is a zero-valued sample for this feature,
    // insert it to `items`.
    if !zero_map.is_empty() {
        let zero = WeightedFeature::new(0.0, zero_map);
        match res {
            Ok(_) => {
                panic!("Zero-valued feature is not grouped properly");
            },
            Err(idx) => { items.insert(idx, zero); },
        }
    }
    items
}

