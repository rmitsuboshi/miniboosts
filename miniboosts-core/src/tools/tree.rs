//! This file defines some tools for tree algorithms
use serde::{Serialize, Deserialize};
use std::{fmt, cmp, ops};
use std::collections::HashMap;
use crate::Sample;

/// This is an alias from label of type `i32`
/// to non-negative weight of type `f64`.
pub type LabelToWeight = HashMap<i32, f64>;

#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Prediction<T>(pub T);

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
pub struct Confidence<T>(pub T);

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
pub struct LossValue(pub f64);

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
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Depth(usize);

impl fmt::Display for Depth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let depth = self.0;
        write!(f, "{depth}")
    }
}

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
        if self.0 < 1 {
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
pub struct FeatureValue(pub f64);

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
pub struct Mass(pub f64);

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
pub struct Target(pub f64);

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

/// The output of the function `split` of `SplitRule`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeftRight {
    Left,
    Right,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Splitter {
    pub feature: String,
    pub threshold: f64,
}

impl Splitter {
    #[inline]
    pub fn new(name: &str, threshold: f64) -> Self {
        let feature = name.to_string();
        Self {
            feature,
            threshold
        }
    }

    /// Defines the splitting.
    #[inline]
    pub fn split(&self, sample: &Sample, row: usize) -> LeftRight {
        let name = &self.feature;

        let value = sample[name][row];

        if value < self.threshold { LeftRight::Left } else { LeftRight::Right }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_add_01() {
        let p1 = Prediction::from(1.0);
        let p2 = Prediction::from(1.0);
        let res = p1 + p2;
        let exp = Prediction::from(2.0);
        assert_eq!(exp, res, "expected {exp:?}, got {res:?}.");
    }

    #[test]
    fn test_prediction_add_02() {
        let p1 = Prediction::from(1.0);
        let p2 = Prediction::from(-1.0);
        let res = p1 + p2;
        let exp = Prediction::from(0.0);
        assert_eq!(exp, res, "expected {exp:?}, got {res:?}.");
    }

    #[test]
    fn test_confidence_add_01() {
        let c1 = Confidence::from(1.0);
        let c2 = Confidence::from(1.0);
        let res = c1 + c2;
        let exp = Confidence::from(2.0);
        assert_eq!(exp, res, "expected {exp:?}, got {res:?}.");
    }

    #[test]
    fn test_confidence_add_02() {
        let c1 = Confidence::from(1.0);
        let c2 = Confidence::from(-1.0);
        let res = c1 + c2;
        let exp = Confidence::from(0.0);
        assert_eq!(exp, res, "expected {exp:?}, got {res:?}.");
    }

    #[test]
    fn test_depth_sub_01() {
        let d1 = Depth::from(3);
        let res = d1 - 1;
        let exp = Depth::from(2);
        assert_eq!(exp, res, "expected {exp:?}, got {res:?}.");
    }

    #[test]
    fn test_depth_sub_02() {
        let d1 = Depth::from(0);
        let res = d1 - 1;
        let exp = Depth::from(0);
        assert_eq!(exp, res, "expected {exp:?}, got {res:?}.");
    }

    #[test]
    fn test_depth_cmp_01() {
        let d1 = Depth::from(2);
        let rhs = 3;
        let res = d1 < rhs;
        assert!(res, "failed for {d1:?} < {rhs}. got {res}.");
    }

    #[test]
    fn test_depth_cmp_02() {
        let d1 = Depth::from(2);
        let rhs = 0;
        let res = !(d1 < rhs);
        assert!(res, "failed for !({d1:?} < {rhs}). got {res}.");
    }

    #[test]
    fn test_depth_cmp_03() {
        let d1 = Depth::from(2);
        let rhs = 2;
        let res = !(d1 < rhs);
        assert!(res, "failed for !({d1:?} < {rhs}). got {res}.");
    }

    #[test]
    fn test_depth_cmp_04() {
        let d1 = Depth::from(2);
        let rhs = 2;
        let res = d1 <= rhs;
        assert!(res, "failed for {d1:?} <= {rhs}. got {res}.");
    }
}

