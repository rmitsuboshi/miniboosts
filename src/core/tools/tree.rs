//! Small types used to build and evaluate decision trees.
//!
//! Most are thin `f64`/`usize` newtypes ([`Prediction`], [`Confidence`],
//! [`LossValue`], [`Depth`], [`FeatureValue`], [`Mass`], [`Target`]) that
//! make tree code self-describing. [`Splitter`] is the threshold rule stored
//! in each branch of a [`crate::DecisionTreeClassifier`].
use crate::core::Sample;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::{cmp, fmt, ops};

/// A map from a label (converted to `i32`) to the total nonnegative weight
/// of the examples with that label.
pub type LabelToWeight = HashMap<i32, f64>;

/// A predicted value (label or regression output) of type `T`.
/// Supports `+` when `T` does.
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
where
    T: ops::Add<Output = T>,
{
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

/// A confidence value of type `T`; for classifiers in this crate, an `f64`
/// in `[-1, 1]`. Supports `+` when `T` does.
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
where
    T: ops::Add<Output = T>,
{
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

/// A loss value. Supports `+`, comparison, and equality with `f64`.
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

/// The remaining depth budget of a tree node, a wrapper for `usize`.
///
/// The root of a tree with maximal depth `d` is grown with `Depth::from(d)`;
/// each child receives one less, and a node with remaining depth `0`
/// becomes a leaf.
///
/// # Examples
/// ```
/// use miniboosts::core::tree::Depth;
///
/// let depth = Depth::from(2);
/// assert!(depth - 1 == 1);
/// assert!(depth - 2 < 1); // a leaf
/// assert_eq!(Depth::from(0) - 1, Depth::from(0)); // saturates at zero
/// assert_eq!(depth.to_string(), "2");
/// ```
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
    /// Subtracts `other` from the depth.
    /// A depth of `0` stays `0`. For a positive depth, plain `usize`
    /// subtraction is used, so subtracting more than the current depth
    /// overflows (panics in debug builds).
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

/// A feature value. Supports `+`, comparison, and equality with `f64`.
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

/// A probability mass (sum of example weights).
/// Supports `+` and equality with `f64`.
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

/// A target (label) value. Supports `+`, comparison, and equality with
/// `f64`.
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

/// The branch chosen by [`Splitter::split`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeftRight {
    /// The feature value is below the threshold.
    Left,
    /// The feature value is at least the threshold (or is NaN).
    Right,
}

/// A threshold rule on a single feature: an example goes left iff
/// `x[feature] < threshold`.
///
/// # Examples
/// ```
/// use miniboosts::Sample;
/// use miniboosts::core::tree::{LeftRight, Splitter};
///
/// let rows = vec![vec![0.0], vec![0.5], vec![f64::MAX]];
/// let sample = Sample::from_dense(&rows, None).unwrap();
/// let rule = Splitter::new("feature_0", 0.5);
/// assert_eq!(rule.split(&sample, 0), LeftRight::Left);
/// assert_eq!(rule.split(&sample, 1), LeftRight::Right); // ties go right
/// assert_eq!(rule.split(&sample, 2), LeftRight::Right);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Splitter {
    /// The name of the feature column tested.
    pub feature: String,
    /// The threshold; values strictly below it go left.
    pub threshold: f64,
}

impl Splitter {
    /// Creates the rule `x[name] < threshold`.
    #[inline]
    pub fn new(name: &str, threshold: f64) -> Self {
        let feature = name.to_string();
        Self { feature, threshold }
    }

    /// Returns [`LeftRight::Left`] if the value of feature `self.feature` at
    /// `row` of `sample` is `< self.threshold`, and [`LeftRight::Right`]
    /// otherwise (including NaN).
    ///
    /// # Panics
    /// Panics if `sample` has no feature named `self.feature`, or if `row` is
    /// out of bounds for a dense feature.
    #[inline]
    pub fn split(&self, sample: &Sample, row: usize) -> LeftRight {
        let name = &self.feature;

        let value = sample[name][row];

        if value < self.threshold {
            LeftRight::Left
        } else {
            LeftRight::Right
        }
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
