use serde::{Serialize, Deserialize};
use std::ops;
use std::cmp;

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


