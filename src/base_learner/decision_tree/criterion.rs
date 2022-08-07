//! Defines the inner representation 
//! of the Decision Tree class.


use serde::{Serialize, Deserialize};

use std::cmp::Ordering;
use std::ops::{Mul, Add};


/// Edge
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Edge(f64);


impl From<f64> for Edge {
    #[inline(always)]
    fn from(edge: f64) -> Self {
        Edge(edge)
    }
}


impl PartialEq for Edge {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}


impl PartialOrd for Edge {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}



/// Impurity
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Impurity(f64);


impl From<f64> for Impurity {
    #[inline(always)]
    fn from(impurity: f64) -> Self {
        Impurity(impurity)
    }
}


impl PartialEq for Impurity {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}


impl PartialOrd for Impurity {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}


impl Mul for Impurity {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self::Output {
        Self(self.0 * other.0)
    }
}


impl Add for Impurity {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self::Output {
        Self(self.0 + other.0)
    }
}


// TODO
//      Add other criterions.
//      E.g., Gini criterion, Twoing criterion (page 38 of CART)
/// Maximization objectives.
/// * `Criterion::Gini` is the gini-index,
/// * `Criterion::Entropy` is the entropy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Criterion {
    /// Binary entropy function.
    Entropy,
    /// Weighted accuracy.
    Edge,
    // /// Gini index.
    // Gini,
    // /// Twoing rule.
    // Twoing,
}

