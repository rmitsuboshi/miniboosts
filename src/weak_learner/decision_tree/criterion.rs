//! Defines the inner representation 
//! of the Decision Tree class.

use rayon::prelude::*;

use serde::{Serialize, Deserialize};

use std::cmp::Ordering;
use std::ops::{Mul, Add};
use std::collections::HashMap;

use crate::{Sample, Feature};


/// Edge
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub(super) struct Edge(f64);


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
pub(super) struct Impurity(f64);


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
// * `Criterion::Gini` is the gini-index,

/// Splitting criteria for growing decision tree.
/// * `Criterion::Edge` maximizes the edge (weighted training accuracy)
///     for given distribution.
/// * `Criterion::Entropy` minimizes entropic impurity for given distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Criterion {
    /// Binary entropy function.
    Entropy,
    /// Weighted accuracy.
    /// This criterion is designed for binary classification problems.
    Edge,
    // /// Gini index.
    // Gini,
    // /// Twoing rule.
    // Twoing,
}


impl Criterion {
    /// Returns the best splitting rule based on the criterion.
    pub(super) fn best_split<'a>(
        &self,
        sample: &'a Sample,
        dist: &[f64],
        idx: &[usize],
    ) -> (&'a str, f64)
    {
        let target = sample.target();
        match self {
            Criterion::Entropy => {
                sample.features()
                    .into_iter()
                    .map(|column| {
                        let (threshold, decrease) = split_entropy(
                            column, target, dist, &idx[..]
                        );

                        (decrease, column.name(), threshold)
                    })
                    .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
                    .map(|(_, name, threshold)| (name, threshold))
                    .expect("No feature that decreases the entropic impurity")
            },
            Criterion::Edge => {
                sample.features()
                    .into_iter()
                    .map(|column| {
                        let (threshold, decrease) = split_edge(
                            column, target, dist, &idx[..]
                        );

                        (decrease, column.name(), threshold)
                    })
                    .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
                    .map(|(_, name, threshold)| (name, threshold))
                    .expect("No feature with max edge")
            }
        }
    }
}


fn split_entropy(
    column: &Feature,
    target: &[f64],
    dist: &[f64],
    idx: &[usize]
) -> (f64, Impurity)
{
    let mut triplets = idx.into_par_iter()
        .copied()
        .map(|i| {
            let x = column[i];
            let y = target[i] as i64;
            (x, dist[i], y)
        })
        .collect::<Vec<(f64, f64, i64)>>();
    triplets.sort_by(|(x1, _, _), (x2, _, _)| x1.partial_cmp(x2).unwrap());


    let total_weight = triplets.par_iter()
        .map(|(_, d, _)| d)
        .sum::<f64>();


    let mut left = TempNodeInfo::empty();
    let mut right = TempNodeInfo::new(&triplets[..]);


    let mut iter = triplets.into_iter().peekable();


    // These variables are used for the best splitting rules.
    let mut best_decrease = right.entropic_impurity();
    let mut best_threshold = iter.peek()
        .map(|(v, _, _)| *v - 2.0_f64)
        .unwrap_or(f64::MIN);

    while let Some((old_val, d, y)) = iter.next() {
        left.insert(y, d);
        right.delete(y, d);


        while let Some(&(xx, dd, yy)) = iter.peek() {
            if xx != old_val { break; }

            left.insert(yy, dd);
            right.delete(yy, dd);

            iter.next();
        }

        let new_val = iter.peek()
            .map(|(xx, _, _)| *xx)
            .unwrap_or(old_val + 2.0_f64);

        let threshold = (old_val + new_val) / 2.0;

        assert!(total_weight > 0.0);

        let lp = left.total / total_weight;
        let rp = 1.0 - lp;


        let decrease = Impurity::from(lp) * left.entropic_impurity()
            + Impurity::from(rp) * right.entropic_impurity();


        if decrease < best_decrease {
            best_decrease = decrease;
            best_threshold = threshold;
        }
    }



    (best_threshold, best_decrease)
}


fn split_edge(
    column: &Feature,
    target: &[f64],
    dist: &[f64],
    idx: &[usize]
)
    -> (f64, Edge)
{
    let mut triplets = idx.into_par_iter()
        .copied()
        .map(|i| {
            let x = column[i];
            let y = target[i];
            (x, dist[i], y)
        })
        .collect::<Vec<(f64, f64, f64)>>();
    triplets.sort_by(|(x1, _, _), (x2, _, _)| x1.partial_cmp(x2).unwrap());


    let mut best_edge;
    let mut best_threshold;
    // Compute the edge of the hypothesis that predicts `+1`
    // for all instances.
    let mut edge = triplets.iter()
        .map(|(_, d, y)| *d * *y)
        .sum::<f64>();


    let mut iter = triplets.into_iter().peekable();


    // best threshold is the smallest value.
    // we define the initial threshold as the smallest value minus 1.0
    best_threshold = iter.peek()
        .map(|(v, _, _)| *v - 1.0_f64)
        .unwrap_or(f64::MIN);

    // best edge
    best_edge = edge.abs();

    while let Some((left, d, y)) = iter.next() {
        edge -= 2.0 * d * y;


        while let Some(&(xx, dd, yy)) = iter.peek() {
            if xx != left { break; }

            edge -= 2.0 * dd * yy;

            iter.next();
        }

        let right = iter.peek()
            .map(|(xx, _, _)| *xx)
            .unwrap_or(left + 2.0_f64);

        let threshold = (left + right) / 2.0;


        if best_edge < edge.abs() {
            best_edge = edge.abs();
            best_threshold = threshold;
        }
    }

    (best_threshold, Edge::from(best_edge))
}


/// Some information that are useful in `produce(..)`.
struct TempNodeInfo {
    map: HashMap<i64, f64>,
    total: f64,
}


impl TempNodeInfo {
    /// Build an empty instance of `TempNodeInfo`.
    #[inline(always)]
    pub(self) fn empty() -> Self {
        Self {
            map: HashMap::new(),
            total: 0.0_f64,
        }
    }


    /// Build an instance of `TempNodeInfo`.
    #[inline(always)]
    pub(self) fn new(triplets: &[(f64, f64, i64)]) -> Self {
        let mut total = 0.0_f64;
        let mut map: HashMap<i64, f64> = HashMap::new();
        triplets.iter()
            .for_each(|(_, d, y)| {
                total += *d;
                let cnt = map.entry(*y).or_insert(0.0);
                *cnt += *d;
            });

        Self { map, total }
    }


    /// Returns the impurity of this node.
    #[inline(always)]
    pub(self) fn entropic_impurity(&self) -> Impurity {
        if self.total == 0.0 || self.map.is_empty() {
            return 0.0.into();
        }

        self.map.par_iter()
            .map(|(_, &p)| {
                let r = p / self.total;
                if r == 0.0 { 0.0 } else { -r * r.ln() }
            })
            .sum::<f64>()
            .into()
    }


    /// Increase the number of positive examples by one.
    pub(self) fn insert(&mut self, y: i64, weight: f64) {
        let cnt = self.map.entry(y).or_insert(0.0);
        *cnt += weight;
        self.total += weight;
    }


    /// Decrease the number of positive examples by one.
    pub(self) fn delete(&mut self, y: i64, weight: f64) {
        if let Some(key) = self.map.get_mut(&y) {
            *key -= weight;
            self.total -= weight;
        }
    }
}
