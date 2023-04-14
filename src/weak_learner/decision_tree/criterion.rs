//! Defines the inner representation 
//! of the Decision Tree class.

use rayon::prelude::*;

use serde::{Serialize, Deserialize};

use std::cmp::Ordering;
use std::ops::{Mul, Add};
use std::collections::HashMap;

use crate::Sample;
use crate::weak_learner::common::{
    group_by_x,
    WeightedFeature,
};


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


/// Gini
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub(super) struct Gini(f64);


impl From<f64> for Gini {
    #[inline(always)]
    fn from(gini: f64) -> Self {
        Self(gini)
    }
}


impl PartialEq for Gini {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}


impl PartialOrd for Gini {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}


impl Mul for Gini {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self::Output {
        Self(self.0 * other.0)
    }
}


impl Add for Gini {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self::Output {
        Self(self.0 + other.0)
    }
}



/// Entropic Impurity
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub(super) struct EntropicImpurity(f64);


impl From<f64> for EntropicImpurity {
    #[inline(always)]
    fn from(impurity: f64) -> Self {
        EntropicImpurity(impurity)
    }
}


impl PartialEq for EntropicImpurity {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}


impl PartialOrd for EntropicImpurity {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}


impl Mul for EntropicImpurity {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self::Output {
        Self(self.0 * other.0)
    }
}


impl Add for EntropicImpurity {
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
    /// Gini index.
    Gini,
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
        let target = &target[..];
        match self {
            Criterion::Entropy => {
                sample.features()
                    .iter()
                    .map(|column| {
                        let items = group_by_x(column, target, idx, dist);
                        let (threshold, decrease) = split_by_entropy(items);

                        (decrease, column.name(), threshold)
                    })
                    .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
                    .map(|(_, name, threshold)| (name, threshold))
                    .expect("No feature that decreases the entropic impurity")
            },
            Criterion::Edge => {
                sample.features()
                    .iter()
                    .map(|column| {
                        let items = group_by_x(column, target, idx, dist);
                        let (threshold, decrease) = split_by_edge(items);

                        (decrease, column.name(), threshold)
                    })
                    .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
                    .map(|(_, name, threshold)| (name, threshold))
                    .expect("No feature with max edge")
            },
            Criterion::Gini => {
                sample.features()
                    .iter()
                    .map(|column| {
                        let items = group_by_x(column, target, idx, dist);
                        let (threshold, decrease) = split_by_gini(items);

                        (decrease, column.name(), threshold)
                    })
                    .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
                    .map(|(_, name, threshold)| (name, threshold))
                    .expect("No feature that minimizes Gini impurity")
            },
        }
    }
}


fn split_by_entropy(ws: Vec<WeightedFeature>) -> (f64, EntropicImpurity) {
    let total_weight = ws.iter()
        .map(|wf| wf.total_weight())
        .sum::<f64>();

    let mut left = ImpurityKeeper::empty();
    let mut right = ImpurityKeeper::new(&ws[..]);


    let mut iter = ws.into_iter().peekable();
    // These variables are used for the best splitting rules.
    let mut best_decrease = right.entropic_impurity();
    let mut best_threshold = iter.peek()
        .map(|wf| wf.feature_val - 1.0_f64)
        .unwrap_or(f64::MIN);

    while let Some(wf) = iter.next() {
        let curr_x = wf.feature_val;
        left.insert(&wf);
        right.delete(&wf);


        let next_x = iter.peek()
            .map(|next_wf| next_wf.feature_val)
            .unwrap_or(curr_x + 2.0_f64);

        let threshold = (curr_x + next_x) / 2.0;

        assert!(total_weight > 0.0);

        let lp = left.total / total_weight;
        let rp = (1.0 - lp).max(0.0);


        let decrease = EntropicImpurity::from(lp) * left.entropic_impurity()
            + EntropicImpurity::from(rp) * right.entropic_impurity();


        if decrease < best_decrease {
            best_decrease = decrease;
            best_threshold = threshold;
        }
    }


    (best_threshold, best_decrease)
}


fn split_by_edge(ws: Vec<WeightedFeature>) -> (f64, Edge) {
    // Compute the edge of the hypothesis that predicts `+1`
    // for all instances.
    let mut edge = ws.iter()
        .map(|wf| {
            wf.label_to_weight.iter()
                .map(|(y, d)| *y as f64 * d)
                .sum::<f64>()
        })
        .sum::<f64>();
    let mut iter = ws.into_iter().peekable();


    let mut best_edge;
    let mut best_threshold;
    // best threshold is the smallest value.
    // we define the initial threshold as the smallest value minus 1.0
    best_threshold = iter.peek()
        .map(|wf| wf.feature_val - 1.0_f64)
        .unwrap_or(f64::MIN);

    // best edge
    best_edge = edge.abs();

    while let Some(wf) = iter.next() {
        let left = wf.feature_val;
        edge -= 2.0 * wf.label_to_weight.iter()
            .map(|(y, d)| *y as f64 * d)
            .sum::<f64>();


        let right = iter.peek()
            .map(|wf_next| wf_next.feature_val)
            .unwrap_or(left + 2.0_f64);

        let threshold = (left + right) / 2.0;


        if best_edge < edge.abs() {
            best_edge = edge.abs();
            best_threshold = threshold;
        }
    }

    (best_threshold, Edge::from(best_edge))
}


fn split_by_gini(ws: Vec<WeightedFeature>) -> (f64, Gini) {
    let total_weight = ws.iter()
        .map(|wf| wf.total_weight())
        .sum::<f64>();

    let mut left = ImpurityKeeper::empty();
    let mut right = ImpurityKeeper::new(&ws[..]);


    let mut iter = ws.into_iter().peekable();
    // These variables are used for the best splitting rules.
    let mut best_decrease = right.gini_impurity();
    let mut best_threshold = iter.peek()
        .map(|wf| wf.feature_val - 1.0_f64)
        .unwrap_or(f64::MIN);

    while let Some(wf) = iter.next() {
        let curr_x = wf.feature_val;
        left.insert(&wf);
        right.delete(&wf);


        let next_x = iter.peek()
            .map(|next_wf| next_wf.feature_val)
            .unwrap_or(curr_x + 2.0_f64);

        let threshold = (curr_x + next_x) / 2.0;

        assert!(total_weight > 0.0);

        let lp = left.total / total_weight;
        let rp = (1.0 - lp).max(0.0);


        let decrease = Gini::from(lp) * left.gini_impurity()
            + Gini::from(rp) * right.gini_impurity();


        if decrease < best_decrease {
            best_decrease = decrease;
            best_threshold = threshold;
        }
    }


    (best_threshold, best_decrease)
}


/// Some information that are useful in `produce(..)`.
struct ImpurityKeeper {
    map: HashMap<i64, f64>,
    total: f64,
}


impl ImpurityKeeper {
    /// Build an empty instance of `ImpurityKeeper`.
    #[inline(always)]
    pub(self) fn empty() -> Self {
        Self {
            map: HashMap::new(),
            total: 0.0_f64,
        }
    }


    /// Build an instance of `ImpurityKeeper`.
    #[inline(always)]
    pub(self) fn new(ws: &[WeightedFeature]) -> Self {
        let mut total = 0.0_f64;
        let mut map: HashMap<i64, f64> = HashMap::new();
        ws.iter()
            .for_each(|wf| {
                total += wf.total_weight();
                wf.label_to_weight.iter()
                    .for_each(|(y, d)| {
                        let cnt = map.entry(*y).or_insert(0.0);
                        *cnt += d;
                    });
            });

        Self { map, total }
    }


    /// Returns the entropic-impurity of this node.
    #[inline(always)]
    pub(self) fn entropic_impurity(&self) -> EntropicImpurity {
        if self.total <= 0.0 || self.map.is_empty() { return 0.0.into(); }

        self.map.par_iter()
            .map(|(_, &p)| {
                let r = p / self.total;
                if r <= 0.0 { 0.0 } else { -r * r.ln() }
            })
            .sum::<f64>()
            .into()
    }


    /// Returns the gini-impurity of this node.
    #[inline(always)]
    pub(self) fn gini_impurity(&self) -> Gini {
        if self.total <= 0.0 || self.map.is_empty() { return 0.0.into(); }

        let correct = self.map.par_iter()
            .map(|(_, &w)| (w / self.total).powi(2))
            .sum::<f64>();

        Gini::from(1.0 - correct)
    }


    /// Increase the number of positive examples by one.
    pub(self) fn insert(&mut self, item: &WeightedFeature) {
        item.label_to_weight.iter()
            .for_each(|(y, d)| {
                let dd = self.map.entry(*y).or_insert(0.0);
                *dd += d;
            });
        self.total += item.total_weight();
    }


    /// Decrease the number of positive examples by one.
    pub(self) fn delete(&mut self, item: &WeightedFeature) {
        let mut to_be_removed = Vec::new();
        item.label_to_weight.iter()
            .for_each(|(y, d)| {
                if let Some(val) = self.map.get_mut(y) {
                    *val -= d;
                    self.total -= d;

                    if *val <= 0.0 { to_be_removed.push(y); }
                }
            });

        to_be_removed.into_iter()
            .for_each(|y| { self.map.remove(y); });
    }
}
