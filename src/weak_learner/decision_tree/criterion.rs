//! Defines the inner representation 
//! of the Decision Tree class.

use rayon::prelude::*;

use serde::{Serialize, Deserialize};

use std::fmt;
use std::cmp::Ordering;
use std::ops::{Mul, Add};
use std::collections::HashMap;

use crate::Sample;
use super::bin::*;
use crate::weak_learner::common::{
    type_and_struct::*,
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


impl fmt::Display for Criterion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Entropy => "Entropy",
            Self::Edge => "Edge (Weighted accuracy)",
            Self::Gini => "Gini index",
        };

        write!(f, "{name}")
    }
}


impl Criterion {
    /// Returns the best splitting rule based on the criterion.
    pub(super) fn best_split<'a>(
        &self,
        bins_map: &HashMap<&'a str, Bins>,
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
                    .map(|feature| {
                        let name = feature.name();
                        let bin = bins_map.get(name).unwrap();
                        let pack = bin.pack(idx, feature, target, dist);
                        let (threshold, score) = split_by_entropy(pack);

                        (score, name, threshold)
                    })
                    .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
                    .map(|(_, name, threshold)| (name, threshold))
                    .expect("No feature that decreases the entropic impurity")
            },
            Criterion::Edge => {
                sample.features()
                    .iter()
                    .map(|feature| {
                        let name = feature.name();
                        let bin = bins_map.get(name).unwrap();
                        let pack = bin.pack(idx, feature, target, dist);
                        let (threshold, score) = split_by_edge(pack);

                        (score, name, threshold)
                    })
                    .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
                    .map(|(_, name, threshold)| (name, threshold))
                    .expect("No feature with max edge")
            },
            Criterion::Gini => {
                sample.features()
                    .iter()
                    .map(|feature| {
                        let name = feature.name();
                        let bin = bins_map.get(name).unwrap();
                        let pack = bin.pack(idx, feature, target, dist);
                        let (threshold, score) = split_by_gini(pack);

                        (score, name, threshold)
                    })
                    .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
                    .map(|(_, name, threshold)| (name, threshold))
                    .expect("No feature that minimizes Gini impurity")
            },
        }
    }
}


fn split_by_entropy(pack: Vec<(Bin, LabelToWeight)>)
    -> (f64, EntropicImpurity)
{
    let weight_sum = pack.iter()
        .map(|(_, mp)| mp.values().sum::<f64>())
        .sum::<f64>();


    let mut left_weight = LabelToWeight::new();
    let mut left_weight_sum = 0.0;
    let mut right_weight = LabelToWeight::new();
    let mut right_weight_sum = 0.0;

    for (_, mp) in pack.iter() {
        for (y, w) in mp.iter() {
            let entry = right_weight.entry(*y).or_insert(0.0);
            *entry += w;
            right_weight_sum += w;
        }
    }

    let mut best_score = entropic_impurity(&right_weight);
    let mut best_threshold = f64::MIN;

    for (bin, map) in pack {
        for (y, w) in map {
            let entry = left_weight.entry(y).or_insert(0.0);
            *entry += w;
            left_weight_sum += w;
            let entry = right_weight.get_mut(&y).unwrap();
            *entry -= w;
            right_weight_sum -= w;
        }
        let lp = left_weight_sum / weight_sum;
        let rp = (1.0 - lp).max(0.0);

        let left_impurity = entropic_impurity(&left_weight);
        let right_impurity = entropic_impurity(&right_weight);
        let score = lp * left_impurity + rp * right_impurity;


        if score < best_score {
            best_score = score;
            best_threshold = bin.0.end;
        }
    }
    let best_score = EntropicImpurity::from(best_score);
    (best_threshold, best_score)
}


fn split_by_edge(pack: Vec<(Bin, LabelToWeight)>) -> (f64, Edge) {
    // Compute the edge of the hypothesis that predicts `+1`
    // for all instances.
    let mut edge = pack.iter()
        .map(|(_, map)| {
            map.iter()
                .map(|(y, d)| *y as f64 * d)
                .sum::<f64>()
        })
        .sum::<f64>();

    let mut best_edge = edge.abs();
    let mut best_threshold = f64::MIN;


    for (bin, map) in pack {
        edge -= 2.0 * map.into_iter()
            .map(|(y, d)| y as f64 * d)
            .sum::<f64>();


        if best_edge < edge.abs() {
            best_edge = edge.abs();
            best_threshold = bin.0.end;
        }
    }
    let best_edge = Edge::from(best_edge);
    (best_threshold, best_edge)
}


fn split_by_gini(pack: Vec<(Bin, LabelToWeight)>) -> (f64, Gini) {
    let weight_sum = pack.iter()
        .map(|(_, mp)| mp.values().sum::<f64>())
        .sum::<f64>();


    let mut left_weight = LabelToWeight::new();
    let mut left_weight_sum = 0.0;
    let mut right_weight = LabelToWeight::new();
    let mut right_weight_sum = 0.0;

    for (_, mp) in pack.iter() {
        for (y, w) in mp.iter() {
            let entry = right_weight.entry(*y).or_insert(0.0);
            *entry += w;
            right_weight_sum += w;
        }
    }

    let mut best_score = gini_impurity(&right_weight);
    let mut best_threshold = f64::MIN;

    for (bin, map) in pack {
        for (y, w) in map {
            let entry = left_weight.entry(y).or_insert(0.0);
            *entry += w;
            left_weight_sum += w;
            let entry = right_weight.get_mut(&y).unwrap();
            *entry -= w;
            right_weight_sum -= w;
        }
        let lp = left_weight_sum / weight_sum;
        let rp = (1.0 - lp).max(0.0);

        let left_impurity = gini_impurity(&left_weight);
        let right_impurity = gini_impurity(&right_weight);
        let score = lp * left_impurity + rp * right_impurity;


        if score < best_score {
            best_score = score;
            best_threshold = bin.0.end;
        }
    }
    let best_score = Gini::from(best_score);
    (best_threshold, best_score)
}


/// Returns the entropic-impurity of the given map.
#[inline(always)]
pub(self) fn entropic_impurity(map: &HashMap<i32, f64>) -> f64 {
    let total = map.values().sum::<f64>();
    if total <= 0.0 || map.is_empty() { return 0.0.into(); }

    map.par_iter()
        .map(|(_, &p)| {
            let r = p / total;
            if r <= 0.0 { 0.0 } else { -r * r.ln() }
        })
        .sum::<f64>()
}


/// Returns the gini-impurity of the given map.
#[inline(always)]
pub(self) fn gini_impurity(map: &HashMap<i32, f64>) -> f64 {
    let total = map.values().sum::<f64>();
    if total <= 0.0 || map.is_empty() { return 0.0.into(); }

    let correct = map.par_iter()
        .map(|(_, &w)| (w / total).powi(2))
        .sum::<f64>();

    (1.0 - correct).max(0.0)
}
