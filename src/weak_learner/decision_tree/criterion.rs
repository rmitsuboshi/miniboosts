//! Defines the inner representation 
//! of the Decision Tree class.

use rayon::prelude::*;

use serde::{Serialize, Deserialize};

use std::fmt;
use std::cmp::Ordering;
use std::ops::{Mul, Add};
use std::collections::{HashSet, HashMap};

use crate::Sample;
use super::bin::*;
use crate::weak_learner::common::{
    type_and_struct::*,
};



/// Score for a splitting.
/// This is just a wrapper for `f64`.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub(self) struct Score(f64);


impl From<f64> for Score {
    #[inline(always)]
    fn from(score: f64) -> Self {
        Self(score)
    }
}


impl PartialEq for Score {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}


impl PartialOrd for Score {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}


impl Mul for Score {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self::Output {
        Self(self.0 * other.0)
    }
}


impl Add for Score {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self::Output {
        Self(self.0 + other.0)
    }
}


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
    /// Twoing rule.
    Twoing,
}


impl fmt::Display for Criterion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Entropy => "Entropy",
            Self::Edge => "Edge (Weighted accuracy)",
            Self::Gini => "Gini index",
            Self::Twoing => "Twoing Rule",
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
                    .par_iter()
                    .map(|feature| {
                        let name = feature.name();
                        let bin = bins_map.get(name).unwrap();
                        let pack = bin.pack(idx, feature, target, dist);
                        let (threshold, score) = split_by_entropy(pack);

                        (score, name, threshold)
                    })
                    .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
                    .map(|(_, name, threshold)| (name, threshold))
                    .expect("No feature minimizes entropic impurity")
            },
            Criterion::Edge => {
                sample.features()
                    .par_iter()
                    .map(|feature| {
                        let name = feature.name();
                        let bin = bins_map.get(name).unwrap();
                        let pack = bin.pack(idx, feature, target, dist);
                        let (threshold, score) = split_by_edge(pack);

                        (score, name, threshold)
                    })
                    .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
                    .map(|(_, name, threshold)| (name, threshold))
                    .expect("No feature maximizes edge")
            },
            Criterion::Gini => {
                sample.features()
                    .par_iter()
                    .map(|feature| {
                        let name = feature.name();
                        let bin = bins_map.get(name).unwrap();
                        let pack = bin.pack(idx, feature, target, dist);
                        let (threshold, score) = split_by_gini(pack);

                        (score, name, threshold)
                    })
                    .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
                    .map(|(_, name, threshold)| (name, threshold))
                    .expect("No feature minimizes Gini impurity")
            },
            Criterion::Twoing => {
                sample.features()
                    .par_iter()
                    .map(|feature| {
                        let name = feature.name();
                        let bin = bins_map.get(name).unwrap();
                        let pack = bin.pack(idx, feature, target, dist);
                        let (threshold, score) = split_by_twoing(pack);

                        (score, name, threshold)
                    })
                    .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
                    .map(|(_, name, threshold)| (name, threshold))
                    .expect("No feature maximizes Twoing rule")
            },
        }
    }
}


fn split_by_entropy(pack: Vec<(Bin, LabelToWeight)>)
    -> (f64, Score)
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
    let best_score = Score::from(best_score);
    (best_threshold, best_score)
}


fn split_by_edge(pack: Vec<(Bin, LabelToWeight)>) -> (f64, Score) {
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
    let best_edge = Score::from(best_edge);
    (best_threshold, best_edge)
}


fn split_by_gini(pack: Vec<(Bin, LabelToWeight)>) -> (f64, Score) {
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


            if *entry <= 0.0 { right_weight.remove(&y); }
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
    let best_score = Score::from(best_score);
    (best_threshold, best_score)
}


fn split_by_twoing(pack: Vec<(Bin, LabelToWeight)>) -> (f64, Score) {
    let mut left_weight = LabelToWeight::new();
    let mut right_weight = LabelToWeight::new();

    let mut labels = HashSet::new();
    for (_, mp) in pack.iter() {
        for (y, w) in mp.iter() {
            let entry = right_weight.entry(*y).or_insert(0.0);
            *entry += w;

            labels.insert(*y);
        }
    }

    let mut best_score = 0.0;
    let mut best_threshold = f64::MIN;

    for (bin, map) in pack {
        // Move the weights in a `pack` from right to left.
        for (y, w) in map {
            let entry = left_weight.entry(y).or_insert(0.0);
            *entry += w;
            let entry = right_weight.get_mut(&y).unwrap();
            *entry -= w;

            if *entry <= 0.0 { right_weight.remove(&y); }
        }


        let score = twoing_score(&labels, &left_weight, &right_weight);


        if score > best_score {
            best_score = score;
            best_threshold = bin.0.end;
        }
    }
    let best_score = Score::from(best_score);
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


/// Returns the gini-impurity of the given map.
#[inline(always)]
pub(self) fn twoing_score(
    labels: &HashSet<i32>,
    left: &HashMap<i32, f64>,
    right: &HashMap<i32, f64>,
) -> f64
{
    let pl = left.values().sum::<f64>();
    let pr = right.values().sum::<f64>();
    let pt = pl + pr;

    if pl == 0.0 || pr == 0.0 { return 0.0; }
    assert!(pt > 0.0);

    let mut score = 0.0;
    for y in labels {
        let l = left.get(y).unwrap_or(&0.0);
        let r = right.get(y).unwrap_or(&0.0);

        score += ((l / pl) - (r / pr)).abs();
    }
    score = score.powi(2) * pl * pr / (2.0 * pt).powi(2);

    score
}
