use rayon::prelude::*;

use serde::{Serialize, Deserialize};

use std::fmt;
use std::cmp::Ordering;
use std::ops::{Mul, Add};
use std::collections::{HashSet, HashMap};

use miniboosts_core::{
    binning::*,
    Sample,
};

/// Score for a splitting.
/// This is just a wrapper for `f64`.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
struct Score(f64);

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitBy {
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

impl fmt::Display for SplitBy {
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

impl SplitBy {
    /// Returns the best splitting rule based on the criterion.
    pub(super) fn best_split<'a>(
        &self,
        bins_map: &HashMap<&'a str, Bins>,
        sample:   &'a Sample,
        dist:     &[f64],
        idx:      &[usize],
    ) -> (&'a str, f64)
    {
        let target = sample.target();
        match self {
            SplitBy::Entropy => {
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
            SplitBy::Edge => {
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
            SplitBy::Gini => {
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
            SplitBy::Twoing => {
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

fn split_by_entropy(pack: Vec<(Bin, HashMap<i32, f64>)>)
    -> (f64, Score)
{
    let weight_sum = pack.iter()
        .map(|(_, mp)| mp.values().sum::<f64>())
        .sum::<f64>();

    let mut left_weight = HashMap::<i32, f64>::new();
    let mut left_weight_sum = 0f64;
    let mut right_weight = HashMap::<i32, f64>::new();

    for (_, mp) in pack.iter() {
        for (y, w) in mp.iter() {
            let entry = right_weight.entry(*y).or_insert(0f64);
            *entry += w;
        }
    }

    let mut best_score = entropic_impurity(&right_weight);
    let mut best_threshold = f64::MIN;

    for (bin, map) in pack {
        for (y, w) in map {
            let entry = left_weight.entry(y).or_insert(0f64);
            *entry += w;
            left_weight_sum += w;
            let entry = right_weight.get_mut(&y).unwrap();
            *entry -= w;
        }
        let lp = left_weight_sum / weight_sum;
        let rp = (1f64 - lp).max(0f64);

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

fn split_by_edge(pack: Vec<(Bin, HashMap<i32, f64>)>) -> (f64, Score) {
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
        edge -= 2f64 * map.into_iter()
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

fn split_by_gini(pack: Vec<(Bin, HashMap<i32, f64>)>) -> (f64, Score) {
    let weight_sum = pack.iter()
        .map(|(_, mp)| mp.values().sum::<f64>())
        .sum::<f64>();

    let mut left_weight = HashMap::<i32, f64>::new();
    let mut left_weight_sum = 0f64;
    let mut right_weight = HashMap::<i32, f64>::new();

    for (_, mp) in pack.iter() {
        for (y, w) in mp.iter() {
            let entry = right_weight.entry(*y).or_insert(0f64);
            *entry += w;
        }
    }

    let mut best_score = gini_impurity(&right_weight);
    let mut best_threshold = f64::MIN;

    for (bin, map) in pack {
        for (y, w) in map {
            let entry = left_weight.entry(y).or_insert(0f64);
            *entry += w;
            left_weight_sum += w;
            if let Some(entry) = right_weight.get_mut(&y) {
                *entry -= w;
                if *entry <= 0f64 { right_weight.remove(&y); }
            }
        }
        let lp = left_weight_sum / weight_sum;
        let rp = (1f64 - lp).max(0f64);

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

fn split_by_twoing(pack: Vec<(Bin, HashMap<i32, f64>)>) -> (f64, Score) {
    let mut left_weight = HashMap::<i32, f64>::new();
    let mut right_weight = HashMap::<i32, f64>::new();

    let mut labels = HashSet::new();
    for (_, mp) in pack.iter() {
        for (y, w) in mp.iter() {
            let entry = right_weight.entry(*y).or_insert(0f64);
            *entry += w;

            labels.insert(*y);
        }
    }

    let mut best_score = f64::MIN;
    let mut best_threshold = f64::MIN;

    for (bin, map) in pack {
        // Move the weights in a `pack` from right to left.
        for (y, w) in map {
            let entry = left_weight.entry(y).or_insert(0f64);
            *entry += w;
            if let Some(entry) = right_weight.get_mut(&y) {
                *entry -= w;
                if *entry <= 0f64 { right_weight.remove(&y); }
            }
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
fn entropic_impurity(map: &HashMap<i32, f64>) -> f64 {
    let total = map.values().sum::<f64>();
    if total <= 0f64 || map.is_empty() { return 0f64; }

    map.par_iter()
        .map(|(_, &p)| {
            let r = p / total;
            if r <= 0f64 { 0f64 } else { -r * r.ln() }
        })
        .sum::<f64>()
}

/// Returns the gini-impurity of the given map.
#[inline(always)]
fn gini_impurity(map: &HashMap<i32, f64>) -> f64 {
    let total = map.values().sum::<f64>();
    if total <= 0f64 || map.is_empty() { return 0f64; }

    let correct = map.par_iter()
        .map(|(_, &w)| (w / total).powi(2))
        .sum::<f64>();

    (1f64 - correct).max(0f64)
}

/// Returns the twoing score of the given map.
#[inline(always)]
fn twoing_score(
    labels: &HashSet<i32>,
    left: &HashMap<i32, f64>,
    right: &HashMap<i32, f64>,
) -> f64
{
    let pl = left.values().sum::<f64>();
    let pr = right.values().sum::<f64>();
    let pt = pl + pr;

    assert!(pt > 0f64);
    if pl == 0f64 || pr == 0f64 { return 0f64; }

    let mut score = 0f64;
    for y in labels {
        let l = left.get(y).unwrap_or(&0f64);
        let r = right.get(y).unwrap_or(&0f64);

        score += ((l / pl) - (r / pr)).abs();
    }
    score = score.powi(2) * pl * pr / (2f64 * pt).powi(2);

    score
}

#[cfg(test)]
mod tests {
    use std::io::BufReader;
    use super::*;
    use miniboosts_core::Feature;

    const TEST_TOLERANCE: f64 = 1e-9;

    fn test_feature() -> Feature {
        let mut feat = Feature::dense("feat");
        feat.append((0, 0.1));
        feat.append((1, 0.2));
        feat.append((2, 0.3));
        feat.append((3, 0.4));
        feat.append((4, 0.5));
        feat.append((5, 0.6));
        feat.append((6, 0.7));
        feat.append((7, 0.8));
        feat.append((8, 0.9));
        feat.append((9, 1.0));
        feat
    }

    fn test_sample() -> Sample {
        let csv = b"\
        feat,unused,class\n\
        0.1,0.0,1.0\n\
        0.2,0.0,1.0\n\
        0.3,0.0,1.0\n\
        0.4,0.3,1.0\n\
        0.5,0.3,1.0\n\
        0.6,0.3,-1.0\n\
        0.7,0.3,-1.0\n\
        0.8,0.3,-1.0\n\
        0.9,0.9,-1.0\n\
        1.0,0.9,-1.0";
        let reader = BufReader::new(&csv[..]);
        Sample::from_reader(reader, true)
            .unwrap()
            .set_target("class")
    }

    fn data_01() -> (Feature, Vec<f64>, Vec<usize>, Vec<f64>, Bins) {
        let m: usize = 10;
        let feat = test_feature();
        let y = {
            let mut y = vec![1f64; 5];
            y.extend(std::iter::repeat_n(-1f64, 5));
            y
        };
        let ix = (0..m).collect::<Vec<_>>();
        let dist = vec![1f64 / m as f64; m];
        let bins = Bins::cut(&feat, m);
        (feat, y, ix, dist, bins)
    }

    fn data_02() -> (Feature, Vec<f64>, Vec<usize>, Vec<f64>, Bins) {
        let m: usize = 10;
        let feat = test_feature();
        let y = {
            let mut y = vec![1f64; 5];
            y.extend(std::iter::repeat_n(-1f64, 5));
            y[4] = -1f64;
            y[5] =  1f64;
            y
        };
        let ix = (0..m).collect::<Vec<_>>();
        let dist = {
            let mut dist = vec![0f64; m];
            dist[4] = 0.5;
            dist[5] = 0.5;
            dist
        };
        let bins = Bins::cut(&feat, m);
        (feat, y, ix, dist, bins)
    }

    fn data_03() -> (Feature, Vec<f64>, Vec<usize>, Vec<f64>, Bins) {
        let m: usize = 10;
        let feat = test_feature();
        let y = {
            let mut y = vec![1f64; 5];
            y.extend(std::iter::repeat_n(-1f64, 5));
            y[4] = -1f64;
            y[5] =  1f64;
            y
        };
        let ix = (0..m).collect::<Vec<_>>();
        let dist = {
            let mut dist = vec![1f64 / m as f64; m];
            dist[4] = 1f64 / (2f64 * m as f64);
            dist[5] = 3f64 / (2f64 * m as f64);
            dist
        };
        let bins = Bins::cut(&feat, m);
        (feat, y, ix, dist, bins)
    }

    #[test]
    fn test_split_by_entropy_01() {
        let (x, y, ix, dist, bins) = data_01();

        let pack = bins.pack(&ix[..], &x, &y[..], &dist[..]);
        let (threshold, score) = split_by_entropy(pack);

        let expected_threshold = 0.55;
        assert!(
            (expected_threshold - threshold).abs() < TEST_TOLERANCE,
            "expected {expected_threshold}, got {threshold}.",
        );

        let expected_score = Score::from(0f64);
        assert!(
            (expected_score.0 - score.0).abs() < TEST_TOLERANCE,
            "expected {expected_score:?}, got {score:?}",
        );
    }

    #[test]
    fn test_split_by_entropy_02() {
        let (x, y, ix, dist, bins) = data_02();

        let pack = bins.pack(&ix[..], &x, &y[..], &dist[..]);
        let (threshold, score) = split_by_entropy(pack);

        let expected_threshold = 0.55;
        assert!(
            (expected_threshold - threshold).abs() < TEST_TOLERANCE,
            "expected {expected_threshold}, got {threshold}.",
        );

        let expected_score = Score::from(0f64);
        assert!(
            (expected_score.0 - score.0).abs() < TEST_TOLERANCE,
            "expected {expected_score:?}, got {score:?}",
        );
    }

    #[test]
    fn test_split_by_entropy_03() {
        let (x, y, ix, dist, bins) = data_03();

        let pack = bins.pack(&ix[..], &x, &y[..], &dist[..]);
        let (threshold, score) = split_by_entropy(pack);

        let expected_threshold = 0.64;
        assert!(
            (expected_threshold - threshold).abs() < TEST_TOLERANCE,
            "expected {expected_threshold}, got {threshold}.",
        );

        let expected_score = Score::from({
            let e = - (11f64 / 12f64) * (11f64 / 12f64).ln()
                - (1f64 / 12f64) * (1f64 / 12f64).ln();
            e * 6f64 / 10f64
        });
        assert!(
            (expected_score.0 - score.0).abs() < TEST_TOLERANCE,
            "expected {expected_score:?}, got {score:?}",
        );
    }

    #[test]
    fn test_split_by_edge_01() {
        let (x, y, ix, dist, bins) = data_01();

        let pack = bins.pack(&ix[..], &x, &y[..], &dist[..]);
        let (threshold, score) = split_by_edge(pack);

        let expected_threshold = 0.55;
        assert!(
            (expected_threshold - threshold).abs() < TEST_TOLERANCE,
            "expected {expected_threshold}, got {threshold}.",
        );

        let expected_score = Score::from(1f64);
        assert!(
            (expected_score.0 - score.0).abs() < TEST_TOLERANCE,
            "expected {expected_score:?}, got {score:?}",
        );
    }

    #[test]
    fn test_split_by_edge_02() {
        let (x, y, ix, dist, bins) = data_02();

        let pack = bins.pack(&ix[..], &x, &y[..], &dist[..]);
        let (threshold, score) = split_by_edge(pack);

        let expected_threshold = 0.55;
        assert!(
            (expected_threshold - threshold).abs() < TEST_TOLERANCE,
            "expected {expected_threshold}, got {threshold}.",
        );

        let expected_score = Score::from(1f64);
        assert!(
            (expected_score.0 - score.0).abs() < TEST_TOLERANCE,
            "expected {expected_score:?}, got {score:?}",
        );
    }

    #[test]
    fn test_split_by_edge_03() {
        let (x, y, ix, dist, bins) = data_03();

        let pack = bins.pack(&ix[..], &x, &y[..], &dist[..]);
        let (threshold, score) = split_by_edge(pack);

        let expected_threshold = 0.64;
        assert!(
            (expected_threshold - threshold).abs() < TEST_TOLERANCE,
            "expected {expected_threshold}, got {threshold}.",
        );

        let expected_score = Score::from(0.9);
        assert!(
            (expected_score.0 - score.0).abs() < TEST_TOLERANCE,
            "expected {expected_score:?}, got {score:?}",
        );
    }

    #[test]
    fn test_split_by_gini_01() {
        let (x, y, ix, dist, bins) = data_01();

        let pack = bins.pack(&ix[..], &x, &y[..], &dist[..]);
        let (threshold, score) = split_by_gini(pack);

        let expected_threshold = 0.55;
        assert!(
            (expected_threshold - threshold).abs() < TEST_TOLERANCE,
            "expected {expected_threshold}, got {threshold}.",
        );

        let expected_score = Score::from(0f64);
        assert!(
            (expected_score.0 - score.0).abs() < TEST_TOLERANCE,
            "expected {expected_score:?}, got {score:?}",
        );
    }

    #[test]
    fn test_split_by_gini_02() {
        let (x, y, ix, dist, bins) = data_02();

        let pack = bins.pack(&ix[..], &x, &y[..], &dist[..]);
        let (threshold, score) = split_by_gini(pack);

        let expected_threshold = 0.55;
        assert!(
            (expected_threshold - threshold).abs() < TEST_TOLERANCE,
            "expected {expected_threshold}, got {threshold}.",
        );

        let expected_score = Score::from(0f64);
        assert!(
            (expected_score.0 - score.0).abs() < TEST_TOLERANCE,
            "expected {expected_score:?}, got {score:?}",
        );
    }

    #[test]
    fn test_split_by_gini_03() {
        let (x, y, ix, dist, bins) = data_03();

        let pack = bins.pack(&ix[..], &x, &y[..], &dist[..]);
        let (threshold, score) = split_by_gini(pack);

        let expected_threshold = 0.64;
        assert!(
            (expected_threshold - threshold).abs() < TEST_TOLERANCE,
            "expected {expected_threshold}, got {threshold}.",
        );

        let expected_score = Score::from({
            let i = 1f64 - (11f64 / 12f64).powi(2) - (1f64/ 12f64).powi(2);
            i * 6f64 / 10f64
        });
        assert!(
            (expected_score.0 - score.0).abs() < TEST_TOLERANCE,
            "expected {expected_score:?}, got {score:?}",
        );
    }

    #[test]
    fn test_split_by_twoing_01() {
        let (x, y, ix, dist, bins) = data_01();

        let pack = bins.pack(&ix[..], &x, &y[..], &dist[..]);
        let (threshold, score) = split_by_twoing(pack);

        let expected_threshold = 0.55;
        assert!(
            (expected_threshold - threshold).abs() < TEST_TOLERANCE,
            "expected {expected_threshold}, got {threshold}.",
        );

        let expected_score = Score::from(1f64 / 4f64);
        assert!(
            (expected_score.0 - score.0).abs() < TEST_TOLERANCE,
            "expected {expected_score:?}, got {score:?}",
        );
    }

    #[test]
    fn test_split_by_twoing_02() {
        let (x, y, ix, dist, bins) = data_02();

        let pack = bins.pack(&ix[..], &x, &y[..], &dist[..]);
        let (threshold, score) = split_by_twoing(pack);

        let expected_threshold = 0.55;
        assert!(
            (expected_threshold - threshold).abs() < TEST_TOLERANCE,
            "expected {expected_threshold}, got {threshold}.",
        );

        let expected_score = Score::from(1f64 / 4f64);
        assert!(
            (expected_score.0 - score.0).abs() < TEST_TOLERANCE,
            "expected {expected_score:?}, got {score:?}",
        );
    }

    #[test]
    fn test_split_by_twoing_03() {
        let (x, y, ix, dist, bins) = data_03();

        let pack = bins.pack(&ix[..], &x, &y[..], &dist[..]);
        let (threshold, score) = split_by_twoing(pack);

        let expected_threshold = 0.64;
        assert!(
            (expected_threshold - threshold).abs() < TEST_TOLERANCE,
            "expected {expected_threshold}, got {threshold}.",
        );

        let expected_score = Score::from(
            (22f64 / 12f64).powi(2) * 3f64 / 50f64
        );
        assert!(
            (expected_score.0 - score.0).abs() < TEST_TOLERANCE,
            "expected {expected_score:?}, got {score:?}",
        );
    }

    #[test]
    fn test_best_split_entropy() {
        let sample = test_sample();
        let bins = sample.features()
            .iter()
            .map(|feature| {
                let name = feature.name();
                (name, Bins::cut(feature, 10))
            })
        .collect::<HashMap<_, _>>();
        let m = sample.shape().0;
        let ix = (0..m).collect::<Vec<_>>();
        let dist = vec![1f64 / m as f64; m];
        let s = SplitBy::Entropy;
        let (name, thr) = s.best_split(&bins, &sample, &dist[..], &ix[..]);

        let expected_name = "feat";
        assert_eq!(expected_name, name);

        let expected_thr = 0.55;
        assert!(
            (expected_thr - thr).abs() < TEST_TOLERANCE,
            "expected {expected_thr}, got {thr}.",
        );
    }

    #[test]
    fn test_best_split_gini() {
        let sample = test_sample();
        let bins = sample.features()
            .iter()
            .map(|feature| {
                let name = feature.name();
                (name, Bins::cut(feature, 10))
            })
        .collect::<HashMap<_, _>>();
        let m = sample.shape().0;
        let ix = (0..m).collect::<Vec<_>>();
        let dist = vec![1f64 / m as f64; m];
        let s = SplitBy::Gini;
        let (name, thr) = s.best_split(&bins, &sample, &dist[..], &ix[..]);

        let expected_name = "feat";
        assert_eq!(expected_name, name);

        let expected_thr = 0.55;
        assert!(
            (expected_thr - thr).abs() < TEST_TOLERANCE,
            "expected {expected_thr}, got {thr}.",
        );
    }

    #[test]
    fn test_best_split_edge() {
        let sample = test_sample();
        let bins = sample.features()
            .iter()
            .map(|feature| {
                let name = feature.name();
                (name, Bins::cut(feature, 10))
            })
        .collect::<HashMap<_, _>>();
        let m = sample.shape().0;
        let ix = (0..m).collect::<Vec<_>>();
        let dist = vec![1f64 / m as f64; m];
        let s = SplitBy::Edge;
        let (name, thr) = s.best_split(&bins, &sample, &dist[..], &ix[..]);

        let expected_name = "feat";
        assert_eq!(expected_name, name);

        let expected_thr = 0.55;
        assert!(
            (expected_thr - thr).abs() < TEST_TOLERANCE,
            "expected {expected_thr}, got {thr}.",
        );
    }

    #[test]
    fn test_best_split_twoing() {
        let sample = test_sample();
        let bins = sample.features()
            .iter()
            .map(|feature| {
                let name = feature.name();
                (name, Bins::cut(feature, 10))
            })
        .collect::<HashMap<_, _>>();
        let m = sample.shape().0;
        let ix = (0..m).collect::<Vec<_>>();
        let dist = vec![1f64 / m as f64; m];
        let s = SplitBy::Twoing;
        let (name, thr) = s.best_split(&bins, &sample, &dist[..], &ix[..]);

        let expected_name = "feat";
        assert_eq!(expected_name, name);

        let expected_thr = 0.55;
        assert!(
            (expected_thr - thr).abs() < TEST_TOLERANCE,
            "expected {expected_thr}, got {thr}.",
        );
    }
}

