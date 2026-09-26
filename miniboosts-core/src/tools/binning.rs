use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::ops::Range;

use crate::{
    Feature,
    constants::{PERTURBATION, PRINT_WIDTH_BINNING},
};

/// Binning: A feature processing.
#[derive(Debug, Clone)]
pub struct Bin(pub Range<f64>);

impl Bin {
    /// Create a new instance of `Bin`.
    #[inline(always)]
    pub fn new(range: Range<f64>) -> Self {
        Self(range)
    }

    /// Test membership. A bin ending at `f64::MAX` also includes that value.
    #[inline(always)]
    pub fn contains(&self, item: &f64) -> bool {
        self.0.contains(item) || (*item == f64::MAX && self.0.end == f64::MAX)
    }

    pub fn start(&self) -> f64 {
        self.0.start
    }

    pub fn set_start(&mut self, s: f64) {
        self.0.start = s;
    }

    pub fn end(&self) -> f64 {
        self.0.end
    }

    pub fn set_end(&mut self, e: f64) {
        self.0.end = e;
    }
}

/// A wrapper of `Vec<Bin>`.
#[derive(Debug)]
pub struct Bins(Vec<Bin>);

impl Bins {
    /// Returns the number of bins.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Cut a nonempty, finite feature into exactly `n_bin` equal-width bins.
    /// Sparse implicit zeros participate in the range. Constant values are
    /// perturbed where representable; rounding may leave empty bins.
    /// The outer bins cover all finite values, including `f64::MAX`.
    /// Panics for zero bins, empty features, or non-finite values.
    pub fn cut(feature: &Feature, n_bin: usize) -> Self {
        assert!(n_bin > 0, "bin count must be positive");
        let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut include = |value: f64| {
            assert!(value.is_finite(), "binning requires finite feature values");
            min = min.min(value);
            max = max.max(value);
        };
        match feature {
            Feature::Dense { vals, .. } => {
                assert!(!vals.is_empty(), "cannot bin an empty feature");
                for &value in vals {
                    include(value);
                }
            }
            Feature::Sparse { vals, size, .. } => {
                assert!(*size > 0, "cannot bin an empty feature");
                for &(_, value) in vals {
                    include(value);
                }
                if feature.has_zero() {
                    include(0.0);
                }
            }
        }
        if min == max {
            min -= PERTURBATION;
            max += PERTURBATION;
        }
        let mut bins = Vec::with_capacity(n_bin);
        let mut left = f64::MIN;
        for i in 1..=n_bin {
            let t = i as f64 / n_bin as f64;
            let right = if i == n_bin {
                f64::MAX
            } else if min.signum() == max.signum() {
                min + (max - min) * t
            } else {
                // Avoid overflow when the range spans both finite extremes.
                min * (1.0 - t) + max * t
            };
            bins.push(Bin::new(left..right));
            left = right;
        }
        Self(bins)
    }

    pub fn pack(
        &self,
        indices: &[usize],
        feature: &Feature,
        labels: &[f64],
        dist: &[f64],
    ) -> Vec<(Bin, HashMap<i32, f64>)> {
        let n_bins = self.0.len();
        let mut packed = vec![HashMap::<i32, f64>::new(); n_bins];

        for &i in indices {
            let xi = feature[i];
            let yi = labels[i] as i32;
            let di = dist[i];

            let pos = self
                .0
                .binary_search_by(|range| {
                    if range.contains(&xi) {
                        return Ordering::Equal;
                    }
                    range.0.start.partial_cmp(&xi).unwrap()
                })
                .unwrap();
            let weight = packed[pos].entry(yi).or_insert(0.0);
            *weight += di;
        }
        self.remove_zero_weight_pack_and_normalize(packed)
    }

    /// This method removes bins with zero weights.
    /// # Example
    /// Assume that we have bins and its weight.
    /// ```text
    /// Bins       | [-3.0, 2.5), [2.5, 7.0), [7.0, 8.1), [8.1, 9.0)
    /// Weights(+) |     0.5,        0.0,        0.0,        0.2
    /// Weights(-) |     0.0,        0.0,        0.0,        0.1
    /// ```
    /// This method remove zero-weight bins and normalize the weights
    /// like this:
    /// ```text
    /// Bins       | [-3.0, 5.3), [5.3, 9.0)
    /// Weights(+) |     0.625,      0.25
    /// Weights(-) |     0.0,        0.125
    /// ```
    /// That is, this method
    /// - Change the bin bounds,
    /// -
    fn remove_zero_weight_pack_and_normalize(
        &self,
        pack: Vec<HashMap<i32, f64>>,
    ) -> Vec<(Bin, HashMap<i32, f64>)> {
        let mut pack = self
            .0
            .iter()
            .cloned()
            .zip(pack)
            .filter(|(_, weightmap)| !weightmap.is_empty())
            .collect::<Vec<_>>();

        let n = pack.len();
        for i in 0..n - 1 {
            let t = {
                let left = &pack[i].0;
                let righ = &pack[i + 1].0;

                let e = left.end();
                let s = righ.start();

                s / 2.0 + e / 2.0
            };
            pack[i].0.set_end(t);
            pack[i + 1].0.set_start(t);
        }
        let leftmost = &mut pack[0].0;
        leftmost.set_start(f64::MIN);
        let rightmost = &mut pack[n - 1].0;
        rightmost.set_end(f64::MAX);
        pack
    }
}

impl fmt::Display for Bins {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bins = &self.0;
        let n_bins = bins.len();
        if n_bins > PRINT_WIDTH_BINNING {
            let head = bins[..2]
                .iter()
                .map(|bin| format!("{bin}"))
                .collect::<Vec<_>>()
                .join(", ");
            let tail = bins.last().map(|bin| format!("{bin}")).unwrap();
            write!(f, "{head}, ..., {tail}")
        } else {
            let line = bins
                .iter()
                .map(|bin| format!("{}", bin))
                .collect::<Vec<_>>()
                .join(", ");
            write!(f, "{line}")
        }
    }
}

impl fmt::Display for Bin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let start = if self.0.start == f64::MIN {
            String::from("-Inf")
        } else {
            let start = self.0.start;
            let sgn = if start > 0.0 {
                '+'
            } else if start < 0.0 {
                '-'
            } else {
                ' '
            };
            let start = start.abs();
            format!("{sgn}{start: >.2}")
        };
        let end = if self.0.end == f64::MAX {
            String::from("+Inf")
        } else {
            let end = self.0.end;
            let sgn = if end > 0.0 {
                '+'
            } else if end < 0.0 {
                '-'
            } else {
                ' '
            };
            let end = end.abs();
            format!("{sgn}{end: >.2}")
        };

        write!(f, "[{start}, {end})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_sparse_binning_parity_across_scales() {
        for values in [
            vec![1e-6, 2e-6],
            vec![1e-300, 2e-300],
            vec![0.0, 0.0],
            vec![1e300, 1e300],
            vec![f64::MIN, f64::MAX],
        ] {
            let mut dense = Feature::dense("x");
            let mut sparse = Feature::sparse("x", values.len());
            for (i, &v) in values.iter().enumerate() {
                dense.append((i, v));
                sparse.append((i, v));
            }
            for count in [1, 2, 5] {
                let d = Bins::cut(&dense, count);
                let s = Bins::cut(&sparse, count);
                assert_eq!(d.len(), count);
                for (a, b) in d.0.iter().zip(&s.0) {
                    assert_eq!(a.0, b.0);
                    assert!(a.start().is_finite() && a.end().is_finite());
                    assert!(a.start() <= a.end());
                }
                for value in &values {
                    assert!(s.0.iter().any(|bin| bin.contains(value)));
                }
                let packed = s.pack(&[0, 1], &sparse, &[1.0, -1.0], &[0.5, 0.5]);
                assert_eq!(
                    packed.iter().flat_map(|(_, m)| m.values()).sum::<f64>(),
                    1.0
                );
            }
        }
    }

    #[test]
    fn invalid_binning_inputs_are_rejected() {
        assert!(std::panic::catch_unwind(|| Bins::cut(&Feature::dense("x"), 1)).is_err());
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut dense = Feature::dense("x");
            let mut sparse = Feature::sparse("x", 1);
            dense.append((0, value));
            sparse.append((0, value));
            for f in [dense, sparse] {
                assert!(std::panic::catch_unwind(|| Bins::cut(&f, 1)).is_err());
            }
        }
        let mut f = Feature::dense("x");
        f.append((0, 1.0));
        assert!(std::panic::catch_unwind(|| Bins::cut(&f, 0)).is_err());
    }

    const NUMERIC_ERROR_TOLERANCE: f64 = 1e-9;

    #[test]
    fn test_bin_new() {
        let rng = 0f64..1f64;

        let expect = rng.clone();
        let result = Bin::new(rng).0;
        assert_eq!(expect, result, "expected {expect:?}, got {result:?}.");
    }

    #[test]
    fn test_bin_contains_01() {
        let rng = 0f64..1f64;
        let bin = Bin::new(rng);
        let itm = 0.5f64;

        let result = bin.contains(&itm);
        let expect = true;
        assert_eq!(expect, result, "expected {expect}, got {result}.");
    }

    #[test]
    fn test_bin_contains_02() {
        let rng = 0f64..1f64;
        let bin = Bin::new(rng);
        let itm = 0f64;

        let result = bin.contains(&itm);
        let expect = true;
        assert_eq!(expect, result, "expected {expect}, got {result}.");
    }

    #[test]
    fn test_bin_contains_03() {
        let rng = 0f64..1f64;
        let bin = Bin::new(rng);
        let itm = 1f64;

        let result = bin.contains(&itm);
        let expect = false;
        assert_eq!(expect, result, "expected {expect}, got {result}.");
    }

    #[test]
    fn test_bin_contains_04() {
        let rng = 0f64..1f64;
        let bin = Bin::new(rng);
        let itm = -100f64;

        let result = bin.contains(&itm);
        let expect = false;
        assert_eq!(expect, result, "expected {expect}, got {result}.");
    }

    #[test]
    fn test_bin_display_01() {
        let rng = 0f64..1f64;
        let bin = Bin::new(rng);

        let result = format!("{bin}");
        let expect = "[ 0.00, +1.00)".to_string();
        assert_eq!(expect, result, "expected {expect}, got {result}.");
    }

    #[test]
    fn test_bin_display_02() {
        let rng = -2f64..100f64;
        let bin = Bin::new(rng);

        let result = format!("{bin}");
        let expect = "[-2.00, +100.00)".to_string();
        assert_eq!(expect, result, "expected {expect}, got {result}.");
    }

    #[test]
    fn test_bin_display_03() {
        let rng = f64::MIN..f64::MAX;
        let bin = Bin::new(rng);

        let result = format!("{bin}");
        let expect = "[-Inf, +Inf)".to_string();
        assert_eq!(expect, result, "expected {expect}, got {result}.");
    }

    #[test]
    fn test_cut_01() {
        let mut feature = Feature::dense("dense");
        feature.append((0, 0f64));
        feature.append((0, 4f64));
        feature.append((0, 1f64));

        let result = Bins::cut(&feature, 2);
        let expect = vec![Bin::new(f64::MIN..2f64), Bin::new(2f64..f64::MAX)];

        assert_eq!(result.0.len(), expect.len());
        for (r, e) in result.0.into_iter().zip(expect) {
            assert_eq!(r.0, e.0);
        }
    }

    #[test]
    fn test_cut_02() {
        let mut feature = Feature::dense("dense");
        feature.append((0, -10f64));
        feature.append((0, 4f64));
        feature.append((0, 10f64));

        let result = Bins::cut(&feature, 4);
        let expect = vec![
            Bin::new(f64::MIN..-5f64),
            Bin::new(-5f64..0f64),
            Bin::new(0f64..5f64),
            Bin::new(5f64..f64::MAX),
        ];

        assert_eq!(result.0.len(), expect.len());
        for (r, e) in result.0.into_iter().zip(expect) {
            assert_eq!(r.0, e.0);
        }
    }

    #[test]
    fn test_cut_03() {
        let mut feature = Feature::sparse("sparse", 1_000);
        feature.append((0, 0f64));
        feature.append((100, 4f64));
        feature.append((12, 1f64));

        let result = Bins::cut(&feature, 2);
        let expect = vec![Bin::new(f64::MIN..2f64), Bin::new(2f64..f64::MAX)];

        assert_eq!(result.0.len(), expect.len());
        for (r, e) in result.0.into_iter().zip(expect) {
            assert_eq!(r.0, e.0);
        }
    }

    #[test]
    fn test_cut_04() {
        let mut feature = Feature::sparse("sparse", 1_000);
        feature.append((0, 0f64));
        feature.append((100, -10f64));
        feature.append((7, 10f64));
        feature.append((12, 1f64));

        let result = Bins::cut(&feature, 4);
        let expect = vec![
            Bin::new(f64::MIN..-5f64),
            Bin::new(-5f64..0f64),
            Bin::new(0f64..5f64),
            Bin::new(5f64..f64::MAX),
        ];

        assert_eq!(result.0.len(), expect.len());
        for (r, e) in result.0.into_iter().zip(expect) {
            assert_eq!(r.0, e.0);
        }
    }

    #[test]
    fn test_cut_05() {
        let mut feature = Feature::dense("dense");
        feature.append((0, 0f64));

        let result = Bins::cut(&feature, 2);
        let expect = vec![Bin::new(f64::MIN..0f64), Bin::new(0f64..f64::MAX)];

        assert_eq!(result.0.len(), expect.len());
        for (r, e) in result.0.into_iter().zip(expect) {
            assert_eq!(r.0, e.0);
        }
    }

    #[test]
    fn test_cut_06() {
        let mut feature = Feature::sparse("dense", 1_000);
        feature.append((12, -10f64));
        feature.append((12, 8f64));

        let result = Bins::cut(&feature, 2);
        let expect = vec![Bin::new(f64::MIN..-1f64), Bin::new(-1f64..f64::MAX)];

        assert_eq!(result.0.len(), expect.len());
        for (r, e) in result.0.into_iter().zip(expect) {
            assert_eq!(r.0, e.0);
        }
    }

    #[test]
    fn test_cut_07() {
        let mut feature = Feature::sparse("dense", 1_000);
        feature.append((12, -10f64));

        let result = Bins::cut(&feature, 2);
        let expect = vec![Bin::new(f64::MIN..-5f64), Bin::new(-5f64..f64::MAX)];

        assert_eq!(result.0.len(), expect.len());
        for (r, e) in result.0.into_iter().zip(expect) {
            assert_eq!(r.0, e.0);
        }
    }

    #[test]
    fn test_cut_08() {
        let mut feature = Feature::sparse("dense", 1_000);
        feature.append((12, 10f64));

        let result = Bins::cut(&feature, 2);
        let expect = vec![Bin::new(f64::MIN..5f64), Bin::new(5f64..f64::MAX)];

        assert_eq!(result.0.len(), expect.len());
        for (r, e) in result.0.into_iter().zip(expect) {
            assert_eq!(r.0, e.0);
        }
    }

    #[test]
    fn test_pack_01() {
        let feature = {
            let mut feature = Feature::dense("dense");
            feature.append((0, 0f64));
            feature.append((0, 1f64));
            feature.append((0, 1f64));
            feature.append((0, 10f64));
            feature.append((0, 2f64));
            feature.append((0, 9f64));
            feature.append((0, 5f64));
            feature.append((0, 6f64));
            feature.append((0, 3f64));
            feature
        };
        let bins = Bins::cut(&feature, 4);
        let expect = vec![
            Bin::new(f64::MIN..2.5f64),
            Bin::new(2.5f64..5.0f64),
            Bin::new(5.0f64..7.5f64),
            Bin::new(7.5f64..f64::MAX),
        ];

        assert_eq!(bins.0.len(), expect.len());
        for (b, e) in bins.0.iter().zip(&expect[..]) {
            assert_eq!(b.0, e.0);
        }

        let indices = (0..9).collect::<Vec<usize>>();
        let labels = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0];
        let dist = [0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1];

        let result = bins.pack(&indices[..], &feature, &labels[..], &dist[..]);
        let expect = {
            let maps = vec![
                HashMap::from([(1, 0.2), (-1, 0.3)]),
                HashMap::from([(1, 0.1)]),
                HashMap::from([(-1, 0.2)]),
                HashMap::from([(1, 0.2)]),
            ];
            expect.into_iter().zip(maps).collect::<Vec<_>>()
        };
        for ((rpack, rmap), (epack, emap)) in result.iter().zip(expect) {
            assert_eq!(rpack.0, epack.0, "{result:?}");
            assert_eq!(rmap.len(), emap.len());
            for (ek, ev) in emap {
                if let Some(rv) = rmap.get(&ek) {
                    assert!(
                        (*rv - ev).abs() < NUMERIC_ERROR_TOLERANCE,
                        "expected ({ek}, {ev}), got ({ek}, {rv}).",
                    );
                } else {
                    panic!(
                        "failed to get a value for {ek}. \
                        expected {ev}, got None."
                    );
                }
            }
        }
    }

    #[test]
    fn test_pack_02() {
        let feature = {
            let mut feature = Feature::dense("dense");
            feature.append((0, 0f64));
            feature.append((0, 1f64));
            feature.append((0, 1f64));
            feature.append((0, 10f64));
            feature.append((0, 2f64));
            feature.append((0, 9f64));
            feature.append((0, 5f64));
            feature.append((0, 6f64));
            feature
        };
        let bins = Bins::cut(&feature, 4);
        let expect = vec![
            Bin::new(f64::MIN..2.5f64),
            Bin::new(2.5f64..5.0f64),
            Bin::new(5.0f64..7.5f64),
            Bin::new(7.5f64..f64::MAX),
        ];

        assert_eq!(bins.0.len(), expect.len());
        for (b, e) in bins.0.iter().zip(&expect[..]) {
            assert_eq!(b.0, e.0);
        }

        let indices = (0..8).collect::<Vec<usize>>();
        let labels = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0];
        let dist = [0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1];

        let result = bins.pack(&indices[..], &feature, &labels[..], &dist[..]);
        let expect = {
            let bins = vec![
                Bin::new(f64::MIN..3.75f64),
                Bin::new(3.75f64..7.5f64),
                Bin::new(7.5f64..f64::MAX),
            ];
            let maps = vec![
                HashMap::from([(1, 0.2), (-1, 0.4)]),
                HashMap::from([(-1, 0.2)]),
                HashMap::from([(1, 0.2)]),
            ];
            bins.into_iter().zip(maps).collect::<Vec<_>>()
        };
        for ((rpack, rmap), (epack, emap)) in result.iter().zip(expect) {
            assert_eq!(rpack.0, epack.0, "{result:?}");
            assert_eq!(rmap.len(), emap.len());
            for (ek, ev) in emap {
                if let Some(rv) = rmap.get(&ek) {
                    assert!(
                        (*rv - ev).abs() < NUMERIC_ERROR_TOLERANCE,
                        "expected ({ek}, {ev}), got ({ek}, {rv}).",
                    );
                } else {
                    panic!(
                        "failed to get a value for {ek}. \
                        expected {ev}, got None."
                    );
                }
            }
        }
    }

    #[test]
    fn test_pack_03() {
        let feature = {
            let mut feature = Feature::sparse("sparse", 9);
            feature.append((1, 1f64));
            feature.append((2, 1f64));
            feature.append((3, 10f64));
            feature.append((4, 2f64));
            feature.append((5, 9f64));
            feature.append((6, 5f64));
            feature.append((7, 6f64));
            feature.append((8, 3f64));
            feature
        };
        let bins = Bins::cut(&feature, 4);
        let expect = vec![
            Bin::new(f64::MIN..2.5f64),
            Bin::new(2.5f64..5.0f64),
            Bin::new(5.0f64..7.5f64),
            Bin::new(7.5f64..f64::MAX),
        ];

        assert_eq!(bins.0.len(), expect.len());
        for (b, e) in bins.0.iter().zip(&expect[..]) {
            assert_eq!(b.0, e.0);
        }

        let indices = (0..9).collect::<Vec<usize>>();
        let labels = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0];
        let dist = [0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1];

        let result = bins.pack(&indices[..], &feature, &labels[..], &dist[..]);
        let expect = {
            let maps = vec![
                HashMap::from([(1, 0.2), (-1, 0.3)]),
                HashMap::from([(1, 0.1)]),
                HashMap::from([(-1, 0.2)]),
                HashMap::from([(1, 0.2)]),
            ];
            expect.into_iter().zip(maps).collect::<Vec<_>>()
        };
        for ((rpack, rmap), (epack, emap)) in result.iter().zip(expect) {
            assert_eq!(rpack.0, epack.0, "{result:?}");
            assert_eq!(rmap.len(), emap.len());
            for (ek, ev) in emap {
                if let Some(rv) = rmap.get(&ek) {
                    assert!(
                        (*rv - ev).abs() < NUMERIC_ERROR_TOLERANCE,
                        "expected ({ek}, {ev}), got ({ek}, {rv}).",
                    );
                } else {
                    panic!(
                        "failed to get a value for {ek}. \
                        expected {ev}, got None."
                    );
                }
            }
        }
    }

    #[test]
    fn test_pack_04() {
        let feature = {
            let mut feature = Feature::sparse("sparse", 8);
            feature.append((1, 1f64));
            feature.append((2, 1f64));
            feature.append((3, 10f64));
            feature.append((4, 2f64));
            feature.append((5, 9f64));
            feature.append((6, 5f64));
            feature.append((7, 6f64));
            feature
        };
        let bins = Bins::cut(&feature, 4);
        let expect = vec![
            Bin::new(f64::MIN..2.5f64),
            Bin::new(2.5f64..5.0f64),
            Bin::new(5.0f64..7.5f64),
            Bin::new(7.5f64..f64::MAX),
        ];

        assert_eq!(bins.0.len(), expect.len());
        for (b, e) in bins.0.iter().zip(&expect[..]) {
            assert_eq!(b.0, e.0);
        }

        let indices = (0..8).collect::<Vec<usize>>();
        let labels = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0];
        let dist = [0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1];

        let result = bins.pack(&indices[..], &feature, &labels[..], &dist[..]);
        let expect = {
            let bins = vec![
                Bin::new(f64::MIN..3.75f64),
                Bin::new(3.75f64..7.5f64),
                Bin::new(7.5f64..f64::MAX),
            ];
            let maps = vec![
                HashMap::from([(1, 0.2), (-1, 0.4)]),
                HashMap::from([(-1, 0.2)]),
                HashMap::from([(1, 0.2)]),
            ];
            bins.into_iter().zip(maps).collect::<Vec<_>>()
        };
        for ((rpack, rmap), (epack, emap)) in result.iter().zip(expect) {
            assert_eq!(rpack.0, epack.0, "{result:?}");
            assert_eq!(rmap.len(), emap.len());
            for (ek, ev) in emap {
                if let Some(rv) = rmap.get(&ek) {
                    assert!(
                        (*rv - ev).abs() < NUMERIC_ERROR_TOLERANCE,
                        "expected ({ek}, {ev}), got ({ek}, {rv}).",
                    );
                } else {
                    panic!(
                        "failed to get a value for {ek}. \
                        expected {ev}, got None."
                    );
                }
            }
        }
    }
}
