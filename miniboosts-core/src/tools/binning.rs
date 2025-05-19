use std::fmt;
use std::ops::Range;
use std::cmp::Ordering;
use std::collections::HashMap;

use crate::{
    constants::{
        NUMERIC_TOLERANCE,
        PERTURBATION,
        PRINT_WIDTH_BINNING,
    },
    Feature,
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

    /// Check whether the given `item` is conteined by `self.`
    #[inline(always)]
    pub fn contains(&self, item: &f64) -> bool {
        self.0.contains(item)
    }

    pub fn start(&self) -> f64 { self.0.start }

    pub fn set_start(&mut self, s: f64) {
        self.0.start = s;
    }

    pub fn end(&self) -> f64 { self.0.end }

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

    /// Cut the given `Feature` into `n_bins` bins.
    /// This method naively cut the given slice with same width.
    #[inline(always)]
    pub fn cut(feature: &Feature, n_bin: usize) -> Self
    {
        let mut bins = {
            let has_zero = feature.has_zero();
            match feature {
                Feature::Dense { vals, .. } => {
                    Self::cut_dense(&vals[..], n_bin)
                },
                Feature::Sparse { vals, .. } => {
                    Self::cut_sparse(&vals[..], n_bin, has_zero)
                },
            }
        };

        // The `start` of the left-most bin should be `f64::MIN`.
        bins.0.first_mut().unwrap().0.start = f64::MIN;
        // The `end` of the right-most bin should be `f64::MAX`.
        bins.0.last_mut().unwrap().0.end = f64::MAX;

        bins
    }

    fn cut_dense(vals: &[f64], n_bin: usize) -> Self
    {
        let mut min = f64::MAX;
        let mut max = f64::MIN;
        vals.iter()
            .copied()
            .for_each(|val| {
                min = min.min(val);
                max = max.max(val);
            });

        // If the minimum value equals to the maximum one,
        // slightly perturb them.
        if min == max {
            min -= PERTURBATION;
            max += PERTURBATION;
        }

        let width = (max - min) / n_bin as f64;

        let mut bins = Vec::with_capacity(n_bin);

        let mut left = min;
        for i in 0..n_bin {
            let l = if i == 0 { f64::MIN } else { left };
            let r = if i == n_bin - 1 { f64::MAX } else { left + width };
            bins.push(Bin::new(l..r));

            left = r;
        }

        assert_eq!(bins.len(), n_bin);

        Self(bins)
    }

    fn cut_sparse(
        vals:     &[(usize, f64)],
        n_bin:    usize,
        has_zero: bool,
    ) -> Self
    {
        let mut min = f64::MAX;
        let mut max = f64::MIN;
        vals.iter()
            .copied()
            .for_each(|(_, val)| {
                min = min.min(val);
                max = max.max(val);
            });

        if min > 0.0 && has_zero {
            min = 0.0;
        }

        if max < 0.0 && has_zero {
            max = 0.0;
        }

        // If the minimum value equals to the maximum one,
        // slightly perturb them.
        if min == max {
            min -= PERTURBATION;
            max += PERTURBATION;
        }

        let intercept = (max - min) / n_bin as f64;

        let mut bins = Vec::with_capacity(n_bin);

        let mut left = min;
        while left < max {
            let right = left + intercept;
            bins.push(Bin::new(left..right));

            // Numerical error leads an unexpected split.
            // So, we ignore the bin with width smaller than 1e-9.
            if (right - max).abs() < NUMERIC_TOLERANCE { break; }

            left = right;
        }

        assert_eq!(bins.len(), n_bin);

        Self(bins)
    }

    pub fn pack(
        &self,
        indices: &[usize],
        feature: &Feature,
        labels: &[f64],
        dist: &[f64]
    ) -> Vec<(Bin, HashMap::<i32,f64>)>
    {
        let n_bins = self.0.len();
        let mut packed = vec![HashMap::<i32,f64>::new(); n_bins];

        for &i in indices {
            let xi = feature[i];
            let yi = labels[i] as i32;
            let di = dist[i];

            let pos = self.0.binary_search_by(|range| {
                    if range.contains(&xi) { return Ordering::Equal; }
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
        pack: Vec<HashMap::<i32,f64>>,
    ) -> Vec<(Bin, HashMap::<i32,f64>)>
    {
        let mut pack = self.0.iter()
            .cloned()
            .zip(pack)
            .filter(|(_, weightmap)| !weightmap.is_empty())
            .collect::<Vec<_>>();

        let n = pack.len();
        for i in 0..n-1 {
            let t = {
                let left = &pack[i].0;
                let righ = &pack[i+1].0;

                let e = left.end();
                let s = righ.start();

                (s + e) / 2f64
            };
            pack[i].0.set_end(t);
            pack[i+1].0.set_start(t);
        }
        let leftmost = &mut pack[0].0;
        leftmost.set_start(f64::MIN);
        let rightmost = &mut pack[n-1].0;
        rightmost.set_end(f64::MAX);
        pack
    }
}

impl fmt::Display for Bins {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bins = &self.0;
        let n_bins = bins.len();
        if n_bins > PRINT_WIDTH_BINNING {
            let head = bins[..2].iter()
                .map(|bin| format!("{bin}"))
                .collect::<Vec<_>>()
                .join(", ");
            let tail = bins.last()
                .map(|bin| format!("{bin}"))
                .unwrap();
            write!(f, "{head}, ..., {tail}")
        } else {
            let line = bins.iter()
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
        let expect = vec![
            Bin::new(f64::MIN..2f64),
            Bin::new(2f64..f64::MAX),
        ];

        assert_eq!(result.0.len(), expect.len());
        for (r, e) in result.0.into_iter().zip(expect) {
            assert_eq!(r.0, e.0);
        }
    }

    #[test]
    fn test_cut_02() {
        let mut feature = Feature::dense("dense");
        feature.append((0, -10f64));
        feature.append((0,   4f64));
        feature.append((0,  10f64));

        let result = Bins::cut(&feature, 4);
        let expect = vec![
            Bin::new(f64::MIN..-5f64   ),
            Bin::new(   -5f64..0f64    ),
            Bin::new(    0f64..5f64    ),
            Bin::new(    5f64..f64::MAX),
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
        let expect = vec![
            Bin::new(f64::MIN..2f64),
            Bin::new(2f64..f64::MAX),
        ];

        assert_eq!(result.0.len(), expect.len());
        for (r, e) in result.0.into_iter().zip(expect) {
            assert_eq!(r.0, e.0);
        }
    }

    #[test]
    fn test_cut_04() {
        let mut feature = Feature::sparse("sparse", 1_000);
        feature.append((0,     0f64));
        feature.append((100, -10f64));
        feature.append((7,    10f64));
        feature.append((12,    1f64));

        let result = Bins::cut(&feature, 4);
        let expect = vec![
            Bin::new(f64::MIN..-5f64   ),
            Bin::new(   -5f64..0f64    ),
            Bin::new(    0f64..5f64    ),
            Bin::new(    5f64..f64::MAX),
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
        let expect = vec![
            Bin::new(f64::MIN..0f64),
            Bin::new(0f64..f64::MAX),
        ];

        assert_eq!(result.0.len(), expect.len());
        for (r, e) in result.0.into_iter().zip(expect) {
            assert_eq!(r.0, e.0);
        }
    }

    #[test]
    fn test_cut_06() {
        let mut feature = Feature::sparse("dense", 1_000);
        feature.append((12, -10f64));
        feature.append((12,   8f64));

        let result = Bins::cut(&feature, 2);
        let expect = vec![
            Bin::new(f64::MIN..-1f64   ),
            Bin::new(   -1f64..f64::MAX),
        ];

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
        let expect = vec![
            Bin::new(f64::MIN..-5f64   ),
            Bin::new(   -5f64..f64::MAX),
        ];

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
        let expect = vec![
            Bin::new(f64::MIN..5f64    ),
            Bin::new(    5f64..f64::MAX),
        ];

        assert_eq!(result.0.len(), expect.len());
        for (r, e) in result.0.into_iter().zip(expect) {
            assert_eq!(r.0, e.0);
        }
    }

    #[test]
    fn test_pack_01() {
        let feature = {
            let mut feature = Feature::dense("dense");
            feature.append((0,  0f64));
            feature.append((0,  1f64));
            feature.append((0,  1f64));
            feature.append((0, 10f64));
            feature.append((0,  2f64));
            feature.append((0,  9f64));
            feature.append((0,  5f64));
            feature.append((0,  6f64));
            feature.append((0,  3f64));
            feature
        };
        let bins = Bins::cut(&feature, 4);
        let expect = vec![
            Bin::new(f64::MIN..2.5f64  ),
            Bin::new(  2.5f64..5.0f64  ),
            Bin::new(  5.0f64..7.5f64  ),
            Bin::new(  7.5f64..f64::MAX),
        ];

        assert_eq!(bins.0.len(), expect.len());
        for (b, e) in bins.0.iter().zip(&expect[..]) {
            assert_eq!(b.0, e.0);
        }

        let indices = (0..9).collect::<Vec<usize>>();
        let labels  = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0];
        let dist    = [0.1,  0.1, 0.1, 0.1,  0.2, 0.1,  0.1,  0.1, 0.1];

        let result = bins.pack(&indices[..], &feature, &labels[..], &dist[..]);
        let expect = {
            let maps = vec![
                HashMap::from([( 1, 0.2), (-1, 0.3)]),
                HashMap::from([( 1, 0.1)]),
                HashMap::from([(-1, 0.2)]),
                HashMap::from([( 1, 0.2)]),
            ];
            expect.into_iter()
                .zip(maps)
                .collect::<Vec<_>>()
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
            feature.append((0,  0f64));
            feature.append((0,  1f64));
            feature.append((0,  1f64));
            feature.append((0, 10f64));
            feature.append((0,  2f64));
            feature.append((0,  9f64));
            feature.append((0,  5f64));
            feature.append((0,  6f64));
            feature
        };
        let bins = Bins::cut(&feature, 4);
        let expect = vec![
            Bin::new(f64::MIN..2.5f64  ),
            Bin::new(  2.5f64..5.0f64  ),
            Bin::new(  5.0f64..7.5f64  ),
            Bin::new(  7.5f64..f64::MAX),
        ];

        assert_eq!(bins.0.len(), expect.len());
        for (b, e) in bins.0.iter().zip(&expect[..]) {
            assert_eq!(b.0, e.0);
        }

        let indices = (0..8).collect::<Vec<usize>>();
        let labels  = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0];
        let dist    = [0.1,  0.2, 0.1, 0.1,  0.2, 0.1,  0.1,  0.1];

        let result = bins.pack(&indices[..], &feature, &labels[..], &dist[..]);
        let expect = {
            let bins = vec![
            Bin::new(f64::MIN..3.75f64 ),
            Bin::new( 3.75f64..7.5f64  ),
            Bin::new(  7.5f64..f64::MAX),
            ];
            let maps = vec![
                HashMap::from([(1, 0.2), (-1, 0.4)]),
                HashMap::from([(-1, 0.2)]),
                HashMap::from([(1, 0.2)]),
            ];
            bins.into_iter()
                .zip(maps)
                .collect::<Vec<_>>()
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
            feature.append((1,  1f64));
            feature.append((2,  1f64));
            feature.append((3, 10f64));
            feature.append((4,  2f64));
            feature.append((5,  9f64));
            feature.append((6,  5f64));
            feature.append((7,  6f64));
            feature.append((8,  3f64));
            feature
        };
        let bins = Bins::cut(&feature, 4);
        let expect = vec![
            Bin::new(f64::MIN..2.5f64  ),
            Bin::new(  2.5f64..5.0f64  ),
            Bin::new(  5.0f64..7.5f64  ),
            Bin::new(  7.5f64..f64::MAX),
        ];

        assert_eq!(bins.0.len(), expect.len());
        for (b, e) in bins.0.iter().zip(&expect[..]) {
            assert_eq!(b.0, e.0);
        }

        let indices = (0..9).collect::<Vec<usize>>();
        let labels  = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0];
        let dist    = [0.1,  0.1, 0.1, 0.1,  0.2, 0.1,  0.1,  0.1, 0.1];

        let result = bins.pack(&indices[..], &feature, &labels[..], &dist[..]);
        let expect = {
            let maps = vec![
                HashMap::from([( 1, 0.2), (-1, 0.3)]),
                HashMap::from([( 1, 0.1)]),
                HashMap::from([(-1, 0.2)]),
                HashMap::from([( 1, 0.2)]),
            ];
            expect.into_iter()
                .zip(maps)
                .collect::<Vec<_>>()
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
            feature.append((1,  1f64));
            feature.append((2,  1f64));
            feature.append((3, 10f64));
            feature.append((4,  2f64));
            feature.append((5,  9f64));
            feature.append((6,  5f64));
            feature.append((7,  6f64));
            feature
        };
        let bins = Bins::cut(&feature, 4);
        let expect = vec![
            Bin::new(f64::MIN..2.5f64  ),
            Bin::new(  2.5f64..5.0f64  ),
            Bin::new(  5.0f64..7.5f64  ),
            Bin::new(  7.5f64..f64::MAX),
        ];

        assert_eq!(bins.0.len(), expect.len());
        for (b, e) in bins.0.iter().zip(&expect[..]) {
            assert_eq!(b.0, e.0);
        }

        let indices = (0..8).collect::<Vec<usize>>();
        let labels  = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0];
        let dist    = [0.1,  0.2, 0.1, 0.1,  0.2, 0.1,  0.1,  0.1];

        let result = bins.pack(&indices[..], &feature, &labels[..], &dist[..]);
        let expect = {
            let bins = vec![
            Bin::new(f64::MIN..3.75f64 ),
            Bin::new( 3.75f64..7.5f64  ),
            Bin::new(  7.5f64..f64::MAX),
            ];
            let maps = vec![
                HashMap::from([(1, 0.2), (-1, 0.4)]),
                HashMap::from([(-1, 0.2)]),
                HashMap::from([(1, 0.2)]),
            ];
            bins.into_iter()
                .zip(maps)
                .collect::<Vec<_>>()
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

