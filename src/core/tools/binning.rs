//! Equal-width binning of feature values.
//!
//! The decision tree ([`crate::DecisionTree`]) does not consider every
//! distinct feature value as a split threshold. Instead, each feature is cut
//! once into equal-width bins ([`Bins::cut`]), and only bin boundaries are
//! candidate thresholds. At each node, [`Bins::pack`] accumulates the
//! booster's example weights per bin and label. The regression tree
//! ([`crate::RegressionTree`]) instead gives each distinct value its own bin
//! when there are at most as many values as its bin cap.
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::ops::Range;

use crate::core::{
    Feature,
    constants::{PERTURBATION, PRINT_WIDTH_BINNING},
    sample::feature::gallop,
};

/// A half-open interval `[start, end)` of feature values.
///
/// By convention `f64::MIN` and `f64::MAX` stand for `-∞` and `+∞`
/// (see the `Display` implementation); a bin ending at `f64::MAX` also
/// contains `f64::MAX`.
///
/// # Examples
/// ```
/// use miniboosts::core::binning::Bin;
///
/// let mut bin = Bin::new(0.0..1.0);
/// assert!(bin.contains(&0.0));
/// assert!(!bin.contains(&1.0)); // half-open
/// bin.set_end(f64::MAX);
/// assert!(bin.contains(&f64::MAX)); // the last bin is closed
/// bin.set_start(f64::MIN);
/// assert_eq!((bin.start(), bin.end()), (f64::MIN, f64::MAX));
/// assert_eq!(bin.to_string(), "[-Inf, +Inf)");
/// ```
#[derive(Debug, Clone)]
pub struct Bin(pub Range<f64>);

impl Bin {
    /// Creates the bin `[range.start, range.end)`.
    pub fn new(range: Range<f64>) -> Self {
        Self(range)
    }

    /// Returns `true` if `start <= item < end`.
    /// A bin ending at `f64::MAX` also includes that value.
    pub fn contains(&self, item: &f64) -> bool {
        self.0.contains(item) || (*item == f64::MAX && self.0.end == f64::MAX)
    }

    /// Returns the inclusive lower bound.
    pub fn start(&self) -> f64 {
        self.0.start
    }

    /// Sets the inclusive lower bound to `s`.
    pub fn set_start(&mut self, s: f64) {
        self.0.start = s;
    }

    /// Returns the exclusive upper bound.
    pub fn end(&self) -> f64 {
        self.0.end
    }

    /// Sets the exclusive upper bound to `e`.
    pub fn set_end(&mut self, e: f64) {
        self.0.end = e;
    }
}

/// An ordered list of contiguous [`Bin`]s covering all finite values,
/// created by [`Bins::cut`].
#[derive(Debug)]
pub struct Bins(Vec<Bin>);

impl Bins {
    /// Returns the number of bins.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if there are no bins.
    /// [`Bins::cut`] always produces at least one bin.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Cuts a nonempty, finite feature into exactly `n_bin` equal-width bins.
    ///
    /// With `lo` and `hi` the minimum and maximum feature values, the
    /// interior boundaries are `lo + (hi - lo) * k / n_bin` for
    /// `k = 1, ..., n_bin - 1`; the first bin starts at `f64::MIN` and the
    /// last ends at `f64::MAX`, so the outer bins cover all finite values,
    /// including `f64::MAX`. Sparse implicit zeros participate in the range.
    /// Constant values are perturbed by
    /// [`PERTURBATION`] where
    /// representable; rounding may leave empty bins.
    ///
    /// # Panics
    /// Panics for zero bins, empty features, or non-finite values.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::core::{Feature, binning::Bins};
    ///
    /// let mut x = Feature::dense("x");
    /// for v in [0.0, 1.0, 3.0, 4.0] {
    ///     x.append((0, v));
    /// }
    /// // Range [0, 4] in 4 bins: interior boundaries 1, 2, 3 (exact).
    /// let bins = Bins::cut(&x, 4);
    /// assert_eq!(bins.len(), 4);
    /// assert_eq!(bins.to_string(), "[-Inf, +1.00), [+1.00, +2.00), ..., [+3.00, +Inf)");
    /// ```
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

    /// Returns one bin per distinct value of a nonempty, finite feature
    /// (sparse implicit zeros count as a value), so that every threshold
    /// between consecutive distinct values is a bin boundary. The boundary
    /// between `a < b` is their midpoint, or `b` if the midpoint rounds to
    /// `a` (adjacent floats); the outer bins extend to `f64::MIN` and
    /// `f64::MAX`.
    ///
    /// Time complexity: `O(m log m)` for `m` values.
    ///
    /// # Panics
    /// Panics for empty features or non-finite values.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::Sample;
    /// use miniboosts::core::binning::Bins;
    ///
    /// let rows = vec![vec![3.0], vec![1.0], vec![3.0], vec![2.0]];
    /// let sample = Sample::from_dense(&rows, None).unwrap();
    /// // Distinct values 1, 2, 3: boundaries 1.5 and 2.5.
    /// let bins = Bins::per_value(&sample["feature_0"]);
    /// assert_eq!(bins.len(), 3);
    /// ```
    pub fn per_value(feature: &Feature) -> Self {
        let mut values: Vec<f64> = match feature {
            Feature::Dense { vals, .. } => vals.clone(),
            Feature::Sparse { vals, .. } => {
                let mut values: Vec<f64> = vals.iter().map(|&(_, v)| v).collect();
                if feature.has_zero() {
                    values.push(0.0);
                }
                values
            }
        };
        assert!(!values.is_empty(), "cannot bin an empty feature");
        assert!(
            values.iter().all(|v| v.is_finite()),
            "binning requires finite feature values"
        );
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values.dedup();
        let mut bins = Vec::with_capacity(values.len());
        let mut left = f64::MIN;
        for pair in values.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            // Halves first: `a + b` may overflow.
            let mid = a / 2.0 + b / 2.0;
            let right = if a < mid && mid <= b { mid } else { b };
            bins.push(Bin::new(left..right));
            left = right;
        }
        bins.push(Bin::new(left..f64::MAX));
        Self(bins)
    }

    /// Returns the index of the bin containing `x`. The bins cover the
    /// whole real line, so every finite `x` has one.
    pub(crate) fn index_of(&self, x: f64) -> usize {
        self.0
            .binary_search_by(|bin| {
                if bin.contains(&x) {
                    Ordering::Equal
                } else {
                    bin.0.start.partial_cmp(&x).unwrap()
                }
            })
            .expect("every finite value falls into a bin")
    }

    /// Returns the [`BinIndex`] of `feature`, which these bins must cover.
    pub(crate) fn index(&self, feature: &Feature) -> BinIndex {
        match feature {
            Feature::Dense { vals, .. } => {
                BinIndex::Dense(vals.iter().map(|&x| self.index_of(x) as u32).collect())
            }
            Feature::Sparse { vals, .. } => BinIndex::Sparse {
                bin_of_entry: vals.iter().map(|&(_, x)| self.index_of(x) as u32).collect(),
                // Unused when no row is an implicit zero.
                zero: if feature.has_zero() {
                    self.index_of(0.0) as u32
                } else {
                    0
                },
            },
        }
    }

    /// Returns the bins in increasing order.
    pub(crate) fn as_slice(&self) -> &[Bin] {
        &self.0
    }

    /// Accumulates, for each bin, the total weight `dist[i]` of the examples
    /// `i ∈ indices` per label, where example `i` falls in the bin containing
    /// `feature[i]`. Labels are converted with `labels[i] as i32`.
    ///
    /// Bins containing none of `indices` are dropped, and the bounds of the
    /// remaining bins are adjusted so that they stay contiguous: each
    /// boundary next to a dropped run becomes the midpoint of that run, the
    /// first bin starts at `f64::MIN`, and the last ends at `f64::MAX`.
    /// Weights are **not** normalized. Bins are returned in increasing order;
    /// their `end` values are the candidate split thresholds.
    ///
    /// # Panics
    /// Panics if `indices` is empty, if an index is out of bounds for
    /// `labels` or `dist`, or if `feature[i]` is NaN.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::core::{Feature, binning::Bins};
    ///
    /// let mut x = Feature::dense("x");
    /// for v in [0.0, 1.0, 3.0, 4.0] {
    ///     x.append((0, v));
    /// }
    /// let labels = [-1.0, -1.0, 1.0, 1.0];
    /// let dist = [0.125, 0.125, 0.25, 0.5];
    /// let bins = Bins::cut(&x, 4); // boundaries 1, 2, 3
    ///
    /// // Only examples 0, 1, and 3 are at this node.
    /// let packed = bins.pack(&[0, 1, 3], &x, &labels, &dist);
    /// // They fall in bins (-∞, 1), [1, 2), and [3, ∞). Bin [2, 3) is empty
    /// // and dropped; its neighbors now meet at the midpoint 2.5.
    /// assert_eq!(packed.len(), 3);
    /// let thresholds: Vec<f64> = packed.iter().map(|(bin, _)| bin.end()).collect();
    /// assert_eq!(thresholds, vec![1.0, 2.5, f64::MAX]);
    /// // Per-bin, per-label weights (not normalized).
    /// assert_eq!(packed[0].1[&-1], 0.125);
    /// assert_eq!(packed[2].1[&1], 0.5);
    /// assert!(!packed[2].1.contains_key(&-1));
    /// ```
    pub fn pack(
        &self,
        indices: &[usize],
        feature: &Feature,
        labels: &[f64],
        dist: &[f64],
    ) -> Vec<(Bin, HashMap<i32, f64>)> {
        let n_bins = self.0.len();
        let mut packed = vec![HashMap::<i32, f64>::new(); n_bins];

        let mut values = feature.cursor();
        for &i in indices {
            let xi = values.value(i);
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
        self.drop_empty_bins(packed)
    }

    /// Same as [`Bins::pack`], with the bin of every row looked up in
    /// `bin_index` (the [`BinIndex`] of `feature` for these bins) instead of
    /// searched for, and the labels given as [`RowLabels`]. Weights are added
    /// in the order of `indices`, as in `pack`, so both return the same sums.
    ///
    /// With `totals` (the [`LabelTotals`] of `indices`), a sparse feature
    /// with few stored entries is packed by [`pack_stored_entries`], which
    /// rounds the weights of the bin containing `0.0` differently.
    pub(crate) fn pack_by_index(
        &self,
        indices: &[usize],
        feature: &Feature,
        bin_index: &BinIndex,
        row_labels: RowLabels<'_>,
        dist: &[f64],
        totals: Option<&LabelTotals>,
    ) -> Vec<(Bin, HashMap<i32, f64>)> {
        let RowLabels {
            index: label_index,
            distinct: labels,
        } = row_labels;
        let n_labels = labels.len();
        let mut weights = vec![0.0_f64; self.0.len() * n_labels];
        let mut present = vec![false; self.0.len() * n_labels];
        match bin_index {
            BinIndex::Dense(bin_of_row) => {
                for &i in indices {
                    let cell = bin_of_row[i] as usize * n_labels + label_index[i] as usize;
                    weights[cell] += dist[i];
                    present[cell] = true;
                }
            }
            BinIndex::Sparse { bin_of_entry, zero } => {
                let by_difference = totals.is_some_and(|totals| {
                    pack_stored_entries(
                        indices,
                        feature,
                        (bin_of_entry, *zero),
                        label_index,
                        dist,
                        totals,
                        (&mut weights, &mut present),
                    )
                });
                if !by_difference {
                    weights.fill(0.0);
                    present.fill(false);
                    let mut entries = feature.cursor();
                    for &i in indices {
                        let bin = entries.entry(i).map_or(*zero, |p| bin_of_entry[p]);
                        let cell = bin as usize * n_labels + label_index[i] as usize;
                        weights[cell] += dist[i];
                        present[cell] = true;
                    }
                }
            }
        }
        let packed = weights
            .chunks(n_labels)
            .zip(present.chunks(n_labels))
            .map(|(weights, present)| {
                labels
                    .iter()
                    .zip(weights)
                    .zip(present)
                    .filter(|&(_, &present)| present)
                    .map(|((&y, &w), _)| (y, w))
                    .collect::<HashMap<i32, f64>>()
            })
            .collect();
        self.drop_empty_bins(packed)
    }

    /// This method removes bins that received no example.
    /// # Examples
    /// Assume that we have bins and their weights.
    /// ```text
    /// Bins       | [-3.0, 2.5), [2.5, 7.0), [7.0, 8.1), [8.1, 9.0)
    /// Weights(+) |     0.5,        0.0,        0.0,        0.2
    /// Weights(-) |     0.0,        0.0,        0.0,        0.1
    /// ```
    /// This method removes the empty bins (those whose label-to-weight map is
    /// empty) like this:
    /// ```text
    /// Bins       | [-3.0, 5.3), [5.3, 9.0)
    /// Weights(+) |     0.5,        0.2
    /// Weights(-) |     0.0,        0.1
    /// ```
    /// That is, this method
    /// - changes the bin bounds so that adjacent nonempty bins meet at the
    ///   midpoint of the removed gap,
    /// - sets the outermost bounds to `f64::MIN` and `f64::MAX`.
    ///
    /// Weights are not normalized.
    fn drop_empty_bins(&self, pack: Vec<HashMap<i32, f64>>) -> Vec<(Bin, HashMap<i32, f64>)> {
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

/// The bin ([`Bins::index_of`]) of every value of one feature, so that the
/// bin of a row need not be searched for again.
#[derive(Debug, Clone)]
pub(crate) enum BinIndex {
    /// For a dense feature: the bin of each row.
    Dense(Vec<u32>),
    /// For a sparse feature: the bin of each stored entry, in the order of
    /// the entries, and the bin of the implicit zeros. This keeps the memory
    /// proportional to the number of stored entries.
    Sparse {
        /// The bin of each stored entry.
        bin_of_entry: Vec<u32>,
        /// The bin containing `0.0`.
        zero: u32,
    },
}

/// The labels of a sample's rows as indices into its distinct labels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RowLabels<'a> {
    /// The position in `distinct` of the label of each row.
    pub(crate) index: &'a [u32],
    /// The distinct labels, converted with `as i32`.
    pub(crate) distinct: &'a [i32],
}

/// The total weight and the number of a node's examples, per label.
#[derive(Debug, Clone)]
pub(crate) struct LabelTotals {
    weight: Vec<f64>,
    count: Vec<usize>,
}

impl LabelTotals {
    /// Sums `dist[i]` over `indices` for each of the `n_labels` labels;
    /// `label_index[i]` is the label of row `i`.
    pub(crate) fn new(
        indices: &[usize],
        label_index: &[u32],
        n_labels: usize,
        dist: &[f64],
    ) -> Self {
        let mut weight = vec![0.0; n_labels];
        let mut count = vec![0; n_labels];
        for &i in indices {
            weight[label_index[i] as usize] += dist[i];
            count[label_index[i] as usize] += 1;
        }
        Self { weight, count }
    }
}

/// The implicit zeros of a label must hold at least this fraction of the
/// label's weight in a node for their weight to be computed as a difference
/// in [`pack_stored_entries`]. The difference `a - b` of two sums of `k`
/// positive terms has an absolute error of at most about `k ε a`
/// (`ε = 2⁻⁵³`), so its relative error is at most about `k ε / 10⁻³`: below
/// `10⁻⁶` up to `k = 10⁷` examples.
const MIN_ZERO_FRACTION: f64 = 1e-3;

/// [`pack_stored_entries`] is used on a node with at least this many
/// examples per stored entry of the feature in the node's row range. With
/// fewer, visiting every example is as fast (measured: about twice as fast
/// with 2 examples per entry, 1.6 times slower with 20) and exact.
const MIN_ROWS_PER_ENTRY: usize = 4;

/// A numerical shortcut for [`Bins::pack_by_index`] on a sparse feature: adds
/// the weights of the stored entries among `indices` to their cells, and
/// obtains the weight of the implicit zeros of each label as the label's
/// total minus the weight of its stored entries, instead of adding the
/// zeros one by one. The cost is proportional to the number of stored
/// entries, up to a logarithmic factor, not to the number of examples.
///
/// The cells of bins without implicit zeros get the same sums as in
/// `pack_by_index`; the cells of the zero bin are rounded differently.
/// Returns `false`, leaving the cells in an unspecified state, if the
/// feature has too many stored entries for the shortcut to pay off
/// ([`MIN_ROWS_PER_ENTRY`]), or if the zeros of some label hold less than
/// [`MIN_ZERO_FRACTION`] of its weight: the difference would then lose
/// accuracy to cancellation. The caller must then add the examples one by
/// one.
fn pack_stored_entries(
    indices: &[usize],
    feature: &Feature,
    (bin_of_entry, zero): (&[u32], u32),
    label_index: &[u32],
    dist: &[f64],
    totals: &LabelTotals,
    (weights, present): (&mut [f64], &mut [bool]),
) -> bool {
    let Feature::Sparse { vals, .. } = feature else {
        return false;
    };
    let (Some(&first), Some(&last)) = (indices.first(), indices.last()) else {
        return true;
    };
    let n_labels = totals.weight.len();
    let mut stored_weight = vec![0.0; n_labels];
    let mut stored_count = vec![0_usize; n_labels];
    let mut add = |i: usize, p: usize| {
        let label = label_index[i] as usize;
        let cell = bin_of_entry[p] as usize * n_labels + label;
        weights[cell] += dist[i];
        present[cell] = true;
        stored_weight[label] += dist[i];
        stored_count[label] += 1;
    };
    // The stored entries whose rows lie between the node's first and last.
    let start = vals.partition_point(|&(row, _)| row < first);
    let end = vals.partition_point(|&(row, _)| row <= last);
    // Looking an entry up among the node's rows costs more than visiting a
    // row, so the shortcut pays off only with few entries.
    if (end - start).saturating_mul(MIN_ROWS_PER_ENTRY) > indices.len() {
        return false;
    }
    let mut pos = 0;
    for (p, &(row, _)) in vals.iter().enumerate().take(end).skip(start) {
        pos = gallop(indices, pos, |&i| i < row);
        if indices.get(pos) == Some(&row) {
            add(row, p);
        }
    }
    for label in 0..n_labels {
        if totals.count[label] == stored_count[label] {
            continue;
        }
        let zeros = totals.weight[label] - stored_weight[label];
        // Also false for NaN.
        if !(zeros >= MIN_ZERO_FRACTION * totals.weight[label] && zeros > 0.0) {
            return false;
        }
        let cell = zero as usize * n_labels + label;
        weights[cell] += zeros;
        present[cell] = true;
    }
    true
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
                .map(|bin| format!("{bin}"))
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
    fn pack_by_index_matches_pack_for_dense_and_sparse_features() {
        let size = 30;
        let mut dense = Feature::dense("x");
        let mut sparse = Feature::sparse("x", size);
        for row in 0..size {
            let value = if row % 4 == 1 || row % 9 == 0 {
                row as f64 - 12.5
            } else {
                0.0
            };
            dense.append((row, value));
            sparse.append((row, value));
        }
        let labels: Vec<f64> = (0..size)
            .map(|i| if i % 3 == 0 { 1.0 } else { -1.0 })
            .collect();
        let label_index: Vec<u32> = labels.iter().map(|&y| u32::from(y > 0.0)).collect();
        let total = (size * (size + 1) / 2) as f64;
        let dist: Vec<f64> = (1..=size).map(|i| i as f64 / total).collect();
        let bins = Bins::cut(&sparse, 4);

        for indices in [
            (0..size).collect::<Vec<usize>>(),
            (0..size).step_by(3).collect(),
            vec![2, 3, 27, 28, 29],
        ] {
            let expected = bins.pack(&indices, &dense, &labels, &dist);
            let same = |packed: Vec<(Bin, HashMap<i32, f64>)>| {
                assert_eq!(packed.len(), expected.len());
                for ((bin, weights), (ebin, eweights)) in packed.iter().zip(&expected) {
                    assert_eq!(bin.0, ebin.0);
                    // Exact: the weights are added in the same order.
                    assert_eq!(weights, eweights);
                }
            };
            same(bins.pack(&indices, &sparse, &labels, &dist));
            for feature in [&dense, &sparse] {
                same(bins.pack_by_index(
                    &indices,
                    feature,
                    &bins.index(feature),
                    RowLabels {
                        index: &label_index,
                        distinct: &[-1, 1],
                    },
                    &dist,
                    None,
                ));
            }
        }
    }

    /// 400 rows with 20 stored entries (rows 7, 27, ...), alternating labels,
    /// packed over all rows with and without the stored-entries shortcut.
    fn pack_few_entries(dist: &[f64]) -> [Vec<(Bin, HashMap<i32, f64>)>; 2] {
        let size = 400;
        let mut sparse = Feature::sparse("x", size);
        for row in (7..size).step_by(20) {
            sparse.append((row, 1.0 + (row % 3) as f64));
        }
        let label_index: Vec<u32> = (0..size).map(|i| (i % 2) as u32).collect();
        let indices: Vec<usize> = (0..size).collect();
        let bins = Bins::cut(&sparse, 4);
        let labels = RowLabels {
            index: &label_index,
            distinct: &[-1, 1],
        };
        let totals = LabelTotals::new(&indices, &label_index, 2, dist);
        let index = bins.index(&sparse);
        [None, Some(&totals)]
            .map(|totals| bins.pack_by_index(&indices, &sparse, &index, labels, dist, totals))
    }

    #[test]
    fn stored_entries_shortcut_matches_the_sums_over_all_rows() {
        let dist: Vec<f64> = (0..400).map(|i| (1.0 + (i % 13) as f64) / 2800.0).collect();
        let [exact, shortcut] = pack_few_entries(&dist);
        assert_eq!(exact.len(), shortcut.len());
        for ((bin, weights), (sbin, sweights)) in exact.iter().zip(&shortcut) {
            assert_eq!(bin.0, sbin.0);
            assert_eq!(weights.len(), sweights.len());
            for (label, weight) in weights {
                let other = sweights[label];
                if bin.contains(&0.0) {
                    // A difference of two sums of at most 400 terms below 1.
                    assert!((weight - other).abs() < 400.0 * f64::EPSILON);
                } else {
                    assert_eq!(*weight, other);
                }
            }
        }
    }

    #[test]
    fn stored_entries_shortcut_is_not_used_under_cancellation() {
        // The implicit zeros hold 1e-18 of the weight: their sum cannot be
        // recovered as a difference, so the rows are added one by one.
        let dist: Vec<f64> = (0..400)
            .map(|i| if i % 20 == 7 { 0.05 } else { 1e-20 })
            .collect();
        let [exact, shortcut] = pack_few_entries(&dist);
        assert_eq!(exact.len(), shortcut.len());
        for ((bin, weights), (sbin, sweights)) in exact.iter().zip(&shortcut) {
            assert_eq!(bin.0, sbin.0);
            assert_eq!(weights, sweights);
        }
        let zeros: f64 = exact
            .iter()
            .filter(|(bin, _)| bin.contains(&0.0))
            .flat_map(|(_, weights)| weights.values())
            .sum();
        assert!((zeros - 380.0 * 1e-20).abs() < 1e-30);
    }

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
        let expect = [
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
        let expect = [
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
