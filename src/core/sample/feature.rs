//! The [`Feature`] column type used by [`crate::Sample`].
use std::mem;
use std::ops::Index;

use crate::core::{constants::BUFFER_SIZE, tools::helpers};

/// A named feature column of a [`crate::Sample`].
///
/// A column is either dense (one value per example) or sparse (only nonzero
/// entries stored). Sparse entries must have unique, strictly increasing row
/// indices below `size`; omitted entries are zero. [`Feature::append`]
/// preserves insertion order and does not sort, so callers must append sparse
/// entries in increasing row order; otherwise indexing (`feature[i]`, which
/// uses binary search) may return wrong values.
///
/// `feature[i]` returns the value of row `i` (zero for omitted sparse rows).
/// It panics for an out-of-bounds dense row; for a sparse column any row
/// not stored returns `0.0`, even if `i >= size`.
///
/// # Examples
/// Columns are usually obtained from a [`crate::Sample`]:
/// ```
/// use std::io::BufReader;
/// use miniboosts::Sample;
///
/// let csv = b"x,class\n-2,-1\n0,-1\n3,1\n";
/// let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
///     .unwrap()
///     .set_target("class");
/// let x = &sample["x"];
/// assert_eq!(x.name(), "x");
/// assert!(x.is_dense()); // CSV columns are stored densely.
/// assert_eq!(x[2], 3.0);
/// assert_eq!(x.clone().into_vals(), vec![-2.0, 0.0, 3.0]);
/// ```
///
/// Building columns by hand: a sparse column stores only nonzero entries,
/// appended in increasing row order.
/// ```
/// use miniboosts::core::Feature;
///
/// let mut dense = Feature::dense("x");
/// let mut sparse = Feature::sparse("x", 4);
/// for (row, value) in [(0, 1.5), (1, 0.0), (2, 0.0), (3, -2.0)] {
///     dense.append((row, value));
///     sparse.append((row, value));
/// }
/// // Both represent the same values.
/// for row in 0..4 {
///     assert_eq!(dense[row], sparse[row]);
/// }
/// assert_eq!(sparse.into_vals(), dense.into_vals());
/// ```
#[derive(Debug, Clone)]
pub enum Feature {
    /// A column storing every value explicitly.
    Dense {
        /// The column name.
        name: String,
        /// The value of each example, in row order.
        vals: Vec<f64>,
    },
    /// A column storing only nonzero values.
    Sparse {
        /// The column name.
        name: String,
        /// `(row, value)` pairs of the nonzero entries, in increasing row
        /// order.
        vals: Vec<(usize, f64)>,
        /// The number of rows (examples), including implicit zeros.
        size: usize,
    },
}

impl Feature {
    /// Returns the column name.
    pub fn name(&self) -> &str {
        match self {
            Self::Dense { name, .. } => name,
            Self::Sparse { name, .. } => name,
        }
    }

    /// Creates an empty dense column named `name`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::core::Feature;
    ///
    /// let mut f = Feature::dense("age");
    /// assert!(f.is_empty());
    /// f.append((0, 31.0)); // the row index is ignored for dense columns
    /// f.append((0, 0.0));
    /// assert_eq!((f[0], f[1]), (31.0, 0.0));
    /// assert!(f.has_zero());
    /// ```
    pub fn dense<T: ToString>(name: T) -> Self {
        Self::Dense {
            name: name.to_string(),
            vals: Vec::with_capacity(BUFFER_SIZE),
        }
    }

    /// Creates an empty sparse column named `name` with `size` rows,
    /// all initially zero.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::core::Feature;
    ///
    /// let mut f = Feature::sparse("word_17", 1_000);
    /// assert!(f.is_sparse());
    /// f.append((3, 2.0));
    /// f.append((10, 0.0)); // zeros are not stored
    /// f.append((500, 1.0));
    /// assert_eq!((f[3], f[10], f[500], f[999]), (2.0, 0.0, 1.0, 0.0));
    /// // Distinct values: 0 (implicit), 1, and 2.
    /// assert_eq!(f.distinct_value_count(), 3);
    /// ```
    pub fn sparse<T: ToString>(name: T, size: usize) -> Self {
        Self::Sparse {
            name: name.to_string(),
            vals: Vec::with_capacity(BUFFER_SIZE),
            size,
        }
    }

    /// Consumes the column and returns its values as a dense vector:
    /// `vals` for a dense column, or a length-`size` vector with implicit
    /// zeros filled in for a sparse column.
    ///
    /// # Panics
    /// Panics if a sparse entry has a row index `>= size`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::core::Feature;
    ///
    /// let mut f = Feature::sparse("x", 4);
    /// f.append((1, 5.0));
    /// assert_eq!(f.into_vals(), vec![0.0, 5.0, 0.0, 0.0]);
    /// ```
    pub fn into_vals(self) -> Vec<f64> {
        match self {
            Self::Dense { vals, .. } => vals,
            Self::Sparse { vals, size, .. } => {
                let mut ret = vec![0f64; size];
                vals.into_iter().for_each(|(i, v)| {
                    ret[i] = v;
                });
                ret
            }
        }
    }

    /// Returns `true` for a [`Feature::Sparse`] column.
    pub fn is_sparse(&self) -> bool {
        match self {
            Self::Dense { .. } => false,
            Self::Sparse { .. } => true,
        }
    }

    /// Returns `true` for a [`Feature::Dense`] column.
    pub fn is_dense(&self) -> bool {
        !self.is_sparse()
    }

    /// Appends the value `val` of row `ix`.
    ///
    /// For a dense column, `ix` is ignored and `val` is pushed to the end, so
    /// rows must be appended in order. For a sparse column, `(ix, val)` is
    /// pushed only if `val != 0.0`; `ix` must exceed every stored row index
    /// and be below `size` (not checked).
    pub fn append(&mut self, (ix, val): (usize, f64)) {
        match self {
            Self::Dense { vals, .. } => vals.push(val),
            Self::Sparse { vals, .. } => {
                if val != 0f64 {
                    vals.push((ix, val));
                }
            }
        }
    }

    pub(crate) fn replace_name<T>(&mut self, name: T) -> String
    where
        T: ToString,
    {
        let n = name.to_string();
        match self {
            Self::Dense { name, .. } => mem::replace(name, n),
            Self::Sparse { name, .. } => mem::replace(name, n),
        }
    }

    pub(crate) fn set_size(&mut self, size: usize) {
        let s = size;
        if let Self::Sparse { size, .. } = self {
            *size = s;
        }
    }

    /// Returns `true` if no value is stored: a dense column with no rows, or
    /// a sparse column with no nonzero entry (regardless of `size`).
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Dense { vals, .. } => vals.is_empty(),
            Self::Sparse { vals, .. } => vals.is_empty(),
        }
    }

    fn zero_counts(&self) -> usize {
        match self {
            Self::Dense { vals, .. } => vals.iter().filter(|&&v| v == 0f64).count(),
            Self::Sparse { vals, size, .. } => size - vals.len(),
        }
    }

    /// Returns `true` if at least one row has value `0.0` (explicit for a
    /// dense column, implicit for a sparse one).
    pub fn has_zero(&self) -> bool {
        self.zero_counts() > 0
    }

    /// Returns the number of distinct values in the column, counting an
    /// implicit sparse zero once. Returns `0` for an empty dense column.
    ///
    /// # Panics
    /// Panics if the column contains NaN.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::core::Feature;
    ///
    /// let mut f = Feature::dense("x");
    /// for v in [1.0, 2.0, 1.0, 2.0, 3.0] {
    ///     f.append((0, v));
    /// }
    /// assert_eq!(f.distinct_value_count(), 3);
    /// assert_eq!(Feature::dense("empty").distinct_value_count(), 0);
    /// ```
    pub fn distinct_value_count(&self) -> usize {
        let mut values = match self {
            Self::Dense { vals, .. } => vals.clone(),
            Self::Sparse { vals, .. } => {
                let mut values = vals.iter().map(|&(_, v)| v).collect::<Vec<_>>();
                if self.has_zero() {
                    values.push(0f64);
                }
                values
            }
        };

        if values.is_empty() {
            return 0;
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (mut count, mut value) = (1, values[0]);
        for &v in values.iter().skip(1) {
            if v != value {
                value = v;
                count += 1;
            }
        }
        count
    }

    /// Returns `Σ_i weights[i] * x_i`, the weighted mean when `weights` is a
    /// probability vector. Weights are not normalized or validated.
    ///
    /// For a dense column, extra entries of the longer of `weights` and the
    /// column are ignored. For a sparse column, panics if `weights` is shorter
    /// than a stored row index requires.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::core::Feature;
    ///
    /// let mut f = Feature::sparse("x", 4);
    /// f.append((0, 4.0));
    /// f.append((3, 8.0));
    /// // 0.25 * 4 + 0.25 * 8 = 3, exact in floating point.
    /// assert_eq!(f.weighted_mean(&[0.25; 4]), 3.0);
    /// ```
    pub fn weighted_mean(&self, weights: &[f64]) -> f64 {
        match self {
            Self::Dense { vals, .. } => helpers::inner_product(weights, &vals[..]),
            Self::Sparse { vals, .. } => vals.iter().map(|&(ix, val)| weights[ix] * val).sum(),
        }
    }

    /// Returns `Σ_i weights[i] * (x_i - mean)^2`, the weighted population
    /// variance about `mean` when `weights` is a probability vector.
    /// Weights are not normalized or validated. For a sparse column, the
    /// implicit zeros contribute `(Σ_i weights[i] - Σ_{nonzero i} weights[i]) * mean^2`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::core::Feature;
    ///
    /// let mut f = Feature::sparse("x", 4);
    /// f.append((0, 4.0));
    /// f.append((3, 8.0));
    /// // Values (4, 0, 0, 8) with uniform weights: mean 3 and variance
    /// // (1 + 9 + 9 + 25) / 4 = 11, both exact.
    /// assert_eq!(f.weighted_variance(3.0, &[0.25; 4]), 11.0);
    /// ```
    pub fn weighted_variance(&self, mean: f64, weights: &[f64]) -> f64 {
        match self {
            Self::Dense { vals, .. } => vals
                .iter()
                .zip(weights)
                .map(|(&v, &w)| w * (v - mean).powi(2))
                .sum(),
            Self::Sparse { vals, .. } => {
                let w0 = {
                    let wsum = weights.iter().sum::<f64>();
                    let nz_w = vals.iter().map(|&(i, _)| weights[i]).sum::<f64>();
                    wsum - nz_w
                };
                vals.iter()
                    .map(|&(i, v)| weights[i] * (v - mean).powi(2))
                    .sum::<f64>()
                    + w0 * mean.powi(2)
            }
        }
    }

    // Validate the conditional distribution once for both representations.
    fn label_mass(&self, label: f64, labels: &[f64], weights: &[f64]) -> f64 {
        let size = match self {
            Self::Dense { vals, .. } => vals.len(),
            Self::Sparse { size, .. } => *size,
        };
        assert_eq!(labels.len(), size, "labels must match the feature length");
        assert_eq!(weights.len(), size, "weights must match the feature length");
        assert!(
            weights.iter().all(|w| w.is_finite() && *w >= 0.0),
            "weights must be finite and nonnegative"
        );
        let total = labels
            .iter()
            .zip(weights)
            .filter(|&(&y, _)| y == label)
            .map(|(_, &w)| w)
            .sum::<f64>();
        assert!(
            total.is_finite() && total > 0.0,
            "weight sum for label y = {label} must be finite and positive"
        );
        total
    }

    /// Conditional weighted mean over rows whose label equals `label`:
    /// `Σ_{i: y_i = label} w_i x_i / Σ_{i: y_i = label} w_i`.
    /// Labels and weights must match the feature length. Weights need not sum
    /// to one, but must be finite and nonnegative.
    ///
    /// # Panics
    /// Panics if the selected class is absent, has zero/non-finite total
    /// weight, or dimensions are invalid.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::core::Feature;
    ///
    /// let mut f = Feature::dense("x");
    /// for v in [2.0, 10.0, 0.0, 4.0] {
    ///     f.append((0, v));
    /// }
    /// let labels = [1.0, -1.0, 1.0, 1.0];
    /// let weights = [3.0, 7.0, 2.0, 1.0];
    /// // (3 * 2 + 2 * 0 + 1 * 4) / (3 + 2 + 1) = 10 / 6.
    /// let mean = f.weighted_mean_for_label(1.0, &labels[..], &weights[..]);
    /// // Tolerance: a few ulps of the rounded quotient.
    /// assert!((mean - 10.0 / 6.0).abs() < 1e-15);
    /// ```
    pub fn weighted_mean_for_label<T>(&self, label: f64, labels: T, weights: T) -> f64
    where
        T: AsRef<[f64]>,
    {
        let labels = labels.as_ref();
        let weights = weights.as_ref();
        let total = self.label_mass(label, labels, weights);
        let numerator = match self {
            Self::Dense { vals, .. } => vals
                .iter()
                .zip(weights)
                .zip(labels)
                .filter(|&(_, &y)| y == label)
                .map(|((&v, &w), _)| v * w)
                .sum::<f64>(),
            Self::Sparse { vals, .. } => vals
                .iter()
                .filter(|&&(i, _)| labels[i] == label)
                .map(|&(i, v)| weights[i] * v)
                .sum::<f64>(),
        };
        numerator / total
    }

    /// Conditional weighted population variance about `mean`:
    /// `Σ_{i: y_i = label} w_i (x_i - mean)^2 / Σ_{i: y_i = label} w_i`.
    /// Uses the same weight and dimension contract as
    /// [`Feature::weighted_mean_for_label`].
    /// Implicit sparse zeros contribute only when their label matches.
    ///
    /// # Panics
    /// Same conditions as [`Feature::weighted_mean_for_label`].
    ///
    /// # Examples
    /// ```
    /// use miniboosts::core::Feature;
    ///
    /// let mut f = Feature::sparse("x", 3);
    /// f.append((0, 2.0));
    /// f.append((2, 6.0));
    /// let labels = [1.0, 1.0, -1.0];
    /// // Label +1 selects values 2 and 0 (implicit) with equal weight:
    /// // about mean 1, the variance is (1 + 1) / 2 = 1.
    /// assert_eq!(f.weighted_variance_for_label(1.0, 1.0, &labels, &[0.5; 3]), 1.0);
    /// ```
    pub fn weighted_variance_for_label(
        &self,
        mean: f64,
        label: f64,
        labels: &[f64],
        weights: &[f64],
    ) -> f64 {
        let total = self.label_mass(label, labels, weights);
        match self {
            Self::Dense { vals, .. } => {
                vals.iter()
                    .zip(weights)
                    .zip(labels)
                    .filter(|&(_, &y)| y == label)
                    .map(|((&v, &w), _)| w * (v - mean).powi(2))
                    .sum::<f64>()
                    / total
            }
            Self::Sparse { vals, .. } => {
                let nonzero = vals
                    .iter()
                    .filter(|&&(i, _)| labels[i] == label)
                    .map(|&(i, _)| weights[i])
                    .sum::<f64>();
                let numerator = vals
                    .iter()
                    .filter(|&&(i, _)| labels[i] == label)
                    .map(|&(i, v)| weights[i] * (v - mean).powi(2))
                    .sum::<f64>();
                (numerator + (total - nonzero).max(0.0) * mean.powi(2)) / total
            }
        }
    }

    /// Returns `(m, v)` where `m = `[`Feature::weighted_mean`]`(weights)` and
    /// `v = `[`Feature::weighted_variance`]`(m, weights)`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::core::Feature;
    ///
    /// let mut f = Feature::dense("x");
    /// for v in [1.0, 3.0] {
    ///     f.append((0, v));
    /// }
    /// assert_eq!(f.weighted_mean_and_variance([0.5, 0.5]), (2.0, 1.0));
    /// ```
    pub fn weighted_mean_and_variance<T>(&self, weights: T) -> (f64, f64)
    where
        T: AsRef<[f64]>,
    {
        let weights = weights.as_ref();
        let mean = self.weighted_mean(weights);
        let vars = self.weighted_variance(mean, weights);
        (mean, vars)
    }

    /// Returns `(m, v)` where `m` is
    /// [`Feature::weighted_mean_for_label`] and `v` is
    /// [`Feature::weighted_variance_for_label`] about `m`.
    ///
    /// # Panics
    /// Same conditions as [`Feature::weighted_mean_for_label`].
    ///
    /// # Examples
    /// ```
    /// use miniboosts::core::Feature;
    ///
    /// let mut f = Feature::dense("x");
    /// for v in [1.0, 100.0, 3.0] {
    ///     f.append((0, v));
    /// }
    /// let labels = [1.0, -1.0, 1.0];
    /// let stats = f.weighted_mean_and_variance_for_label(1.0, &labels[..], &[1.0; 3][..]);
    /// assert_eq!(stats, (2.0, 1.0)); // the -1 example (100) is excluded
    /// ```
    pub fn weighted_mean_and_variance_for_label<T>(
        &self,
        label: f64,
        labels: T,
        weights: T,
    ) -> (f64, f64)
    where
        T: AsRef<[f64]>,
    {
        let labels = labels.as_ref();
        let weights = weights.as_ref();
        let mean = self.weighted_mean_for_label(label, labels, weights);
        let vars = self.weighted_variance_for_label(mean, label, labels, weights);
        (mean, vars)
    }
}

impl Feature {
    /// Returns a reader of this feature's values for rows visited in
    /// increasing order; see [`Cursor`].
    pub(crate) fn cursor(&self) -> Cursor<'_> {
        Cursor {
            feature: self,
            pos: 0,
            last: 0,
        }
    }

    /// Splits `rows` into those whose value is `< threshold` and the others
    /// (including NaN), keeping their order: the rule of
    /// [`Splitter::split`](crate::core::tree::Splitter::split), with the
    /// values read as by a [`Cursor`].
    pub(crate) fn partition_rows(
        &self,
        rows: Vec<usize>,
        threshold: f64,
    ) -> (Vec<usize>, Vec<usize>) {
        match self {
            Self::Dense { vals, .. } => rows.into_iter().partition(|&row| vals[row] < threshold),
            Self::Sparse { .. } => {
                let mut values = self.cursor();
                rows.into_iter()
                    .partition(|&row| values.value(row) < threshold)
            }
        }
    }
}

/// Reads the values of a [`Feature`] at given rows. The values equal
/// `feature[row]`. On a sparse feature, `feature[row]` searches all nonzero
/// entries; a cursor continues from the entry of the previous row, so that
/// visiting `k` increasing rows costs `O(k log(nnz / k))` rather than
/// `O(k log nnz)`, and `O(k)` when the rows are consecutive entries. Rows may
/// also decrease, at the cost of a search from the start.
#[derive(Debug, Clone)]
pub(crate) struct Cursor<'a> {
    feature: &'a Feature,
    // Sparse features only: every entry before `pos` has a row `< last`.
    pos: usize,
    last: usize,
}

impl Cursor<'_> {
    /// Returns the value at `row`, like `feature[row]`.
    ///
    /// # Panics
    /// Panics if `row` is out of bounds for a dense feature.
    #[inline]
    pub(crate) fn value(&mut self, row: usize) -> f64 {
        match self.feature {
            Feature::Dense { vals, .. } => vals[row],
            Feature::Sparse { vals, .. } => self.entry(row).map_or(0.0, |p| vals[p].1),
        }
    }

    /// For a sparse feature, returns the position of `row` among the stored
    /// entries, or `None` if the value at `row` is an implicit zero. Always
    /// `None` for a dense feature.
    #[inline]
    pub(crate) fn entry(&mut self, row: usize) -> Option<usize> {
        let Feature::Sparse { vals, .. } = self.feature else {
            return None;
        };
        if row < self.last {
            self.pos = 0;
        }
        self.last = row;
        self.pos = gallop(vals, self.pos, |&(r, _)| r < row);
        (self.pos < vals.len() && vals[self.pos].0 == row).then_some(self.pos)
    }
}

/// Returns the first position `p >= from` with `!before(&items[p])`, or
/// `items.len()`. `before` must hold on a prefix of `items[from..]` only.
/// The step doubles until the answer is passed and the last interval is then
/// searched, which costs `O(log(p - from))`.
#[inline]
pub(crate) fn gallop<T>(items: &[T], from: usize, before: impl Fn(&T) -> bool) -> usize {
    let (mut lo, mut hi, mut step) = (from, from, 1);
    while hi < items.len() && before(&items[hi]) {
        lo = hi + 1;
        hi += step;
        step *= 2;
    }
    let hi = hi.min(items.len());
    lo + items[lo..hi].partition_point(before)
}

/// Returns `Σ_k term(k, x_k)` for every row of `columns`, a set of features
/// with `n_rows` rows of which at least one is sparse, in time proportional
/// to the dense values and the stored sparse entries rather than to
/// `n_rows × columns.len()`.
///
/// A sparse feature `k` contributes `term(k, 0.0)` to a base value shared by
/// all rows, and `term(k, x) - term(k, 0.0)` to the rows where it stores a
/// nonzero `x`. This is a different order of summation than adding
/// `term(k, x_k)` feature by feature, so the result differs from that sum by
/// rounding. [`separable_sum_at`] performs the same operations for one row
/// and returns the same bits.
pub(crate) fn separable_sums(
    columns: &[&Feature],
    n_rows: usize,
    term: impl Fn(usize, f64) -> f64,
) -> Vec<f64> {
    let mut sums = vec![separable_base(columns, &term); n_rows];
    for (k, feature) in columns.iter().enumerate() {
        match feature {
            Feature::Dense { vals, .. } => {
                for (sum, &x) in sums.iter_mut().zip(vals) {
                    *sum += term(k, x);
                }
            }
            Feature::Sparse { vals, .. } => {
                let at_zero = term(k, 0.0);
                for &(row, x) in vals {
                    if x != 0.0 {
                        sums[row] += term(k, x) - at_zero;
                    }
                }
            }
        }
    }
    sums
}

/// The entry of [`separable_sums`] for `row`.
pub(crate) fn separable_sum_at(
    columns: &[&Feature],
    row: usize,
    term: impl Fn(usize, f64) -> f64,
) -> f64 {
    let mut sum = separable_base(columns, &term);
    for (k, feature) in columns.iter().enumerate() {
        let x = feature[row];
        if feature.is_dense() {
            sum += term(k, x);
        } else if x != 0.0 {
            sum += term(k, x) - term(k, 0.0);
        }
    }
    sum
}

/// The sum of `term(k, 0.0)` over the sparse features.
fn separable_base(columns: &[&Feature], term: &impl Fn(usize, f64) -> f64) -> f64 {
    columns
        .iter()
        .enumerate()
        .filter(|(_, feature)| feature.is_sparse())
        .map(|(k, _)| term(k, 0.0))
        .sum()
}

impl Index<usize> for Feature {
    type Output = f64;
    fn index(&self, idx: usize) -> &Self::Output {
        match self {
            Self::Dense { vals, .. } => &vals[idx],
            Self::Sparse { vals, .. } => {
                let pos = vals.binary_search_by(|(i, _)| i.cmp(&idx));
                match pos {
                    Ok(p) => &vals[p].1,
                    Err(_) => &0f64,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dense and a sparse feature with the same 40 values; 8 are nonzero.
    fn dense_and_sparse() -> (Feature, Feature) {
        let size = 40;
        let mut dense = Feature::dense("x");
        let mut sparse = Feature::sparse("x", size);
        for row in 0..size {
            let value = if row % 7 == 3 || row % 11 == 0 {
                row as f64 - 20.5
            } else {
                0.0
            };
            dense.append((row, value));
            sparse.append((row, value));
        }
        (dense, sparse)
    }

    #[test]
    fn cursor_and_partition_agree_with_indexing() {
        let (dense, sparse) = dense_and_sparse();
        let orders: [Vec<usize>; 5] = [
            (0..40).collect(),
            (0..40).step_by(3).collect(),
            vec![33, 36, 38, 39],
            // Not increasing, with a repetition.
            vec![39, 0, 22, 22, 3, 38, 11],
            vec![],
        ];
        for rows in orders {
            for feature in [&dense, &sparse] {
                let mut cursor = feature.cursor();
                for &row in &rows {
                    assert_eq!(cursor.value(row).to_bits(), feature[row].to_bits());
                }
                for threshold in [-5.0, 0.0, 0.5, f64::NAN] {
                    let expected: (Vec<usize>, Vec<usize>) =
                        rows.iter().partition(|&&row| feature[row] < threshold);
                    assert_eq!(feature.partition_rows(rows.clone(), threshold), expected);
                }
            }
        }
    }

    #[test]
    fn cursor_entry_is_the_position_of_a_stored_row() {
        let (dense, sparse) = dense_and_sparse();
        let Feature::Sparse { vals, .. } = &sparse else {
            unreachable!()
        };
        let mut cursor = sparse.cursor();
        for row in 0..40 {
            let expected = vals.iter().position(|&(r, _)| r == row);
            assert_eq!(cursor.entry(row), expected, "row {row}");
        }
        assert_eq!(dense.cursor().entry(3), None);
    }

    #[test]
    fn conditional_statistics_match_hand_calculation() {
        let mut dense = Feature::dense("x");
        let mut sparse = Feature::sparse("x", 4);
        for (i, value) in [2.0, 10.0, 0.0, 4.0].into_iter().enumerate() {
            dense.append((i, value));
            sparse.append((i, value));
        }
        let labels = [1.0, -1.0, 1.0, 1.0];
        // Selected mass is 6, mean is 10/6, variance is 17/9.
        let weights = [3.0, 7.0, 2.0, 1.0];
        for feature in [dense, sparse] {
            let (mean, variance) =
                feature.weighted_mean_and_variance_for_label(1.0, &labels[..], &weights[..]);
            assert!((mean - 5.0 / 3.0).abs() < 1e-12);
            assert!((variance - 17.0 / 9.0).abs() < 1e-12);
            assert_eq!(
                feature.weighted_mean_and_variance_for_label(-1.0, &labels[..], &weights[..]),
                (10.0, 0.0)
            );
            for label in [2.0, 1.0] {
                let zero_weights = [0.0; 4];
                assert!(
                    std::panic::catch_unwind(|| feature.weighted_mean_for_label(
                        label,
                        &labels[..],
                        &zero_weights[..]
                    ))
                    .is_err()
                );
                assert!(
                    std::panic::catch_unwind(|| feature.weighted_variance_for_label(
                        0.0,
                        label,
                        &labels,
                        &zero_weights
                    ))
                    .is_err()
                );
            }
            assert!(
                std::panic::catch_unwind(|| feature.weighted_mean_for_label(
                    1.0,
                    &labels[..],
                    &[1.0][..]
                ))
                .is_err()
            );
        }
    }

    #[test]
    fn test_into_vals_02() {
        let n = "test-004";
        let s = 10;
        let mut f = Feature::sparse(n, s);
        f.append((0, 0.7));
        f.append((8, 0.0));
        f.append((9, 9.3));
        f.append((5, -30.4));
        f.append((3, 300.8));

        let result = f.into_vals();
        let expect = vec![0.7, 0.0, 0.0, 300.8, 0.0, -30.4, 0.0, 0.0, 0.0, 9.3];
        assert_eq!(
            result.len(),
            expect.len(),
            "expected {length}-length vector, got {result:?}",
            length = expect.len(),
        );

        for (r, e) in result.into_iter().zip(expect) {
            assert_eq!(r, e, "expected {e}, got {r}");
        }
    }

    #[test]
    fn zero_counts_include_the_rows_a_sparse_feature_does_not_store() {
        let entries = [(0, 0.7), (8, 0.0), (9, 9.3), (5, -30.4), (3, 300.8)];
        let mut dense = Feature::dense("x");
        let mut sparse = Feature::sparse("x", 1_000);
        for entry in entries {
            dense.append(entry);
            sparse.append(entry);
        }
        assert_eq!(dense.zero_counts(), 1);
        assert_eq!(sparse.zero_counts(), 996);
        assert!(dense.is_dense() && !dense.is_sparse());
        assert!(sparse.is_sparse() && !sparse.is_dense());
        assert_eq!((dense.name(), sparse.name()), ("x", "x"));
    }
}
