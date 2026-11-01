//! Margin-preserving sparsification of hypothesis weights, the second stage
//! of [`SparsiBoost`](crate::SparsiBoost).
//!
//! Given a margin matrix `U ∈ [-1, 1]^{n×m}` (`u_ij = y_i h_j(x_i)`), weights
//! `w ∈ R^m` with `‖w‖_1 = 1`, and a target `T`, [`sparsify`] returns `w'`
//! with `‖w'‖_1 = 1` and `‖w'‖_0 ≤ T` such that every margin `(U w')_i`
//! stays close to `(U w)_i`. It follows Algorithm 2 (Sparsification) of
//! Mathiasen, Larsen, and Grønlund (ICML 2019); the column subset of each
//! halving step comes from [`discrepancy_coloring`].
//!
//! [`sparsify_within`] instead stops on a measured accuracy: it returns the
//! sparsest weights on the same halving path whose margins all stay within a
//! given distance of the original ones, optionally capped at `T` entries.
//!
//! The matrix is passed column by column: `columns[j][i] = u_ij`, the margin
//! of hypothesis `j` on example `i`. [`sparsify_ensemble`] builds these
//! columns from a [`WeightedMajority`] and a [`Sample`].
//!
//! # Reference
//! Allan Grønlund, Kasper Green Larsen, and Alexander Mathiasen.
//! *Optimal Minimal Margin Maximization with Boosting.*
//! Proceedings of the 36th International Conference on Machine Learning
//! (ICML), PMLR 97:4392–4401, 2019. <https://arxiv.org/abs/1901.10789>
//!
//! # Examples
//! Sparsify a uniform combination of 16 hypotheses down to 4 and measure how
//! far the margins moved.
//! ```
//! use miniboosts::optimization::sparsification::{max_margin_change, sparsify};
//!
//! // 16 hypotheses on 8 examples with a fixed ±1 margin pattern.
//! let columns: Vec<Vec<f64>> = (0..16)
//!     .map(|j| {
//!         (0..8)
//!             .map(|i| if (j * 37 + i * 11) % 7 < 4 { 1.0 } else { -1.0 })
//!             .collect()
//!     })
//!     .collect();
//! let weights = vec![1.0 / 16.0; 16];
//!
//! let sparse = sparsify(&columns, &weights, 4, 0);
//! assert!(sparse.iter().filter(|w| **w != 0.0).count() <= 4);
//! let l1: f64 = sparse.iter().map(|w| w.abs()).sum();
//! assert!((l1 - 1.0).abs() < 1e-12);
//!
//! // Margins live in [-1, 1], so they can move by at most 2.
//! let change = max_margin_change(&columns, &weights, &sparse);
//! assert!((0.0..=2.0).contains(&change));
//! ```
use crate::core::{Classifier, Sample};
use crate::hypotheses::WeightedMajority;

/// Returns weights `w'` with `‖w'‖_1 = 1` and at most `target` nonzero
/// entries that approximately preserve every margin `(U w)_i`.
///
/// `columns[j]` holds the margins `u_ij = y_i h_j(x_i)` of hypothesis `j` on
/// the `n` examples, and `weights[j]` is the weight of hypothesis `j`. The
/// weights are first normalized to `‖w‖_1 = 1`; if they already have at most
/// `target` nonzero entries, the normalized weights are returned.
///
/// # Algorithm
/// Algorithm 2 of the paper. While `‖w'‖_0 > T`:
/// 1. Let `R` be the indices of the `‖w'‖_0 / 3` entries of `w'` with largest
///    absolute value, and `ω` the largest `|w'_j|` outside `R`.
/// 2. Twice: let `π(1), ..., π(k)` be the nonzero entries outside `R` and
///    form `A ∈ R^{(n+1)×k}` with `a_ij = u_{iπ(j)} w'_{π(j)} / ω` and the
///    extra row `a_{(n+1)j} = |w'_{π(j)}| / ω`, which keeps `‖w'‖_1` nearly
///    unchanged. Find a coloring `x ∈ {-1, +1}^k` with small `‖A x‖_∞`, let
///    `σ` be the sign taken by at most `k/2` entries of `x`, double
///    `w'_{π(j)}` where `x_j = σ`, and zero it elsewhere.
/// 3. Normalize `w' ← w' / ‖w'‖_1`.
///
/// Each pass keeps at most `‖w'‖_0 / 3 + (2‖w'‖_0 / 3) / 4 = ‖w'‖_0 / 2`
/// entries. Weights are only copied, doubled, or zeroed, so signs are kept
/// and the support of `w'` is a subset of the support of `w`.
///
/// Implementation choices (not part of the original method):
/// - `|R| = max(1, ⌊‖w'‖_0 / 3⌋)`, so `R` is never empty and every pass
///   strictly reduces `‖w'‖_0` for `‖w'‖_0 ≥ 2`. Ties in `|w'_j|` are broken
///   by the smaller index.
/// - When both signs occur exactly `k/2` times, `σ = +1`. Both choices give
///   the same row errors `±(A x)_i`.
/// - The coloring comes from [`discrepancy_coloring`], a heuristic; see
///   *Limitations*.
///
/// # Theoretical guarantee
/// Theorem 2.2 (Sparsification Theorem): for every `U ∈ [-1, 1]^{n×m}`,
/// `w` with `‖w‖_1 = 1`, and `T ≤ m`, there is `w'` with `‖w'‖_1 = 1`,
/// `‖w'‖_0 ≤ T`, and `‖U w - U w'‖_∞ = O(√(lg(2 + n/T) / T))`. The proof is
/// constructive and yields this `w'` whenever each coloring meets
/// Spencer's bound `‖A x‖_∞ = O(√(k lg(2 + n/k)))` (Theorem 2.3).
///
/// # Limitations
/// - [`discrepancy_coloring`] does not certify Spencer's bound, so the
///   theorem's guarantee is not certified either. Measure the result with
///   [`max_margin_change`].
/// - A pass can remove more entries than needed, so the result may have far
///   fewer than `target` nonzero entries (for example `‖w‖_0 = 100`,
///   `target = 99` gives at most 50).
/// - The output depends on `seed`; equal seeds give equal outputs.
/// - Time `O(Σ (n k² + k³))` over the halving steps, dominated by the first
///   one with `k ≈ 2‖w‖_0 / 3`, and `O(n k)` memory.
///
/// # Panics
/// Panics if `target == 0`, if `columns` and `weights` differ in length, if
/// the columns differ in length or are empty, if an entry of `columns` is not
/// in `[-1, 1]`, or if a weight is not finite or all weights are zero.
///
/// # Examples
/// One dominant weight survives every halving step, because the largest
/// third of the entries is copied unchanged.
/// ```
/// use miniboosts::optimization::sparsification::sparsify;
///
/// let columns: Vec<Vec<f64>> = (0..9)
///     .map(|j| (0..5).map(|i| if (i + j) % 3 == 0 { -1.0 } else { 1.0 }).collect())
///     .collect();
/// let mut weights = vec![1.0; 9];
/// weights[4] = 100.0;
///
/// let sparse = sparsify(&columns, &weights, 3, 7);
/// assert!(sparse.iter().filter(|w| **w != 0.0).count() <= 3);
/// assert!(sparse[4] > 0.9);
/// // Signs are kept: nonnegative weights stay nonnegative.
/// assert!(sparse.iter().all(|w| *w >= 0.0));
/// // The same seed gives the same result.
/// assert_eq!(sparse, sparsify(&columns, &weights, 3, 7));
/// ```
pub fn sparsify(columns: &[Vec<f64>], weights: &[f64], target: usize, seed: u64) -> Vec<f64> {
    assert!(target > 0, "sparsification target must be positive");
    let mut w = normalized_weights(columns, weights);
    let mut rng = SplitMix64::new(seed);
    while nonzeros(&w) > target {
        halving_pass(columns, &mut w, &mut rng);
    }
    w
}

/// Returns the sparsest weights `w'` on the halving path of [`sparsify`]
/// whose margins stay within `max_change` of the original ones, capped at
/// `max_nonzeros` nonzero entries if given.
///
/// The weights are normalized to `‖w‖_1 = 1`, and the halving passes of
/// Algorithm 2 (see [`sparsify`]) are applied one at a time. After each pass
/// the change `‖U w - U w'‖_∞` relative to the normalized input is measured
/// with [`max_margin_change`]:
/// - while it is at most `max_change`, the pass is accepted and halving
///   continues (until one nonzero entry is left);
/// - at the first pass that exceeds `max_change`, the previous weights are
///   returned, unless they have more than `max_nonzeros` nonzero entries.
///   In that case the cap takes precedence: halving continues until at most
///   `max_nonzeros` entries are left, and the result may exceed
///   `max_change`. Check it with [`max_margin_change`].
///
/// This turns the accuracy-driven description of SparsiBoost (Section 1.2
/// of the paper: change every margin by at most `v/2`) into a rule that
/// needs no constants hidden in `O(·)`: the change is measured, not bounded
/// by Theorem 2.2. Stopping at the first pass that exceeds `max_change` is
/// an implementation choice; a later pass could in principle move margins
/// back.
///
/// # Panics
/// Panics if `max_change` is negative or NaN, if `max_nonzeros == Some(0)`,
/// or for the inputs on which [`sparsify`] panics.
///
/// # Examples
/// ```
/// use miniboosts::optimization::sparsification::{max_margin_change, sparsify_within};
///
/// let columns: Vec<Vec<f64>> = (0..32)
///     .map(|j| (0..10).map(|i| if (j * 7 + i * 3) % 5 < 3 { 1.0 } else { -1.0 }).collect())
///     .collect();
/// let weights = vec![1.0 / 32.0; 32];
///
/// // Every margin may move by at most 0.25.
/// let w = sparsify_within(&columns, &weights, 0.25, None, 0);
/// assert!(max_margin_change(&columns, &weights, &w) <= 0.25);
///
/// // A cap of 2 hypotheses takes precedence over the accuracy goal.
/// let w = sparsify_within(&columns, &weights, 0.25, Some(2), 0);
/// assert!(w.iter().filter(|wj| **wj != 0.0).count() <= 2);
/// ```
pub fn sparsify_within(
    columns: &[Vec<f64>],
    weights: &[f64],
    max_change: f64,
    max_nonzeros: Option<usize>,
    seed: u64,
) -> Vec<f64> {
    assert!(
        max_change >= 0.0,
        "the allowed margin change must be nonnegative"
    );
    assert!(max_nonzeros != Some(0), "the nonzero cap must be positive");
    let original = normalized_weights(columns, weights);
    let cap = max_nonzeros.unwrap_or(usize::MAX);
    let mut rng = SplitMix64::new(seed);
    let mut w = original.clone();
    while nonzeros(&w) > 1 {
        let mut next = w.clone();
        halving_pass(columns, &mut next, &mut rng);
        let within = max_margin_change(columns, &original, &next) <= max_change;
        if !within && nonzeros(&w) <= cap {
            break;
        }
        w = next;
        if !within && nonzeros(&w) <= cap {
            // The cap forced this pass; stop as soon as it is met.
            break;
        }
    }
    w
}

/// Validates the inputs and returns `weights / ‖weights‖_1`.
fn normalized_weights(columns: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    assert_eq!(
        columns.len(),
        weights.len(),
        "sparsification needs one weight per column"
    );
    check_columns(columns);
    assert!(
        weights.iter().all(|w| w.is_finite()),
        "sparsification weights must be finite"
    );
    let l1 = weights.iter().map(|w| w.abs()).sum::<f64>();
    assert!(
        l1 > 0.0 && l1.is_finite(),
        "sparsification weights must have a positive, finite l1 norm"
    );
    weights.iter().map(|wj| wj / l1).collect()
}

fn nonzeros(w: &[f64]) -> usize {
    w.iter().filter(|wj| **wj != 0.0).count()
}

/// One pass of the `while` loop of Algorithm 2 (steps 3-14) on `w` with
/// `‖w‖_0 ≥ 2`; keeps at most `‖w‖_0 / 2` nonzero entries.
fn halving_pass(columns: &[Vec<f64>], w: &mut [f64], rng: &mut SplitMix64) {
    let mut support: Vec<usize> = (0..w.len()).filter(|&j| w[j] != 0.0).collect();
    debug_assert!(support.len() >= 2);
    // Steps 3-4: R = the largest third (by |w'_j|, ties by index).
    support.sort_by(|&a, &b| w[b].abs().total_cmp(&w[a].abs()).then(a.cmp(&b)));
    let n_kept = (support.len() / 3).max(1);
    let mut rest = support.split_off(n_kept);
    let omega = w[rest[0]].abs();

    // Steps 5-13: halve the entries outside R twice.
    for _ in 0..2 {
        rest.retain(|&j| w[j] != 0.0);
        if rest.is_empty() {
            break;
        }
        let a: Vec<Vec<f64>> = rest
            .iter()
            .map(|&j| {
                let scale = w[j] / omega;
                let mut col: Vec<f64> = columns[j].iter().map(|u| u * scale).collect();
                col.push(scale.abs());
                col
            })
            .collect();
        let x = coloring(&a, rng);
        let n_plus = x.iter().filter(|&&xj| xj > 0.0).count();
        let sigma = if 2 * n_plus <= x.len() { 1.0 } else { -1.0 };
        for (&j, &xj) in rest.iter().zip(&x) {
            w[j] = if xj == sigma { 2.0 * w[j] } else { 0.0 };
        }
    }

    // Step 14.
    let l1 = w.iter().map(|wj| wj.abs()).sum::<f64>();
    assert!(
        l1 > 0.0 && l1.is_finite(),
        "sparsification produced a non-finite or zero l1 norm"
    );
    w.iter_mut().for_each(|wj| *wj /= l1);
}

/// Returns `‖U w - U w'‖_∞ = max_i |Σ_j u_ij (w_j - w'_j)|`, the largest
/// change of any margin when the weights `before` are replaced by `after`.
///
/// `columns[j][i] = u_ij` as in [`sparsify`]. Both weight vectors are used as
/// given (no normalization). Returns `0.0` when there are no examples.
///
/// Time complexity: `O(n m)`.
///
/// # Panics
/// Panics if `columns`, `before`, and `after` differ in length or the
/// columns differ in length.
///
/// # Examples
/// ```
/// use miniboosts::optimization::sparsification::max_margin_change;
///
/// // Two examples, two hypotheses that disagree on the second example.
/// let columns = vec![vec![1.0, 1.0], vec![1.0, -1.0]];
/// // Moving all weight to the first hypothesis changes margin 2 from 0 to 1.
/// let change = max_margin_change(&columns, &[0.5, 0.5], &[1.0, 0.0]);
/// assert_eq!(change, 1.0);
/// ```
pub fn max_margin_change(columns: &[Vec<f64>], before: &[f64], after: &[f64]) -> f64 {
    assert_eq!(columns.len(), before.len(), "one weight per column");
    assert_eq!(columns.len(), after.len(), "one weight per column");
    let n_rows = columns.first().map_or(0, |c| c.len());
    assert!(
        columns.iter().all(|c| c.len() == n_rows),
        "all columns must have the same length"
    );
    let mut diff = vec![0.0; n_rows];
    for ((col, b), a) in columns.iter().zip(before).zip(after) {
        let delta = b - a;
        if delta != 0.0 {
            diff.iter_mut().zip(col).for_each(|(d, u)| *d += u * delta);
        }
    }
    diff.iter().fold(0.0, |acc, d| acc.max(d.abs()))
}

/// Sparsifies the ensemble `f` to at most `target` hypotheses with
/// [`sparsify`], using the margins of `f`'s hypotheses on `sample`.
///
/// The margin matrix is `u_ij = y_i h_j(x_i)` with `h_j(x_i)` the
/// [`Classifier::confidence`] of hypothesis `j` on example `i`. Section 1.2
/// of the paper notes that the sparsification applies to the output of any
/// boosting algorithm; the minimal-margin guarantee of Theorem 2.1 is stated
/// for SparsiBoost (AdaBoostV followed by this step).
///
/// Returns the sparsified ensemble (hypotheses with zero weight removed,
/// weights normalized to `‖w'‖_1 = 1`) and the measured
/// [`max_margin_change`] on `sample`. An empty ensemble is returned
/// unchanged with change `0.0`.
///
/// Time complexity: `O(n m)` confidence evaluations plus [`sparsify`].
///
/// # Panics
/// Panics as [`sparsify`] does, in particular if a confidence lies outside
/// `[-1, 1]` or the sample has no labels.
///
/// # Examples
/// Shrink a 30-round AdaBoost ensemble to at most 5 hypotheses.
/// ```
/// # use std::io::BufReader;
/// use miniboosts::{Sample, prelude::*};
/// use miniboosts::optimization::sparsification::sparsify_ensemble;
///
/// # let csv = b"x,y,class\n\
/// #     -2,1,-1\n-1,-2,-1\n1,-3,-1\n-3,2,-1\n0,-1,-1\n-2,-1,-1\n\
/// #     2,1,1\n1,2,1\n-1,3,1\n3,-1,1\n0,2,1\n2,2,1\n";
/// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
/// #     .unwrap()
/// #     .set_target("class");
/// let tree = DecisionTreeBuilder::new(&sample).max_depth(1).build();
/// let f = AdaBoost::init(&sample).force_quit_at(30).run(&tree);
///
/// let (g, change) = sparsify_ensemble(&sample, &f, 5, 0);
/// assert!(g.hypotheses.len() <= 5);
/// assert!((g.weights.iter().map(|w| w.abs()).sum::<f64>() - 1.0).abs() < 1e-12);
/// println!("largest margin change: {change}");
/// ```
pub fn sparsify_ensemble<H>(
    sample: &Sample,
    f: &WeightedMajority<H>,
    target: usize,
    seed: u64,
) -> (WeightedMajority<H>, f64)
where
    H: Classifier + Clone,
{
    assert!(target > 0, "sparsification target must be positive");
    if f.hypotheses.is_empty() {
        return (WeightedMajority::from_slices(&[], &f.hypotheses), 0.0);
    }
    let columns = margin_columns(sample, &f.hypotheses);
    let sparse = sparsify(&columns, &f.weights, target, seed);
    let l1 = f.weights.iter().map(|w| w.abs()).sum::<f64>();
    let before: Vec<f64> = f.weights.iter().map(|w| w / l1).collect();
    let change = max_margin_change(&columns, &before, &sparse);
    (
        WeightedMajority::from_slices(&sparse, &f.hypotheses),
        change,
    )
}

/// Returns `columns[j][i] = y_i h_j(x_i)` for the hypotheses on `sample`.
pub(crate) fn margin_columns<H: Classifier>(sample: &Sample, hypotheses: &[H]) -> Vec<Vec<f64>> {
    let targets = sample.target();
    hypotheses
        .iter()
        .map(|h| {
            targets
                .iter()
                .enumerate()
                .map(|(i, y)| y * h.confidence(sample, i))
                .collect()
        })
        .collect()
}

/// Returns a coloring `x ∈ {-1, +1}^k` of the `k` columns of `A` that
/// attempts to make every row sum `(A x)_i = Σ_j a_ij x_j` small, i.e. a
/// low-discrepancy signing of the columns.
///
/// `columns[j][i] = a_ij`. Spencer's theorem (Theorem 2.3 of the SparsiBoost
/// paper) guarantees that some `x` achieves `‖A x‖_∞ = O(√(k lg(2 + n/k)))`
/// when `|a_ij| ≤ 1`; [`sparsify`] uses this to split columns into two halves
/// with nearly equal row sums.
///
/// # Algorithm
/// A partial-coloring walk in the spirit of Lovett and Meka (2015), with
/// jumps instead of small Gaussian steps, followed by local search:
/// 1. Start at `x = 0`. In each phase, with `ℓ` coordinates still
///    fractional (`|x_j| < 1`), block the `⌊ℓ/4⌋` rows with the largest
///    `|(A x)_i|` (ties: larger row norm, then smaller index).
/// 2. Repeat: draw a Gaussian direction `g` on the fractional coordinates,
///    project it orthogonally to the blocked rows and to the coordinates
///    frozen in this phase, flip it if it increases `‖A x‖_2` to first
///    order, and move `x ← x + t g` with the largest `t` keeping
///    `x ∈ [-1, 1]^k`. Coordinates reaching `±1` are frozen, and one more
///    row (the unblocked one with the largest `|(A x)_i|`) is blocked per
///    frozen coordinate. The phase ends when the walk has no free direction.
/// 3. When every coordinate is `±1`, repeatedly flip single coordinates
///    while this strictly lowers `‖A x‖_∞` (at most 100 sweeps).
///
/// Blocked rows do not change during a phase, and each step freezes at
/// least one coordinate, so the walk takes at most `k` steps.
///
/// # Limitations
/// This is a heuristic: it is not the Lovett–Meka algorithm with its
/// parameters, and it does not certify Spencer's bound. If a phase freezes
/// nothing (numerical degeneracy), the remaining coordinates are rounded to
/// their signs before the local search. Time `O(n k² + k³)`, memory
/// `O(n k + k²)`.
///
/// # Panics
/// Panics if the columns differ in length or contain a non-finite entry.
///
/// # Examples
/// Two identical columns can always be balanced exactly.
/// ```
/// use miniboosts::optimization::sparsification::discrepancy_coloring;
///
/// let columns = vec![vec![1.0, -0.5, 0.25], vec![1.0, -0.5, 0.25]];
/// let x = discrepancy_coloring(&columns, 3);
/// assert_eq!(x.len(), 2);
/// assert_eq!(x[0], -x[1]);
/// ```
pub fn discrepancy_coloring(columns: &[Vec<f64>], seed: u64) -> Vec<i8> {
    let n_rows = columns.first().map_or(0, |c| c.len());
    assert!(
        columns.iter().all(|c| c.len() == n_rows),
        "all columns must have the same length"
    );
    assert!(
        columns.iter().flatten().all(|a| a.is_finite()),
        "coloring needs finite entries"
    );
    let mut rng = SplitMix64::new(seed);
    coloring(columns, &mut rng)
        .into_iter()
        .map(|x| if x > 0.0 { 1 } else { -1 })
        .collect()
}

fn check_columns(columns: &[Vec<f64>]) {
    let n_rows = columns.first().map_or(0, |c| c.len());
    assert!(n_rows > 0, "sparsification needs at least one example");
    assert!(
        columns.iter().all(|c| c.len() == n_rows),
        "all columns must have the same length"
    );
    assert!(
        columns.iter().flatten().all(|u| (-1.0..=1.0).contains(u)),
        "margins must lie in [-1, 1]"
    );
}

// Coordinates within this distance of ±1 are frozen and snapped to ±1.
const FREEZE_TOLERANCE: f64 = 1e-9;
// A projected direction shorter than this fraction of the drawn Gaussian is
// treated as zero: the free subspace is numerically empty.
const DEGENERATE_DIRECTION: f64 = 1e-9;
const MAX_LOCAL_SEARCH_SWEEPS: usize = 100;

/// The walk and local search of [`discrepancy_coloring`]; returns `±1.0`.
fn coloring(columns: &[Vec<f64>], rng: &mut SplitMix64) -> Vec<f64> {
    let k = columns.len();
    let n_rows = columns.first().map_or(0, |c| c.len());
    let mut x = vec![0.0; k];
    let mut fractional = vec![true; k];
    let mut ax: Vec<f64> = vec![0.0; n_rows];

    loop {
        let phase: Vec<usize> = (0..k).filter(|&j| fractional[j]).collect();
        let l = phase.len();
        if l == 0 {
            break;
        }
        let row = |i: usize| -> Vec<f64> { phase.iter().map(|&j| columns[j][i]).collect() };
        let mut basis = Basis::default();
        let mut blocked = vec![false; n_rows];

        // Initial blocking: the ⌊ℓ/4⌋ rows most at risk.
        let row_norms: Vec<f64> = (0..n_rows)
            .map(|i| phase.iter().map(|&j| columns[j][i].powi(2)).sum())
            .collect();
        let mut order: Vec<usize> = (0..n_rows).collect();
        order.sort_by(|&a, &b| {
            ax[b]
                .abs()
                .total_cmp(&ax[a].abs())
                .then(row_norms[b].total_cmp(&row_norms[a]))
                .then(a.cmp(&b))
        });
        for &i in order.iter().take(l / 4) {
            blocked[i] = true;
            basis.insert(row(i));
        }

        let mut frozen_in_phase = 0;
        while basis.len() < l {
            let drawn: Vec<f64> = (0..l).map(|_| rng.gaussian()).collect();
            let drawn_norm = norm(&drawn);
            let mut g = drawn;
            basis.project_out(&mut g);
            for (p, &j) in phase.iter().enumerate() {
                if !fractional[j] {
                    g[p] = 0.0;
                }
            }
            if norm(&g) <= DEGENERATE_DIRECTION * drawn_norm {
                break;
            }
            let mut ag: Vec<f64> = (0..n_rows)
                .map(|i| {
                    phase
                        .iter()
                        .zip(&g)
                        .map(|(&j, gp)| columns[j][i] * gp)
                        .sum()
                })
                .collect();
            if dot(&ax, &ag) > 0.0 {
                g.iter_mut().for_each(|gp| *gp = -*gp);
                ag.iter_mut().for_each(|v| *v = -*v);
            }
            // Largest step that keeps every coordinate in [-1, 1].
            let step = phase
                .iter()
                .zip(&g)
                .filter(|&(&j, gp)| fractional[j] && *gp != 0.0)
                .map(|(&j, &gp)| (gp.signum() - x[j]) / gp)
                .fold(f64::INFINITY, f64::min);
            if !step.is_finite() {
                break;
            }
            for (&j, gp) in phase.iter().zip(&g) {
                if fractional[j] {
                    x[j] = (x[j] + step * gp).clamp(-1.0, 1.0);
                }
            }
            ax.iter_mut().zip(&ag).for_each(|(v, d)| *v += step * d);

            for (p, &j) in phase.iter().enumerate() {
                if fractional[j] && x[j].abs() >= 1.0 - FREEZE_TOLERANCE {
                    x[j] = x[j].signum();
                    fractional[j] = false;
                    frozen_in_phase += 1;
                    let mut e = vec![0.0; l];
                    e[p] = 1.0;
                    basis.insert(e);
                    // Block one more row per frozen coordinate.
                    let worst = (0..n_rows)
                        .filter(|&i| !blocked[i])
                        .max_by(|&a, &b| ax[a].abs().total_cmp(&ax[b].abs()).then(b.cmp(&a)));
                    if let Some(i) = worst {
                        if basis.len() < l {
                            blocked[i] = true;
                            basis.insert(row(i));
                        }
                    }
                }
            }
        }

        if frozen_in_phase == 0 {
            // Numerical degeneracy: round what is left and stop walking.
            for j in 0..k {
                if fractional[j] {
                    x[j] = if x[j] >= 0.0 { 1.0 } else { -1.0 };
                    fractional[j] = false;
                }
            }
        }
    }

    // Recompute A x exactly for the integral coloring.
    let mut ax = vec![0.0; n_rows];
    for (col, xj) in columns.iter().zip(&x) {
        ax.iter_mut().zip(col).for_each(|(v, a)| *v += a * xj);
    }
    local_search(columns, &mut x, &mut ax);
    x
}

/// Flips single coordinates while doing so strictly lowers `‖A x‖_∞`.
fn local_search(columns: &[Vec<f64>], x: &mut [f64], ax: &mut [f64]) {
    let mut best = max_abs(ax);
    for _ in 0..MAX_LOCAL_SEARCH_SWEEPS {
        let mut improved = false;
        for (j, col) in columns.iter().enumerate() {
            let flipped = ax
                .iter()
                .zip(col)
                .fold(0.0_f64, |acc, (v, a)| acc.max((v - 2.0 * x[j] * a).abs()));
            // Require a relative improvement so rounding cannot cause cycles.
            if flipped < best * (1.0 - 1e-12) {
                ax.iter_mut()
                    .zip(col)
                    .for_each(|(v, a)| *v -= 2.0 * x[j] * a);
                x[j] = -x[j];
                best = max_abs(ax);
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
}

/// An orthonormal basis built by modified Gram–Schmidt with
/// reorthogonalization.
#[derive(Default)]
struct Basis {
    vectors: Vec<Vec<f64>>,
}

impl Basis {
    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn project_out(&self, v: &mut [f64]) {
        // Two passes ("twice is enough") keep the result orthogonal.
        for _ in 0..2 {
            for q in &self.vectors {
                let c = dot(q, v);
                v.iter_mut().zip(q).for_each(|(vi, qi)| *vi -= c * qi);
            }
        }
    }

    /// Adds the component of `v` orthogonal to the basis, unless it is
    /// numerically zero.
    fn insert(&mut self, mut v: Vec<f64>) {
        let original = norm(&v);
        if original == 0.0 {
            return;
        }
        self.project_out(&mut v);
        let remaining = norm(&v);
        if remaining <= 1e-10 * original {
            return;
        }
        v.iter_mut().for_each(|vi| *vi /= remaining);
        self.vectors.push(v);
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0, |acc, v| acc.max(v.abs()))
}

/// SplitMix64 (Steele, Lea, and Flood, 2014): a small seeded generator, so
/// results are reproducible without an external dependency.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on the open interval `(0, 1)`.
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal by the Box–Muller transform.
    fn gaussian(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed pseudo-random ±1 matrix with `k` columns and `n` rows.
    fn sign_matrix(n: usize, k: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = SplitMix64::new(seed);
        (0..k)
            .map(|_| {
                (0..n)
                    .map(|_| if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 })
                    .collect()
            })
            .collect()
    }

    fn row_sums(columns: &[Vec<f64>], x: &[i8]) -> Vec<f64> {
        let mut ax = vec![0.0; columns[0].len()];
        for (col, &xj) in columns.iter().zip(x) {
            ax.iter_mut()
                .zip(col)
                .for_each(|(v, a)| *v += a * xj as f64);
        }
        ax
    }

    #[test]
    fn splitmix_matches_reference_values() {
        // First outputs for seed 0 from the reference implementation.
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    }

    #[test]
    fn coloring_beats_the_random_signing_bound() {
        // n = 64 rows, k = 48 columns of ±1. A uniformly random signing has
        // E‖Ax‖_∞ ≈ √(2 k ln(2n)) ≈ 30.5; Spencer's bound is
        // O(√(k ln(2 + n/k))) ≈ 8.0 up to its constant.
        let (n, k) = (64, 48);
        for seed in 0..5 {
            let columns = sign_matrix(n, k, seed);
            let x = discrepancy_coloring(&columns, seed);
            assert!(x.iter().all(|&xj| xj == 1 || xj == -1));
            let disc = max_abs(&row_sums(&columns, &x));
            let spencer = (k as f64 * (2.0 + n as f64 / k as f64).ln()).sqrt();
            assert!(disc <= 2.0 * spencer, "seed {seed}: {disc} > 2 * {spencer}");
        }
    }

    #[test]
    fn coloring_is_reproducible_and_handles_edge_cases() {
        let columns = sign_matrix(20, 15, 1);
        assert_eq!(
            discrepancy_coloring(&columns, 9),
            discrepancy_coloring(&columns, 9)
        );
        assert!(discrepancy_coloring(&[], 0).is_empty());
        assert_eq!(discrepancy_coloring(&[vec![0.5]], 0).len(), 1);
        // Zero columns are colored too (any sign is optimal).
        assert_eq!(discrepancy_coloring(&vec![vec![0.0; 3]; 4], 0).len(), 4);
    }

    #[test]
    fn opposite_pairs_are_balanced_exactly() {
        // Columns c, c, d, d: x = (+, -, +, -) gives A x = 0.
        let c = vec![1.0, -1.0, 0.5, 1.0];
        let d = vec![-1.0, -1.0, 1.0, 0.25];
        let columns = vec![c.clone(), c, d.clone(), d];
        let x = discrepancy_coloring(&columns, 0);
        assert_eq!(max_abs(&row_sums(&columns, &x)), 0.0, "{x:?}");
    }

    #[test]
    fn sparsify_meets_the_sparsity_norm_and_support_contract() {
        let columns = sign_matrix(100, 200, 3);
        let weights: Vec<f64> = (0..200).map(|j| 1.0 + (j % 7) as f64).collect();
        for target in [1, 2, 13, 50, 199] {
            let w = sparsify(&columns, &weights, target, 11);
            let nnz = w.iter().filter(|wj| **wj != 0.0).count();
            assert!(1 <= nnz && nnz <= target, "target {target}: {nnz}");
            let l1 = w.iter().map(|wj| wj.abs()).sum::<f64>();
            assert!((l1 - 1.0).abs() < 1e-12);
            assert!(w.iter().all(|wj| *wj >= 0.0));
        }
    }

    #[test]
    fn sparsify_keeps_signs_and_returns_short_inputs_normalized() {
        let columns = sign_matrix(10, 6, 4);
        let weights = [2.0, -1.0, 0.0, 1.0, -4.0, 0.0];
        // Already at most 4 nonzeros: only normalization (‖w‖_1 = 8).
        let w = sparsify(&columns, &weights, 4, 0);
        assert_eq!(w, vec![0.25, -0.125, 0.0, 0.125, -0.5, 0.0]);
        let w = sparsify(&columns, &weights, 2, 0);
        for (wj, vj) in w.iter().zip(weights) {
            assert!(*wj == 0.0 || wj.signum() == vj.signum());
        }
    }

    #[test]
    fn sparsify_error_is_within_a_small_multiple_of_the_theorem_rate() {
        // n = 200 examples, m = 512 hypotheses with uniform weights, T = 32.
        // Theorem 2.2 gives ‖U w - U w'‖_∞ = O(√(lg(2 + n/T) / T)) ≈ 0.31
        // with an unstated constant; over seeds 0..5 the measured ratio is
        // about 1.8. A factor 4 leaves room for other seeds and rounding
        // while still bounding the error well below the trivial value 2.
        let (n, m, target) = (200, 512, 32);
        let columns = sign_matrix(n, m, 5);
        let weights = vec![1.0 / m as f64; m];
        let rate = ((2.0 + n as f64 / target as f64).log2() / target as f64).sqrt();
        let w = sparsify(&columns, &weights, target, 1);
        let change = max_margin_change(&columns, &weights, &w);
        assert!(change <= 4.0 * rate, "{change} > 4 * {rate}");
    }

    #[test]
    fn one_halving_pass_keeps_the_top_third_and_a_quarter_of_the_rest() {
        // m = 12 distinct weights, target 6: one pass suffices. R holds the
        // 4 largest entries, copied unchanged before normalization; the other
        // 8 are halved twice, so at most 2 of them survive, each multiplied
        // by 4.
        let columns = sign_matrix(30, 12, 6);
        let weights: Vec<f64> = (1..=12).map(|j| j as f64).collect();
        let w = sparsify(&columns, &weights, 6, 2);
        let top: f64 = (9..=12).map(|j| j as f64).sum();
        let survivors: Vec<usize> = (0..8).filter(|&j| w[j] != 0.0).collect();
        assert!(survivors.len() <= 2, "{survivors:?}");
        let z = top + survivors.iter().map(|&j| 4.0 * weights[j]).sum::<f64>();
        for j in 0..12 {
            let expected = if j >= 8 {
                weights[j] / z
            } else if survivors.contains(&j) {
                4.0 * weights[j] / z
            } else {
                0.0
            };
            assert!((w[j] - expected).abs() < 1e-15, "entry {j}");
        }
    }

    #[test]
    fn sparsify_within_keeps_the_budget_without_a_cap() {
        let columns = sign_matrix(40, 96, 8);
        let weights = vec![1.0; 96];
        let uniform = vec![1.0 / 96.0; 96];
        let mut previous_nnz = usize::MAX;
        for budget in [0.05, 0.1, 0.2, 0.4] {
            let w = sparsify_within(&columns, &weights, budget, None, 4);
            assert!(max_margin_change(&columns, &uniform, &w) <= budget);
            // The same seed walks the same halving path, so a larger budget
            // stops at the same point or later.
            let nnz = nonzeros(&w);
            assert!(
                nnz <= previous_nnz,
                "budget {budget}: {nnz} > {previous_nnz}"
            );
            previous_nnz = nnz;
        }
        // An unlimited budget halves down to a single hypothesis, and a zero
        // budget keeps the normalized input (each pass moves some margin).
        assert_eq!(
            nonzeros(&sparsify_within(&columns, &weights, f64::INFINITY, None, 4)),
            1
        );
        assert_eq!(sparsify_within(&columns, &weights, 0.0, None, 4), uniform);
    }

    #[test]
    fn sparsify_within_lets_the_cap_take_precedence() {
        let columns = sign_matrix(40, 96, 8);
        let weights = vec![1.0; 96];
        let uniform = vec![1.0 / 96.0; 96];
        // A zero budget alone keeps all 96 entries; the cap forces halving.
        let w = sparsify_within(&columns, &weights, 0.0, Some(10), 4);
        assert!(nonzeros(&w) <= 10);
        assert!(max_margin_change(&columns, &uniform, &w) > 0.0);
        // A cap that the budget already meets changes nothing.
        let loose = sparsify_within(&columns, &weights, 0.2, None, 4);
        let capped = sparsify_within(&columns, &weights, 0.2, Some(96), 4);
        assert_eq!(loose, capped);
    }

    #[test]
    fn sparsify_within_rejects_invalid_budgets() {
        let columns = sign_matrix(5, 4, 0);
        let weights = [1.0; 4];
        for (budget, cap) in [(-0.1, None), (f64::NAN, None), (0.1, Some(0))] {
            let result =
                std::panic::catch_unwind(|| sparsify_within(&columns, &weights, budget, cap, 0));
            assert!(result.is_err(), "{budget} {cap:?}");
        }
    }

    #[test]
    fn max_margin_change_is_the_infinity_norm_of_the_difference() {
        let columns = vec![vec![1.0, -1.0, 0.5], vec![-1.0, 1.0, 1.0]];
        // U (w - w') with w - w' = (0.5, -0.5): rows (1, -1, -0.25).
        let change = max_margin_change(&columns, &[0.5, 0.5], &[0.0, 1.0]);
        assert_eq!(change, 1.0);
        assert_eq!(max_margin_change(&[], &[], &[]), 0.0);
    }

    #[test]
    fn invalid_inputs_panic() {
        let good = vec![vec![1.0, -1.0]; 3];
        let out_of_range = vec![vec![1.5, -1.0]; 3];
        let ragged = vec![vec![1.0], vec![1.0, -1.0]];
        // (columns, weights, target)
        type Case<'a> = (&'a [Vec<f64>], &'a [f64], usize);
        let cases: [Case; 6] = [
            (&good, &[1.0; 3], 0),
            (&good, &[1.0; 2], 1),
            (&out_of_range, &[1.0; 3], 1),
            (&good, &[0.0; 3], 1),
            (&good, &[f64::NAN, 1.0, 1.0], 1),
            (&ragged, &[1.0; 2], 1),
        ];
        for (case, (columns, weights, target)) in cases.into_iter().enumerate() {
            let result = std::panic::catch_unwind(|| sparsify(columns, weights, target, 0));
            assert!(result.is_err(), "case {case}");
        }
    }
}
