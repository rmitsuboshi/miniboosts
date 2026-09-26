//! Provides some helper functions.
use rayon::prelude::*;

use crate::checkers;
use crate::constants::BINARY_SEARCH_TOLERANCE;
use crate::{Classifier, Sample};

/// Returns the edge of a single hypothesis for the given distribution.
/// Here `edge` is the weighted training loss.
///
/// Time complexity: `O(m)`, where `m` is the number of training examples.
#[inline(always)]
pub fn edge<H>(sample: &Sample, dist: &[f64], h: &H) -> f64
where
    H: Classifier,
{
    margins(sample, h)
        .zip(dist)
        .map(|(yh, d)| *d * yh)
        .sum::<f64>()
}

pub fn edge_from_margins(margins: &[f64], dist: &[f64]) -> f64 {
    margins.iter().zip(dist).map(|(yh, d)| *d * yh).sum::<f64>()
}

/// Returns the margin vector of a single hypothesis
/// for the given distribution.
///
/// Time complexity: `O(m)`, where `m` is the number of training examples.
#[inline(always)]
pub fn margins<H>(sample: &Sample, h: &H) -> impl Iterator<Item = f64>
where
    H: Classifier,
{
    let targets = sample.target();

    targets
        .iter()
        .enumerate()
        .map(|(i, y)| y * h.confidence(sample, i))
}

/// Computes the logarithm of
/// the exponential distribution for the given combined hypothesis.
/// The `i` th element of the output vector `d` satisfies:
/// ```txt
/// d[i] = - eta * yi * sum ( w[h] * h(xi) ),
/// ```
/// where `(xi, yi)` is the `i`-th training example.
///
/// Time complexity: `O(m * n)`, where
/// - `m` is the number of training examples and
/// - `n` is the number of hypotheses.
#[inline(always)]
pub fn log_exp_distribution<H>(eta: f64, sample: &Sample, h: &H) -> impl Iterator<Item = f64>
where
    H: Classifier,
{
    margins(sample, h).map(move |yhx| -eta * yhx)
}

/// Computes the exponential distribution for the given combined hypothesis.
/// The `i` th element of the output vector `d` satisfies:
/// ```txt
/// d[i] ∝ exp( - eta * yi * sum ( w[h] * h(xi) ) ),
/// ```
/// where `(xi, yi)` is the `i`-th training example.
///
/// Time complexity: `O(m * n)`, where
/// - `m` is the number of training examples and
/// - `n` is the number of hypotheses.
#[inline(always)]
pub fn exp_distribution<H>(eta: f64, nu: f64, sample: &Sample, h: &H) -> Vec<f64>
where
    H: Classifier,
{
    let log_dist = log_exp_distribution(eta, sample, h);

    project_log_distribution_to_capped_simplex(nu, log_dist)
}

/// This function computes the distribution for
/// Deformed Corrective `t`-ErlpBoost algorithm.
#[inline(always)]
pub fn deformed_exp_distribution<H>(
    deform: f64,
    eta: f64,
    nu: f64,
    sample: &Sample,
    f: &H,
) -> Vec<f64>
where
    H: Classifier,
{
    let q = gradient_of_conjugate_deformed_entropy(deform, eta, sample, f);

    deformed_projection_onto_capped_simplex(nu, deform, q)
}

/// Computes the exponential distribution from the given parameters
/// `eta` and `nu` and an iterator `margins`.
///
/// Computational complexity: `O(m log(m))`,
/// where `m` is the number of training examples.
#[inline(always)]
pub fn deformed_exp_distribution_from_margins<I>(
    deform: f64,
    eta: f64,
    nu: f64,
    margins: I,
) -> Vec<f64>
where
    I: Iterator<Item = f64>,
{
    let power = 1f64 / (1f64 - deform);
    let mut g = margins
        .map(|yf| -(1f64 - deform) * eta * yf)
        .collect::<Vec<f64>>();

    let max = g.iter().fold(f64::MIN, |acc, val| val.max(acc));
    let (mut lb, mut ub) = (-max, 1f64 - max);

    while ub - lb > BINARY_SEARCH_TOLERANCE {
        let normalizer = (ub + lb) / 2f64;
        let sum = g
            .iter()
            .map(|val| (val + normalizer).max(0f64).powf(power))
            .sum::<f64>();

        assert!(sum.is_finite());
        if sum < 1f64 {
            lb = normalizer;
        } else if sum > 1f64 {
            ub = normalizer;
        }
    }

    let normalizer = (lb + ub) / 2f64;
    g.iter_mut().for_each(|val| {
        *val = (*val + normalizer).max(0f64).powf(power);
    });

    assert!(
        g.iter().all(|gi| gi.is_finite()),
        "invalid value in gradient. g = {g:?}"
    );
    checkers::capped_simplex_condition(&g[..], 1f64);

    deformed_projection_onto_capped_simplex(nu, deform, g)
}

#[inline(always)]
fn deformed_projection_onto_capped_simplex(nu: f64, t: f64, mut q: Vec<f64>) -> Vec<f64> {
    assert!(q.iter().all(|qi| qi.is_finite()), "{q:?}");
    fn d(t: f64, q: f64, xi: f64) -> f64 {
        assert!((0f64..=1f64).contains(&t), "t = {t}");
        assert!((0f64..=1f64).contains(&q), "q = {q}");

        let ret = (q.powf(1.0 - t) + (1.0 - t) * xi)
            .max(0f64)
            .powf(1.0 / (1.0 - t));
        assert!(
            ret.is_finite(),
            "d = {ret}, q = {q}, q^(1-t) = {}",
            q.powf(1f64 - t)
        );
        ret
    }

    fn compute_xi(mut lb: f64, amount: f64, t: f64, ix: &[usize], q: &[f64]) -> f64 {
        // DEBUG
        let mut ub = 1f64;
        loop {
            let sum = ix.iter().map(|&i| d(t, q[i], ub)).sum::<f64>();
            if sum >= amount {
                break;
            }
            ub *= 2f64;
        }
        while ub - lb > 0f64 {
            let xi = (lb + ub) / 2f64;
            let sum = ix.iter().map(|&i| d(t, q[i], xi)).sum::<f64>();
            if sum < amount {
                lb = xi;
            } else {
                ub = xi;
            }
            if (sum - amount).abs() < BINARY_SEARCH_TOLERANCE {
                break;
            }
        }
        (lb + ub) / 2f64
    }

    let n_sample = q.len();

    // Construct a vector of indices `ix.`
    let ix = {
        let mut ix = (0..n_sample).collect::<Vec<usize>>();
        // sort `ix` in the descending order of `q`.
        ix.sort_by(|&i, &j| q[j].partial_cmp(&q[i]).unwrap());
        ix
    };

    let lb = -1f64 / (1f64 - t);
    for i in 0..n_sample {
        let amount = 1f64 - (i as f64 / nu);
        let xi = compute_xi(lb, amount, t, &ix[i..], &q[..]);

        if d(t, q[ix[i]], xi) < 1f64 / nu {
            for k in i..n_sample {
                q[ix[k]] = d(t, q[ix[k]], xi);
            }
            break;
        }
        q[ix[i]] = 1f64 / nu;
    }

    checkers::capped_simplex_condition(&q, nu);
    q
}

/// This function computes the vector `q` for the given point `θ`
/// ```txt
///     q := arg max { Σ_{k=1}^{m} q_k θ_k - (1/η) q_k ln_t (q_k) | q ∈ Δ_m },
/// ```
/// where `ln_t (x) = ( x^{1-t} - 1 ) / (1 - t)` is the deformed logarithm.
/// The vecotr `q` can be written in the following form:
/// ```txt
/// q_i = (1 / (2-t))^{1/(1-t)} [(1 - t) η θ + ξ]_+^{1/(1-t)},
/// ```
/// where `[x]_+ = max{ 0, x }` and `ξ` is the normalization factor.
/// This function computes `ξ` by binary search.
pub fn gradient_of_conjugate_deformed_entropy<H>(
    deform: f64,
    eta: f64,
    sample: &Sample,
    f: &H,
) -> Vec<f64>
where
    H: Classifier,
{
    let power = 1f64 / (1f64 - deform);
    // (1-t) η θ
    let mut g = margins(sample, f)
        .map(|yf| -(1f64 - deform) * eta * yf)
        .collect::<Vec<f64>>();

    let max = g.iter().fold(f64::MIN, |acc, val| val.max(acc));
    let (mut lb, mut ub) = (-max, 1f64 - max);

    while ub - lb > BINARY_SEARCH_TOLERANCE {
        let normalizer = (ub + lb) / 2f64;
        let sum = g
            .iter()
            .map(|val| (val + normalizer).max(0f64).powf(power))
            .sum::<f64>();

        assert!(sum.is_finite());
        if sum < 1f64 {
            lb = normalizer;
        } else if sum > 1f64 {
            ub = normalizer;
        }
    }

    let normalizer = (lb + ub) / 2f64;
    g.iter_mut().for_each(|val| {
        *val = (*val + normalizer).max(0f64).powf(power);
    });

    assert!(
        g.iter().all(|gi| gi.is_finite()),
        "invalid value in gradient. g = {g:?}"
    );
    checkers::capped_simplex_condition(&g[..], 1f64);
    g
}

/// Computes the exponential distribution from the given parameters
/// `eta` and `nu` and an iterator `margins`.
///
/// Computational complexity: `O(m log(m))`,
/// where `m` is the number of training examples.
#[inline(always)]
pub fn exp_distribution_from_margins<I>(eta: f64, nu: f64, margins: I) -> Vec<f64>
where
    I: Iterator<Item = f64>,
{
    let iter = margins.map(|yf| -eta * yf);
    project_log_distribution_to_capped_simplex(nu, iter)
}

/// Add two log-domain nonnegative quantities. Negative infinity is zero.
/// Panics for NaN or positive infinity.
pub fn logaddexp(a: f64, b: f64) -> f64 {
    assert!(
        !a.is_nan() && a != f64::INFINITY && !b.is_nan() && b != f64::INFINITY,
        "invalid log weight"
    );
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    a.max(b) + (a.min(b) - a.max(b)).exp().ln_1p()
}

fn log_weight_max(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "log weights must be nonempty");
    assert!(
        values.iter().all(|v| !v.is_nan() && *v != f64::INFINITY),
        "log weights must be finite or negative infinity"
    );
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(max.is_finite(), "log weights must have positive support");
    max
}

// Neumaier summation; absolute-value comparison also supports signed margins.
fn compensated_sum(values: impl Iterator<Item = f64>) -> f64 {
    let (mut sum, mut correction) = (0.0_f64, 0.0_f64);
    for value in values {
        let next = sum + value;
        correction += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    sum + correction
}

/// Normalize log weights in O(n), using max-shifted exponentials and direct
/// division (equation (4) in Blanchard, Higham & Higham,
/// <https://arxiv.org/abs/1909.03469>). The sum uses compensated accumulation.
///
/// Panics for empty input, NaN, positive infinity, or no finite entries.
/// Negative infinity denotes exact zero; finite tiny probabilities may underflow.
pub fn softmax(log_weights: &[f64]) -> Vec<f64> {
    let max = log_weight_max(log_weights);
    let mut probabilities: Vec<_> = log_weights.iter().map(|v| (v - max).exp()).collect();
    let sum = compensated_sum(probabilities.iter().copied());
    probabilities.iter_mut().for_each(|p| *p /= sum);
    probabilities
}

/// Return normalized probabilities and retain normalized logs in `log_weights`.
/// Uses the same contract and shifted computation as [`softmax`], without
/// taking logarithms of rounded probabilities. Finite underflowed probabilities
/// thus retain their log mass for later updates. Panics if a finite normalized
/// log cannot be represented as a finite `f64`.
pub fn normalize_log_weights(log_weights: &mut [f64]) -> Vec<f64> {
    let max = log_weight_max(log_weights);
    let mut probabilities: Vec<_> = log_weights.iter().map(|v| (v - max).exp()).collect();
    let sum = compensated_sum(probabilities.iter().copied());
    let log_sum = sum.ln();
    for (log_weight, probability) in log_weights.iter_mut().zip(&mut probabilities) {
        let was_finite = log_weight.is_finite();
        let normalized = (*log_weight - max) - log_sum;
        assert!(
            !was_finite || normalized.is_finite(),
            "normalized log weight exceeds the finite f64 range"
        );
        *log_weight = normalized;
        *probability /= sum;
    }
    probabilities
}

/// Compute `atanh(edge)` from log weights and margins without mistaking
/// rounded probabilities for perfect classification. Weights need not be
/// normalized. Margins must be finite and in `[-1, 1]`, with matching lengths;
/// log weights follow [`softmax`]'s contract.
///
/// Uses compensated summation and `ln_1p` near zero. Near either endpoint,
/// computes half the log ratio of weighted `(1 + margin)` and `(1 - margin)`
/// sums, avoiding cancellation in `1 - edge`. Returns positive/negative
/// infinity only when every supported margin is exactly positive/negative one.
/// Panics if a finite log-weight spread is not representable in `f64`.
pub fn log_weighted_edge_coefficient(log_weights: &[f64], margins: &[f64]) -> f64 {
    let max = log_weight_max(log_weights);
    assert_eq!(
        log_weights.len(),
        margins.len(),
        "weights and margins must match"
    );
    assert!(
        margins
            .iter()
            .all(|m| m.is_finite() && (-1.0..=1.0).contains(m)),
        "margins must be finite and lie in [-1, 1]"
    );
    let shifted: Vec<_> = log_weights
        .iter()
        .map(|&log| {
            let value = log - max;
            assert!(
                !log.is_finite() || value.is_finite(),
                "log-weight spread exceeds f64 range"
            );
            value
        })
        .collect();
    let total = compensated_sum(shifted.iter().map(|x| x.exp()));
    let edge = compensated_sum(shifted.iter().zip(margins).map(|(x, m)| x.exp() * m)) / total;
    if edge.abs() < 0.5 {
        return 0.5 * (edge.ln_1p() - (-edge).ln_1p());
    }
    let (mut plus, mut minus) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for (&log, &margin) in shifted.iter().zip(margins) {
        if log == f64::NEG_INFINITY {
            continue;
        }
        if margin != -1.0 {
            plus = logaddexp(plus, log + margin.ln_1p());
        }
        if margin != 1.0 {
            minus = logaddexp(minus, log + (-margin).ln_1p());
        }
    }
    0.5 * plus - 0.5 * minus
}

/// KL-project log weights onto `{d: 0 <= d[i] <= 1/nu, sum(d) = 1}`.
/// Uses the sorted active-set method of Shalev-Shwartz and Singer (2010),
/// "On the equivalence of weak learnability and linear separability".
/// Sorting is required here; complexity is O(n log n).
///
/// Panics for invalid log weights (see [`softmax`]), non-finite `nu`, or
/// `nu` outside `1..=support_size`. Negative-infinity entries stay zero, so
/// fewer than `nu` finite entries make the constrained support infeasible.
pub fn project_log_distribution_to_capped_simplex<I>(nu: f64, iter: I) -> Vec<f64>
where
    I: Iterator<Item = f64>,
{
    let logs: Vec<_> = iter.collect();
    log_weight_max(&logs);
    let support = logs.iter().filter(|v| v.is_finite()).count();
    assert!(
        nu.is_finite() && nu >= 1.0 && nu <= support as f64,
        "nu must lie between one and the finite support size"
    );
    if nu == 1.0 {
        return softmax(&logs);
    }
    let cap = 1.0 / nu;
    let mut dist = vec![0.0; logs.len()];
    if nu == support as f64 {
        for (p, log) in dist.iter_mut().zip(&logs) {
            if log.is_finite() {
                *p = cap;
            }
        }
        return dist;
    }
    let mut ix: Vec<_> = (0..logs.len()).filter(|&i| logs[i].is_finite()).collect();
    ix.sort_by(|&i, &j| logs[j].partial_cmp(&logs[i]).unwrap());

    // Store each suffix log-sum relative to its own largest entry. Adding a
    // small log-normalizer to an absolute log weight (e.g. 1e16) loses it.
    let mut suffix = vec![0.0; support];
    for i in (0..support - 1).rev() {
        suffix[i] = logaddexp(0.0, (logs[ix[i + 1]] - logs[ix[i]]) + suffix[i + 1]);
    }
    for i in 0..support {
        let remaining = (nu - i as f64) / nu;
        if remaining * (-suffix[i]).exp() <= cap {
            let max = logs[ix[i]];
            let sum = compensated_sum(ix[i..].iter().map(|&j| (logs[j] - max).exp()));
            for &j in &ix[i..] {
                dist[j] = remaining * ((logs[j] - max).exp() / sum);
            }
            break;
        }
        dist[ix[i]] = cap;
    }
    checkers::capped_simplex_condition(&dist, nu);
    dist
}

/// Compute the relative entropy from the uniform distribution.
#[inline(always)]
pub fn entropy_from_uni_distribution<T: AsRef<[f64]>>(dist: T) -> f64 {
    let dist = dist.as_ref();
    let n_dim = dist.len() as f64;
    let e = entropy(dist);

    e + n_dim.ln()
}

/// Compute the entropy of the given distribution.
#[inline(always)]
pub fn entropy<T: AsRef<[f64]>>(dist: T) -> f64 {
    let dist = dist.as_ref();
    dist.iter()
        .copied()
        .map(|d| if d == 0.0 { 0.0 } else { d * d.ln() })
        .sum::<f64>()
}

/// Compute the inner-product of the given two slices.
#[inline(always)]
pub fn inner_product(v1: &[f64], v2: &[f64]) -> f64 {
    v1.into_par_iter().zip(v2).map(|(a, b)| a * b).sum::<f64>()
}

/// Normalizes the given slice.
#[inline(always)]
pub fn normalize(items: &mut [f64]) {
    let z = items.iter().map(|it| it.abs()).sum::<f64>();

    assert_ne!(z, 0.0, "{items:?}");

    items.par_iter_mut().for_each(|item| {
        *item /= z;
    });
}

/// Computes the Hadamard product of given two matrices.
#[inline(always)]
pub fn hadamard_product(mut m1: Vec<Vec<f64>>, m2: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    assert_eq!(m1.len(), m2.len());
    assert_eq!(m1[0].len(), m2[0].len());

    m1.iter_mut().zip(m2).for_each(|(r1, r2)| {
        r1.iter_mut().zip(r2).for_each(|(a, b)| {
            *a *= b;
        });
    });
    m1
}

pub fn format_unit(value: f64) -> String {
    if value < 1_000f64 {
        return format!("{value}");
    }
    let k = value / 1_000f64;
    if k < 1_000f64 {
        return format!("{k:>.1}K");
    }
    let m = k / 1_000f64;
    if m < 1_000f64 {
        return format!("{m:>.1}M");
    }
    let g = m / 1_000f64;
    format!("{g:>.1}G")
}

/// Computes the deformed logarithm:
/// ```txt
/// ln_t(x) = (x^{1-t} - 1) / (1-t)
/// ```
pub fn deformed_logarithm(t: f64, x: f64) -> f64 {
    checkers::deformation_parameter(t);
    if t == 1f64 {
        x.ln()
    } else {
        // For numerical stability, we use `max(0, x).`
        (x.max(0f64).powf(1f64 - t) - 1f64) / (1f64 - t)
    }
}

/// Computes the deformed exponential:
/// ```txt
/// exp_t(x) = max(0, 1 + (1-t) x)^{1/(1-t)}
/// ```
pub fn deformed_exponential(t: f64, x: f64) -> f64 {
    assert!((0f64..1f64).contains(&t));
    (1f64 + (1f64 - t) * x).max(0f64).powf(1f64 / (1f64 - t))
}

/// Compute the deformed t-entropy
#[inline(always)]
pub fn deformed_entropy<T: AsRef<[f64]>>(t: f64, dist: T) -> f64 {
    dist.as_ref()
        .iter()
        .copied()
        .map(|d| d * deformed_logarithm(t, d))
        .sum::<f64>()
}

/// Returns an index whose entry is the minimal value.
pub fn argmin(arr: &[f64]) -> usize {
    let dim = arr.len();
    let (ix, _) =
        arr.iter().enumerate().fold(
            (dim, f64::MAX),
            |acc, (i, &a)| {
                if acc.1 < a { acc } else { (i, a) }
            },
        );
    assert_ne!(ix, dim, "failed to execute argmin. array is {arr:?}");
    ix
}

/// Returns an index whose entry is the maximal value.
pub fn argmax(arr: &[f64]) -> usize {
    let dim = arr.len();
    let (ix, _) =
        arr.iter().enumerate().fold(
            (dim, f64::MIN),
            |acc, (i, &a)| {
                if acc.1 < a { (i, a) } else { acc }
            },
        );
    assert_ne!(ix, dim, "failed to execute argmax. array is {arr:?}");
    ix
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_distribution(actual: &[f64], expected: &[f64], cap: f64) {
        assert_eq!(actual.len(), expected.len());
        // A few ulps cover exp/division rounding in these small fixtures.
        assert!((actual.iter().sum::<f64>() - 1.0).abs() < 2e-15);
        for (&a, &e) in actual.iter().zip(expected) {
            assert!(a.is_finite() && a >= 0.0 && a <= cap + 2e-15);
            assert!((a - e).abs() < 2e-15, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn softmax_handles_offsets_zeros_and_underflow() {
        assert_eq!(softmax(&[1e16, 1e16]), vec![0.5, 0.5]);
        let mut equal_logs = [1e16, 1e16];
        assert_eq!(normalize_log_weights(&mut equal_logs), vec![0.5, 0.5]);
        assert_eq!(equal_logs, [-2.0_f64.ln(); 2]);
        assert_eq!(softmax(&[-10000.0, -10000.0]), vec![0.5, 0.5]);
        assert_eq!(softmax(&[f64::NEG_INFINITY, 0.0]), vec![0.0, 1.0]);
        assert_eq!(softmax(&[f64::MAX, f64::MIN]), vec![1.0, 0.0]);
        let expected = softmax(&[0.0, -2.0, -4.0]);
        assert_distribution(&softmax(&[1e16, 1e16 - 2.0, 1e16 - 4.0]), &expected, 1.0);
        let mut logs = [0.0, -1000.0, f64::NEG_INFINITY];
        assert_eq!(normalize_log_weights(&mut logs), vec![1.0, 0.0, 0.0]);
        assert_eq!(logs, [0.0, -1000.0, f64::NEG_INFINITY]);
        // A later update can revive a rounded-to-zero but finite log mass.
        logs[1] += 1000.0;
        assert_eq!(normalize_log_weights(&mut logs), vec![0.5, 0.5, 0.0]);
        assert_eq!(logs[0], -2.0_f64.ln());
        assert_eq!(logs[1], logs[0]);
    }

    #[test]
    fn compensated_softmax_keeps_small_collective_mass() {
        let mut logs = vec![(1e-16_f64).ln(); 10_000];
        logs[0] = 0.0;
        let p = softmax(&logs);
        let expected = 1.0 / (1.0 + 9999.0 * 1e-16);
        assert!((p[0] - expected).abs() <= f64::EPSILON);
    }

    #[test]
    fn logaddexp_preserves_small_terms_and_zero_identity() {
        assert_eq!(
            logaddexp(f64::NEG_INFINITY, f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );
        assert_eq!(logaddexp(-1000.0, f64::NEG_INFINITY), -1000.0);
        assert_eq!(logaddexp(0.0, 0.0), 2.0_f64.ln());
        assert!(logaddexp(0.0, -40.0) > 0.0);
    }

    #[test]
    fn log_edge_coefficient_distinguishes_underflow_from_perfection() {
        assert_eq!(
            log_weighted_edge_coefficient(&[0.0, -1000.0], &[1.0, -1.0]),
            500.0
        );
        assert_eq!(
            log_weighted_edge_coefficient(&[0.0, -1000.0], &[-1.0, 1.0]),
            -500.0
        );
        assert_eq!(
            log_weighted_edge_coefficient(&[0.0, f64::NEG_INFINITY], &[1.0, -1.0]),
            f64::INFINITY
        );
        assert_eq!(
            log_weighted_edge_coefficient(&[0.0, -1000.0], &[-1.0, -1.0]),
            f64::NEG_INFINITY
        );
        for edge in [1e-18, -1e-18, 0.2, -0.2] {
            let actual = log_weighted_edge_coefficient(&[1e16, 1e16], &[edge, edge]);
            let expected = edge.atanh();
            assert!((actual - expected).abs() <= 2.0 * f64::EPSILON * expected.abs());
        }
        assert_eq!(
            log_weighted_edge_coefficient(&[0.0, 0.0], &[1.0, -1.0]),
            0.0
        );
        assert!(
            std::panic::catch_unwind(|| log_weighted_edge_coefficient(&[0.0], &[f64::NAN]))
                .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| log_weighted_edge_coefficient(&[0.0], &[1.1])).is_err()
        );
        assert!(
            std::panic::catch_unwind(|| log_weighted_edge_coefficient(&[f64::INFINITY], &[1.0]))
                .is_err()
        );
        assert!(std::panic::catch_unwind(|| log_weighted_edge_coefficient(&[0.0], &[])).is_err());
    }

    #[test]
    fn capped_projection_preserves_support_and_shift_invariance() {
        assert_distribution(
            &project_log_distribution_to_capped_simplex(
                2.0,
                [1000.0, 0.0, 0.0, f64::NEG_INFINITY].into_iter(),
            ),
            &[0.5, 0.25, 0.25, 0.0],
            0.5,
        );
        assert_distribution(
            &project_log_distribution_to_capped_simplex(3.0, [1000.0, 0.0, -1000.0].into_iter()),
            &[1.0 / 3.0; 3],
            1.0 / 3.0,
        );
        assert_distribution(
            &project_log_distribution_to_capped_simplex(
                2.0,
                [0.0, f64::NEG_INFINITY, 0.0].into_iter(),
            ),
            &[0.5, 0.0, 0.5],
            0.5,
        );
        assert_distribution(
            &project_log_distribution_to_capped_simplex(
                2.0,
                [f64::MAX, f64::MIN, f64::MIN].into_iter(),
            ),
            &[0.5, 0.25, 0.25],
            0.5,
        );
        for nu in [1.0, 1.5, 2.0, 2.9999999999999996, 3.0] {
            let base =
                project_log_distribution_to_capped_simplex(nu, [0.0, -2.0, -4.0].into_iter());
            let shifted = project_log_distribution_to_capped_simplex(
                nu,
                [1e16, 1e16 - 2.0, 1e16 - 4.0].into_iter(),
            );
            assert_distribution(&shifted, &base, 1.0 / nu);
        }
        let p = project_log_distribution_to_capped_simplex(2.0, [1e16; 3].into_iter());
        assert_distribution(&p, &[1.0 / 3.0; 3], 0.5);
    }

    #[test]
    fn normalization_rejects_invalid_inputs_and_infeasible_support() {
        for values in [
            vec![],
            vec![f64::NAN],
            vec![f64::INFINITY],
            vec![f64::NEG_INFINITY; 2],
        ] {
            assert!(std::panic::catch_unwind(|| softmax(&values)).is_err());
            assert!(
                std::panic::catch_unwind(|| normalize_log_weights(&mut values.clone())).is_err()
            );
            assert!(
                std::panic::catch_unwind(|| project_log_distribution_to_capped_simplex(
                    1.0,
                    values.iter().copied()
                ))
                .is_err()
            );
        }
        for nu in [0.0, 3.0, f64::NAN, f64::INFINITY] {
            assert!(
                std::panic::catch_unwind(|| project_log_distribution_to_capped_simplex(
                    nu,
                    [0.0, 0.0, f64::NEG_INFINITY].into_iter()
                ))
                .is_err()
            );
        }
    }

    struct TestHypothesis {
        threshold: f64,
    }

    impl TestHypothesis {
        fn new(threshold: f64) -> Self {
            Self { threshold }
        }
    }

    impl Classifier for TestHypothesis {
        fn confidence(&self, sample: &Sample, row: usize) -> f64 {
            let value = sample["test"][row];
            if value < self.threshold { -1f64 } else { 1f64 }
        }
    }

    fn training_examples(bytes: &[u8]) -> Sample {
        use std::io::BufReader;
        let reader = BufReader::new(bytes);
        Sample::from_reader(reader, true)
            .unwrap()
            .set_target("class")
    }

    fn training_examples_case_01() -> Sample {
        let bytes = b"\
            test,dummy,class\n\
            0.1,0.2,1.0\n\
            -8.0,2.0,-1.0\n\
            3.0,-9.0,1.0\n\
            -0.001,0.0,-1.0";
        training_examples(&bytes[..])
    }

    fn training_examples_case_02() -> Sample {
        let bytes = b"\
            test,dummy,class\n\
            0.1,0.2,-1.0\n\
            -8.0,2.0,-1.0\n\
            3.0,-9.0,-1.0\n\
            -0.001,0.0,-1.0";
        training_examples(&bytes[..])
    }

    #[test]
    fn test_edge_01() {
        let sample = training_examples_case_01();
        let h = TestHypothesis::new(0f64);
        let dist = vec![1f64 / 4f64; 4];
        let edge = edge(&sample, &dist[..], &h);
        assert!(edge == 1f64, "expected `edge == 1`, got {edge}");
    }

    #[test]
    fn test_edge_02() {
        let sample = training_examples_case_01();
        let h = TestHypothesis::new(100f64);
        let dist = vec![1.0, 0.0, 0.0, 0.0];
        let edge = edge(&sample, &dist[..], &h);
        assert!(edge == -1f64, "expected `edge == -1`, got {edge}");
    }

    #[test]
    fn test_margins_01() {
        let sample = training_examples_case_01();
        let h = TestHypothesis::new(0f64);
        let margins = margins(&sample, &h);
        let m = sample.shape().0;
        let expected = vec![1f64; m];
        for (i, (e, yh)) in expected.into_iter().zip(margins).enumerate() {
            assert_eq!(e, yh, "failed for {i}th example. expected {e}, got {yh}.");
        }
    }

    #[test]
    fn test_margins_02() {
        let sample = training_examples_case_02();
        let h = TestHypothesis::new(0f64);
        let margins = margins(&sample, &h);
        let expected = [-1.0, 1.0, -1.0, 1.0];
        for (i, (e, yh)) in expected.into_iter().zip(margins).enumerate() {
            assert_eq!(e, yh, "failed for {i}th example. expected {e}, got {yh}.");
        }
    }
}
