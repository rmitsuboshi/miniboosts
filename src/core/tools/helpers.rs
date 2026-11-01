//! Numerical helpers shared by boosters and solvers.
//!
//! Notation: `m` is the number of examples, `y_i ∈ {-1, +1}` the label of
//! example `i`, `h(x_i) ∈ [-1, 1]` a hypothesis' confidence, and
//! `y_i h(x_i)` its margin on example `i`. A *distribution* is a vector of
//! nonnegative entries summing to one.
//!
//! Log-domain functions ([`softmax`], [`normalize_log_weights`],
//! [`log_weighted_edge_coefficient`],
//! [`project_log_distribution_to_capped_simplex`]) share one input contract:
//! entries must be finite or negative infinity (meaning an exact zero weight),
//! with at least one finite entry. They panic otherwise.
use rayon::prelude::*;

use crate::core::checkers;
use crate::core::{Classifier, Sample};

/// Returns the edge of `h` with respect to `dist`:
/// ```txt
/// edge = Σ_i dist[i] * y_i * h(x_i)
/// ```
/// For a distribution and confidences in `[-1, 1]` the edge lies in
/// `[-1, 1]`; for `±1`-valued `h` it equals `1 - 2 * (weighted error)`.
/// It is a weighted correlation, not a loss: larger is better.
/// If `dist` and the sample differ in length, the extra entries are ignored.
///
/// Time complexity: `O(m)`, where `m` is the number of training examples.
///
/// # Examples
/// ```
/// use std::io::BufReader;
/// use miniboosts::{Classifier, Sample};
/// use miniboosts::core::helpers;
///
/// // h(x) = sign(x), a {-1, +1}-valued hypothesis.
/// struct SignOfX;
/// impl Classifier for SignOfX {
///     fn confidence(&self, sample: &Sample, row: usize) -> f64 {
///         sample["x"][row].signum()
///     }
/// }
///
/// let csv = b"x,class\n-2,-1\n-1,1\n1,1\n2,1\n";
/// let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
///     .unwrap()
///     .set_target("class");
///
/// // Only row 1 is misclassified. Under the uniform distribution the
/// // weighted error is 1/4, so the edge is 1 - 2 * 1/4 = 1/2 (exact).
/// let edge = helpers::edge(&sample, &[0.25; 4], &SignOfX);
/// assert_eq!(edge, 0.5);
/// // Moving all weight to the misclassified example gives edge -1.
/// assert_eq!(helpers::edge(&sample, &[0.0, 1.0, 0.0, 0.0], &SignOfX), -1.0);
/// ```
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

/// Returns `Σ_i dist[i] * margins[i]`, i.e., [`edge`] computed from
/// precomputed margins `margins[i] = y_i h(x_i)`.
/// Extra entries of the longer slice are ignored.
///
/// # Examples
/// ```
/// use miniboosts::core::helpers::edge_from_margins;
///
/// // Margins of a hypothesis that errs only on the second example.
/// let margins = [1.0, -1.0, 1.0, 1.0];
/// assert_eq!(edge_from_margins(&margins, &[0.25; 4]), 0.5);
/// ```
pub fn edge_from_margins(margins: &[f64], dist: &[f64]) -> f64 {
    margins.iter().zip(dist).map(|(yh, d)| *d * yh).sum::<f64>()
}

/// Returns an iterator over the margins `y_i * h(x_i)` of `h` on every
/// example of `sample`, in row order.
///
/// Yields nothing if the labels of `sample` are unset.
///
/// Time complexity: `O(m)`, where `m` is the number of training examples.
///
/// # Examples
/// ```
/// # use std::io::BufReader;
/// # use miniboosts::{Classifier, Sample};
/// # use miniboosts::core::helpers;
/// #
/// # // h(x) = sign(x), a {-1, +1}-valued hypothesis.
/// # struct SignOfX;
/// # impl Classifier for SignOfX {
/// #     fn confidence(&self, sample: &Sample, row: usize) -> f64 {
/// #         sample["x"][row].signum()
/// #     }
/// # }
/// #
/// # let csv = b"x,class\n-2,-1\n-1,1\n1,1\n2,1\n";
/// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
/// #     .unwrap()
/// #     .set_target("class");
/// let margins: Vec<f64> = helpers::margins(&sample, &SignOfX).collect();
/// // A positive margin means a correct prediction.
/// assert_eq!(margins, vec![1.0, -1.0, 1.0, 1.0]);
/// // Reuse the margins for several distributions.
/// assert_eq!(helpers::edge_from_margins(&margins, &[0.25; 4]), 0.5);
/// ```
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

/// Returns the unnormalized log weights of the exponential distribution
/// induced by the (typically combined) hypothesis `h`:
/// ```txt
/// d[i] = - eta * y_i * h(x_i),
/// ```
/// where `(x_i, y_i)` is the `i`-th training example. For a combined
/// hypothesis `h = Σ_j w_j h_j`, this is `-eta * y_i * Σ_j w_j h_j(x_i)`.
///
/// Time complexity: `O(m * n)`, where
/// - `m` is the number of training examples and
/// - `n` is the number of hypotheses combined in `h`.
///
/// # Examples
/// ```
/// # use std::io::BufReader;
/// # use miniboosts::{Classifier, Sample};
/// # use miniboosts::core::helpers;
/// #
/// # // h(x) = sign(x), a {-1, +1}-valued hypothesis.
/// # struct SignOfX;
/// # impl Classifier for SignOfX {
/// #     fn confidence(&self, sample: &Sample, row: usize) -> f64 {
/// #         sample["x"][row].signum()
/// #     }
/// # }
/// #
/// # let csv = b"x,class\n-2,-1\n-1,1\n1,1\n2,1\n";
/// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
/// #     .unwrap()
/// #     .set_target("class");
/// let logs: Vec<f64> = helpers::log_exp_distribution(2.0, &sample, &SignOfX).collect();
/// // -eta * margin: misclassified examples get the larger log weight.
/// assert_eq!(logs, vec![-2.0, 2.0, -2.0, -2.0]);
/// ```
#[inline(always)]
pub fn log_exp_distribution<H>(eta: f64, sample: &Sample, h: &H) -> impl Iterator<Item = f64>
where
    H: Classifier,
{
    margins(sample, h).map(move |yhx| -eta * yhx)
}

/// Computes the capped exponential distribution for the (typically
/// combined) hypothesis `h`: the relative-entropy projection of
/// ```txt
/// q[i] ∝ exp( - eta * y_i * h(x_i) )
/// ```
/// onto the capped simplex `{d : 0 <= d_i <= 1/nu, Σ d_i = 1}`, computed by
/// [`project_log_distribution_to_capped_simplex`]. For `nu = 1` the result
/// is `q` itself.
///
/// # Panics
/// Panics if `nu` is not in `[1, m]` or if a log weight `-eta * y_i h(x_i)`
/// is NaN or `+∞` (see [`project_log_distribution_to_capped_simplex`]).
///
/// Time complexity: `O(m * n + m log m)`, where
/// - `m` is the number of training examples and
/// - `n` is the number of hypotheses combined in `h`.
///
/// # Examples
/// ```
/// # use std::io::BufReader;
/// # use miniboosts::{Classifier, Sample};
/// # use miniboosts::core::helpers;
/// #
/// # // h(x) = sign(x), a {-1, +1}-valued hypothesis.
/// # struct SignOfX;
/// # impl Classifier for SignOfX {
/// #     fn confidence(&self, sample: &Sample, row: usize) -> f64 {
/// #         sample["x"][row].signum()
/// #     }
/// # }
/// #
/// # let csv = b"x,class\n-2,-1\n-1,1\n1,1\n2,1\n";
/// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
/// #     .unwrap()
/// #     .set_target("class");
/// // Without capping (nu = 1): d ∝ exp(-eta * margin), so the
/// // misclassified example has e^(2 eta) times the weight of the others.
/// let d = helpers::exp_distribution(0.5, 1.0, &sample, &SignOfX);
/// let ratio = d[1] / d[0];
/// // Tolerance: a few roundings in exp and the normalization.
/// assert!((ratio - 1f64.exp()).abs() < 1e-12);
///
/// // With nu = 2, no entry may exceed 1/2.
/// let d = helpers::exp_distribution(10.0, 2.0, &sample, &SignOfX);
/// assert!(d.iter().all(|&di| di <= 0.5 + 1e-12));
/// assert!((d.iter().sum::<f64>() - 1.0).abs() < 1e-12);
/// ```
#[inline(always)]
pub fn exp_distribution<H>(eta: f64, nu: f64, sample: &Sample, h: &H) -> Vec<f64>
where
    H: Classifier,
{
    let log_dist = log_exp_distribution(eta, sample, h);

    project_log_distribution_to_capped_simplex(nu, log_dist)
}

/// Same as [`exp_distribution`], but from precomputed margins
/// `y_i f(x_i)`: returns the projection of `q[i] ∝ exp(-eta * margins[i])`
/// onto the capped simplex with cap `1 / nu`.
///
/// # Panics
/// Same conditions as [`project_log_distribution_to_capped_simplex`].
///
/// Computational complexity: `O(m log(m))`,
/// where `m` is the number of training examples.
///
/// # Examples
/// ```
/// use miniboosts::core::helpers::exp_distribution_from_margins;
///
/// let margins = [1.0, -1.0, 1.0, 1.0];
/// // Uncapped, d[1] would be e^(2 * 5) / (e^10 + 3) ≈ 0.9999; the cap
/// // 1 / nu = 1/2 binds and the other three share the rest equally.
/// let d = exp_distribution_from_margins(5.0, 2.0, margins.into_iter());
/// // Tolerance: rounding in the log-domain projection.
/// assert!((d[1] - 0.5).abs() < 1e-12);
/// for i in [0, 2, 3] {
///     assert!((d[i] - 1.0 / 6.0).abs() < 1e-12);
/// }
/// ```
#[inline(always)]
pub fn exp_distribution_from_margins<I>(eta: f64, nu: f64, margins: I) -> Vec<f64>
where
    I: Iterator<Item = f64>,
{
    let iter = margins.map(|yf| -eta * yf);
    project_log_distribution_to_capped_simplex(nu, iter)
}

/// Returns `ln(exp(a) + exp(b))`, adding two nonnegative quantities given in
/// the log domain without overflow. Negative infinity represents zero.
///
/// # Panics
/// Panics if either argument is NaN or positive infinity.
///
/// # Examples
/// ```
/// use miniboosts::core::helpers::logaddexp;
///
/// // ln(e^1000 + e^1000) = 1000 + ln 2, although e^1000 overflows.
/// assert!((logaddexp(1000.0, 1000.0) - (1000.0 + 2f64.ln())).abs() < 1e-12);
/// // ln 0 = -∞ is the identity element.
/// assert_eq!(logaddexp(f64::NEG_INFINITY, -3.0), -3.0);
/// ```
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

/// Returns the probabilities `p_i = exp(l_i) / Σ_j exp(l_j)` for log weights
/// `l = log_weights`.
///
/// Normalizes in O(n), using max-shifted exponentials and direct
/// division (equation (4) in Blanchard, Higham & Higham,
/// <https://arxiv.org/abs/1909.03469>). The sum uses compensated
/// (Neumaier) accumulation.
///
/// Negative infinity denotes exact zero; finite tiny probabilities may
/// underflow to `0.0`.
///
/// # Panics
/// Panics for empty input, NaN, positive infinity, or no finite entries.
///
/// # Examples
/// ```
/// use miniboosts::core::helpers::softmax;
///
/// // Log weights ln 1, ln 3, and -∞ (an exact zero), shifted by 800,
/// // which would overflow `exp` without the max shift.
/// let p = softmax(&[800.0, 800.0 + 3f64.ln(), f64::NEG_INFINITY]);
/// // Tolerance: rounding of ln 3 and exp (a few ulps).
/// assert!((p[0] - 0.25).abs() < 1e-12);
/// assert!((p[1] - 0.75).abs() < 1e-12);
/// assert_eq!(p[2], 0.0);
/// ```
pub fn softmax(log_weights: &[f64]) -> Vec<f64> {
    let max = log_weight_max(log_weights);
    let mut probabilities: Vec<_> = log_weights.iter().map(|v| (v - max).exp()).collect();
    let sum = compensated_sum(probabilities.iter().copied());
    probabilities.iter_mut().for_each(|p| *p /= sum);
    probabilities
}

/// Returns normalized probabilities and overwrites `log_weights` with the
/// normalized logs `l_i - ln Σ_j exp(l_j)`.
///
/// Uses the same contract and shifted computation as [`softmax`], without
/// taking logarithms of rounded probabilities. Finite underflowed
/// probabilities thus retain their log mass for later updates.
///
/// # Panics
/// Panics under the same conditions as [`softmax`], or if a finite
/// normalized log cannot be represented as a finite `f64`.
///
/// # Examples
/// ```
/// use miniboosts::core::helpers::normalize_log_weights;
///
/// // The second weight is e^-800 times the first: its probability
/// // underflows to 0, but its normalized log keeps the exact mass.
/// let mut logs = vec![0.0, -800.0];
/// let p = normalize_log_weights(&mut logs);
/// assert_eq!(p, vec![1.0, 0.0]);
/// // ln(1 + e^-800) rounds to 0, so the logs are exact here.
/// assert_eq!(logs, vec![0.0, -800.0]);
///
/// // The normalized logs satisfy ln Σ exp(l_i) = 0 (up to rounding).
/// let mut logs = vec![1.0, 2.0, 3.0];
/// let p = normalize_log_weights(&mut logs);
/// let total: f64 = logs.iter().map(|l| l.exp()).sum();
/// assert!((total - 1.0).abs() < 1e-12);
/// assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
/// ```
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

/// Returns `atanh(edge) = ½ ln((1 + edge) / (1 - edge))`, where
/// `edge = Σ_i p_i margins[i]` and `p = softmax(log_weights)`, without
/// mistaking rounded probabilities for perfect classification.
/// This is AdaBoost's hypothesis coefficient `α = ½ ln((1 + γ) / (1 - γ))`.
///
/// Weights need not be normalized. Margins must be finite and in `[-1, 1]`,
/// with matching lengths; log weights follow [`softmax`]'s contract.
///
/// Uses compensated summation and `ln_1p` near zero (`|edge| < 0.5`).
/// Otherwise computes half the log ratio of weighted `(1 + margin)` and
/// `(1 - margin)` sums, avoiding cancellation in `1 - edge`.
/// Returns positive/negative infinity only when every supported margin
/// (entry with finite log weight) is exactly positive/negative one.
///
/// # Panics
/// Panics for invalid log weights (see [`softmax`]), mismatched lengths,
/// margins that are non-finite or outside `[-1, 1]`, or if a finite
/// log-weight spread is not representable in `f64`.
///
/// # Examples
/// ```
/// use miniboosts::core::helpers::log_weighted_edge_coefficient;
///
/// // Uniform weights (unnormalized logs are fine) and edge 1/2:
/// // α = ½ ln(1.5 / 0.5) = ½ ln 3.
/// let alpha = log_weighted_edge_coefficient(&[0.0; 4], &[1.0, -1.0, 1.0, 1.0]);
/// assert!((alpha - 0.5 * 3f64.ln()).abs() < 1e-12);
///
/// // A perfect hypothesis on the support gives +∞; the example with
/// // log weight -∞ (zero probability) does not count.
/// let logs = [0.0, 0.0, f64::NEG_INFINITY];
/// let alpha = log_weighted_edge_coefficient(&logs, &[1.0, 1.0, -1.0]);
/// assert_eq!(alpha, f64::INFINITY);
///
/// // A tiny but nonzero weight on a mistake keeps α finite, although
/// // its probability e^-800 underflows to zero.
/// let logs = [0.0, -800.0];
/// let alpha = log_weighted_edge_coefficient(&logs, &[1.0, -1.0]);
/// assert!(alpha.is_finite());
/// // With p ∝ (1, e^-800): α = ½ ln((2 p_0) / (2 p_1)) = ½ ln e^800 = 400.
/// // Tolerance: rounding of 800 ± ln 2 (ulp(800) ≈ 1.1e-13).
/// assert!((alpha - 400.0).abs() < 1e-12);
/// ```
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

/// Projects the distribution `q ∝ exp(l)` with log weights `l = iter`
/// onto the capped simplex `{d: 0 <= d[i] <= 1/nu, sum(d) = 1}` in relative
/// entropy (KL divergence), returning `d`.
///
/// The solution caps the largest entries at `1/nu` and rescales the rest
/// proportionally to `q`. Uses the sorted active-set method of
/// Shalev-Shwartz and Singer (2010), "On the equivalence of weak
/// learnability and linear separability". Sorting is required here;
/// complexity is O(n log n). For `nu = 1` this is [`softmax`].
///
/// Negative-infinity entries stay zero, so fewer than `nu` finite entries
/// make the constrained support infeasible.
///
/// # Panics
/// Panics for invalid log weights (see [`softmax`]), non-finite `nu`, or
/// `nu` outside `1..=support_size`, where `support_size` is the number of
/// finite entries. Also panics if the result fails
/// [`checkers::capped_simplex_condition`].
///
/// # Examples
/// ```
/// use miniboosts::core::helpers::project_log_distribution_to_capped_simplex;
///
/// // q ∝ (8, 1, 1, 0): logs ln 8, 0, 0, -∞.
/// let logs = [8f64.ln(), 0.0, 0.0, f64::NEG_INFINITY];
///
/// // nu = 1 is the plain softmax: q = (0.8, 0.1, 0.1, 0).
/// let d = project_log_distribution_to_capped_simplex(1.0, logs.into_iter());
/// assert!((d[0] - 0.8).abs() < 1e-12);
///
/// // nu = 2 caps every entry at 1/2. The largest entry is capped and the
/// // remaining mass 1/2 is split in proportion to q: (0.5, 0.25, 0.25, 0).
/// let d = project_log_distribution_to_capped_simplex(2.0, logs.into_iter());
/// // Tolerance: rounding of ln 8 and the normalization (a few ulps).
/// for (di, expected) in d.iter().zip([0.5, 0.25, 0.25, 0.0]) {
///     assert!((di - expected).abs() < 1e-12);
/// }
///
/// // nu = 3 equals the support size: uniform on the support.
/// let d = project_log_distribution_to_capped_simplex(3.0, logs.into_iter());
/// assert_eq!(d, vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0]);
/// ```
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

/// Returns `Σ_i d_i ln d_i + ln n` for `d = dist` of length `n`
/// (terms with `d_i = 0` count as zero).
///
/// When `dist` is a distribution this is the relative entropy (KL
/// divergence) `KL(d || u)` from the uniform distribution `u`, which lies in
/// `[0, ln n]`. The input is not validated.
///
/// # Examples
/// ```
/// use miniboosts::core::helpers::entropy_from_uni_distribution;
///
/// // KL(u || u) = 0 and KL(e_1 || u) = ln n (tolerance: rounding of ln).
/// assert!(entropy_from_uni_distribution([0.25; 4]).abs() < 1e-15);
/// let point_mass = [1.0, 0.0, 0.0, 0.0];
/// assert!((entropy_from_uni_distribution(point_mass) - 4f64.ln()).abs() < 1e-15);
/// ```
#[inline(always)]
pub fn entropy_from_uni_distribution<T: AsRef<[f64]>>(dist: T) -> f64 {
    let dist = dist.as_ref();
    let n_dim = dist.len() as f64;
    let e = entropy(dist);

    e + n_dim.ln()
}

/// Returns `Σ_i d_i ln d_i` (natural logarithm, terms with `d_i = 0` count
/// as zero).
///
/// **This is the negative Shannon entropy**, not the entropy: for a
/// distribution of length `n` the result lies in `[-ln n, 0]`. The input is
/// not validated; negative entries give NaN.
///
/// # Examples
/// ```
/// use miniboosts::core::helpers::entropy;
///
/// // Uniform on 4 points: 4 * (1/4) ln(1/4) = -ln 4, the minimum.
/// assert!((entropy([0.25; 4]) + 4f64.ln()).abs() < 1e-15);
/// // A point mass gives the maximum 0 (0 ln 0 counts as 0).
/// assert_eq!(entropy([0.0, 1.0, 0.0]), 0.0);
/// ```
#[inline(always)]
pub fn entropy<T: AsRef<[f64]>>(dist: T) -> f64 {
    let dist = dist.as_ref();
    dist.iter()
        .copied()
        .map(|d| if d == 0.0 { 0.0 } else { d * d.ln() })
        .sum::<f64>()
}

/// Returns the inner product `Σ_i v1[i] * v2[i]`.
///
/// Extra entries of the longer slice are ignored. The sum is computed in
/// parallel, so the floating-point summation order is unspecified.
///
/// # Examples
/// ```
/// use miniboosts::core::helpers::inner_product;
///
/// // Small integers: exact in any summation order.
/// assert_eq!(inner_product(&[1.0, 2.0, 3.0], &[4.0, -5.0, 6.0]), 12.0);
/// ```
#[inline(always)]
pub fn inner_product(v1: &[f64], v2: &[f64]) -> f64 {
    v1.into_par_iter().zip(v2).map(|(a, b)| a * b).sum::<f64>()
}

/// Divides every entry by the L1 norm `Σ_i |items[i]|` in place.
///
/// For a nonnegative vector the result is a distribution. Signs are kept:
/// with mixed signs the result has L1 norm one but does not sum to one.
///
/// # Panics
/// Panics if the L1 norm is zero (including an empty slice).
/// A NaN entry is not rejected and propagates.
///
/// # Examples
/// ```
/// use miniboosts::core::helpers::normalize;
///
/// let mut d = vec![1.0, 3.0, 4.0];
/// normalize(&mut d);
/// assert_eq!(d, vec![0.125, 0.375, 0.5]); // exact: division by 8
///
/// // Signs are kept: the L1 norm is one, the sum is not.
/// let mut w = vec![-1.0, 3.0];
/// normalize(&mut w);
/// assert_eq!(w, vec![-0.25, 0.75]);
/// ```
#[inline(always)]
pub fn normalize(items: &mut [f64]) {
    let z = items.iter().map(|it| it.abs()).sum::<f64>();

    assert_ne!(z, 0.0, "{items:?}");

    items.par_iter_mut().for_each(|item| {
        *item /= z;
    });
}

/// Returns the entry-wise (Hadamard) product of two matrices given as
/// row vectors, reusing `m1`'s storage.
///
/// # Panics
/// Panics if `m1` is empty, or if the numbers of rows or the lengths of the
/// first rows differ. Other rows are not checked; extra entries of a longer
/// row are ignored (`m1` keeps them unchanged).
///
/// # Examples
/// ```
/// use miniboosts::core::helpers::hadamard_product;
///
/// let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
/// let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
/// assert_eq!(hadamard_product(a, b), vec![vec![5.0, 12.0], vec![21.0, 32.0]]);
/// ```
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

/// Formats a nonnegative count for display with a metric suffix:
/// values below `1000` are printed as-is; larger values are divided by
/// `1e3`, `1e6`, or `1e9` and printed with one decimal and `K`, `M`, or `G`
/// (e.g., `12345.0` → `"12.3K"`).
///
/// # Examples
/// ```
/// use miniboosts::core::helpers::format_unit;
///
/// assert_eq!(format_unit(999.0), "999");
/// assert_eq!(format_unit(12_345.0), "12.3K");
/// assert_eq!(format_unit(2_500_000.0), "2.5M");
/// assert_eq!(format_unit(7e9), "7.0G");
/// ```
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

/// Returns the index of a minimal entry of `arr`. On ties, the **last**
/// minimal index is returned. NaN entries are not supported.
///
/// # Panics
/// Panics if `arr` is empty or every entry is greater than `f64::MAX`
/// (i.e., `+∞`).
///
/// # Examples
/// ```
/// use miniboosts::core::helpers::argmin;
///
/// assert_eq!(argmin(&[3.0, -1.0, 2.0]), 1);
/// assert_eq!(argmin(&[0.0, 5.0, 0.0]), 2); // ties: the last index
/// ```
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

/// Returns the index of a maximal entry of `arr`. On ties, the **first**
/// maximal index is returned. NaN entries are ignored.
///
/// # Panics
/// Panics if `arr` is empty or no entry is greater than `f64::MIN`
/// (e.g., all entries are `-∞`, `f64::MIN`, or NaN).
///
/// # Examples
/// ```
/// use miniboosts::core::helpers::argmax;
///
/// assert_eq!(argmax(&[3.0, -1.0, 2.0]), 0);
/// assert_eq!(argmax(&[1.0, 5.0, 5.0]), 1); // ties: the first index
/// assert_eq!(argmax(&[f64::NAN, 0.5]), 1); // NaN is ignored
/// ```
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
