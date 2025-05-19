//! Provides some helper functions.
use rayon::prelude::*;

use crate::{
    Sample,
    Classifier,
};
use crate::checkers;
use crate::constants::BINARY_SEARCH_TOLERANCE;

/// Returns the edge of a single hypothesis for the given distribution.
/// Here `edge` is the weighted training loss.
/// 
/// Time complexity: `O(m)`, where `m` is the number of training examples.
#[inline(always)]
pub fn edge<H>(
    sample: &Sample,
    dist: &[f64],
    h: &H
) -> f64
    where H: Classifier,
{
    margins(sample, h)
        .zip(dist)
        .map(|(yh, d)| *d * yh)
        .sum::<f64>()
}

pub fn edge_from_margins(
    margins: &[f64],
    dist: &[f64],
) -> f64
{
    margins.iter()
        .zip(dist)
        .map(|(yh, d)| *d * yh)
        .sum::<f64>()
}

/// Returns the margin vector of a single hypothesis
/// for the given distribution.
/// 
/// Time complexity: `O(m)`, where `m` is the number of training examples.
#[inline(always)]
pub fn margins<H>(sample: &Sample, h: &H)
    -> impl Iterator<Item=f64>
    where H: Classifier,
{
    let targets = sample.target();

    targets.iter()
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
pub fn log_exp_distribution<H>(
    eta: f64,
    sample: &Sample,
    h: &H,
) -> impl Iterator<Item = f64>
    where H: Classifier,
{
    margins(sample, h)
        .map(move |yhx| - eta * yhx)
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
pub fn exp_distribution<H>(
    eta: f64,
    nu: f64,
    sample: &Sample,
    h: &H,
) -> Vec<f64>
    where H: Classifier,
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
    where H: Classifier,
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
    where I: Iterator<Item = f64>,
{
    let power = 1f64 / (1f64 - deform);
    let mut g = margins.map(|yf| - (1f64 - deform) * eta * yf)
        .collect::<Vec<f64>>();

    let max = g.iter().fold(f64::MIN, |acc, val| val.max(acc));
    let (mut lb, mut ub) = (- max, 1f64 - max);

    while ub - lb > BINARY_SEARCH_TOLERANCE {
        let normalizer = (ub + lb) / 2f64;
        let sum = g.iter()
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
    g.iter_mut()
        .for_each(|val| { *val = (*val + normalizer).max(0f64).powf(power); });

    assert!(
        g.iter().all(|gi| gi.is_finite()),
        "invalid value in gradient. g = {g:?}"
    );
    checkers::capped_simplex_condition(&g[..], 1f64);

    deformed_projection_onto_capped_simplex(nu, deform, g)
}

#[inline(always)]
fn deformed_projection_onto_capped_simplex(nu: f64, t: f64, mut q: Vec<f64>)
    -> Vec<f64>
{
    assert!(q.iter().all(|qi| qi.is_finite()), "{q:?}");
    fn d(t: f64, q: f64, xi: f64) -> f64 {
        assert!((0f64..=1f64).contains(&t), "t = {t}");
        assert!((0f64..=1f64).contains(&q), "q = {q}");

        let ret = (q.powf(1.0 - t) + (1.0 - t) * xi).max(0f64)
            .powf(1.0 / (1.0 - t));
        assert!(
            ret.is_finite(),
            "d = {ret}, q = {q}, q^(1-t) = {}", q.powf(1f64 - t)
        );
        ret
    }

    fn compute_xi(
        mut lb: f64,
        amount: f64,
        t: f64,
        ix: &[usize],
        q: &[f64],
    ) -> f64
    {

        // DEBUG
        let mut ub = 1f64;
        loop {
            let sum = ix.iter()
                .map(|&i| d(t, q[i], ub))
                .sum::<f64>();
            if sum >= amount {
                break;
            }
            ub *= 2f64;
        }
        while ub - lb > 0f64 {
            let xi = (lb + ub) / 2f64;
            let sum = ix.iter()
                .map(|&i| d(t, q[i], xi))
                .sum::<f64>();
            if sum < amount { lb = xi; } else { ub = xi; }
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

    let lb = - 1f64 / (1f64 - t);
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
    where H: Classifier,
{
    let power = 1f64 / (1f64 - deform);
    // (1-t) η θ
    let mut g = margins(sample, f)
        .map(|yf| - (1f64 - deform) * eta * yf)
        .collect::<Vec<f64>>();

    let max = g.iter().fold(f64::MIN, |acc, val| val.max(acc));
    let (mut lb, mut ub) = (- max, 1f64 - max);

    while ub - lb > BINARY_SEARCH_TOLERANCE {
        let normalizer = (ub + lb) / 2f64;
        let sum = g.iter()
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
    g.iter_mut()
        .for_each(|val| { *val = (*val + normalizer).max(0f64).powf(power); });

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
pub fn exp_distribution_from_margins<I>(
    eta: f64,
    nu: f64,
    margins: I,
) -> Vec<f64>
    where I: Iterator<Item = f64>,
{
    let iter = margins.map(|yf| - eta * yf);
    project_log_distribution_to_capped_simplex(nu, iter)
}

/// Projects the given logarithmic distribution onto the capped simplex.
/// Capped simplex with parameter `ν (nu)` is defined as
/// 
/// ```txt
/// Δ_{m, ν} := { d ∈ [0, 1/ν]^m | sum( d[i] ) = 1 }
/// ```
/// 
/// That is, each coordinate takes at most `1/ν`.
/// Specifying `ν = 1` yields the no-capped simplex.
#[inline(always)]
pub fn project_log_distribution_to_capped_simplex<I>(
    nu: f64,
    iter: I,
) -> Vec<f64>
    where I: Iterator<Item = f64>,
{
    let mut dist: Vec<_> = iter.collect();
    let m = dist.len();

    // Construct a vector of indices `ix.`
    let mut ix = (0..m).collect::<Vec<usize>>();
    // sort `ix` in the descending order of `dist`.
    ix.sort_by(|&i, &j| dist[j].partial_cmp(&dist[i]).unwrap());

    // `logsums[k] = ln( sum_{i=0}^{k-1} exp( -η (Aw)i ) )
    let mut logsums = Vec::with_capacity(m);
    let mut last    = dist[ix[m-1]];
    logsums.push(last);
    for &i in ix.iter().rev().skip(1) {
        let small = last.min(dist[i]);
        let large = last.max(dist[i]);
        last = large + (1f64 + (small - large).exp()).ln();
        logsums.push(last);
    }

    // NOTE:
    // The following is the efficient projection 
    // onto the probability simplex capped by `1/ν.`
    // This code comes from the paper:
    //
    // Shai Shalev-Shwartz and Yoram Singer.
    // On the equivalence of weak learnability and linear separability:
    // new relaxations and efficient boosting algorithms.
    // [Journal of Machine Learning 2010]
    //
    // Note that the parameter `ν` in the paper corresponds to
    // `1/nu` in this code.
    let ub = 1.0 / nu;
    let log_nu = nu.ln();
    let mut ix_with_logsum = ix.into_iter()
        .zip(logsums.into_iter().rev())
        .enumerate();
    for (i, (i_sorted, logsum)) in ix_with_logsum.by_ref() {
        let log_xi = (1.0 - ub * i as f64).ln() - logsum;
        // TODO replace this line by `get_unchecked`
        let d = dist[i_sorted];

        // Check the stopping criterion
        if log_xi + d + log_nu <= 0.0 {
            dist[i_sorted] = (log_xi + d).exp();
            for (_, (ii, _)) in ix_with_logsum {
                dist[ii] = (log_xi + dist[ii]).exp();
            }
            break;
        }

        dist[i_sorted] = ub;
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
    v1.into_par_iter()
        .zip(v2)
        .map(|(a, b)| a * b)
        .sum::<f64>()
}

/// Normalizes the given slice.
#[inline(always)]
pub fn normalize(items: &mut [f64]) {
    let z = items.iter()
        .map(|it| it.abs())
        .sum::<f64>();

    assert_ne!(z, 0.0, "{items:?}");

    items.par_iter_mut()
        .for_each(|item| { *item /= z; });
}

/// Computes the Hadamard product of given two matrices.
#[inline(always)]
pub fn hadamard_product(mut m1: Vec<Vec<f64>>, m2: Vec<Vec<f64>>)
    -> Vec<Vec<f64>>
{
    assert_eq!(m1.len(), m2.len());
    assert_eq!(m1[0].len(), m2[0].len());

    m1.iter_mut()
        .zip(m2)
        .for_each(|(r1, r2)| {
            r1.iter_mut()
                .zip(r2)
                .for_each(|(a, b)| { *a *= b; });
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
    (1f64 + (1f64 - t) * x)
        .max(0f64)
        .powf(1f64 / (1f64 - t))
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
    let (ix, _) = arr.iter()
        .enumerate()
        .fold((dim, f64::MAX), |acc, (i, &a)| {
            if acc.1 < a {
                acc
            } else {
                (i, a)
            }
        });
    assert_ne!(
        ix, dim,
        "failed to execute argmin. array is {arr:?}"
    );
    ix
}

/// Returns an index whose entry is the maximal value.
pub fn argmax(arr: &[f64]) -> usize {
    let dim = arr.len();
    let (ix, _) = arr.iter()
        .enumerate()
        .fold((dim, f64::MIN), |acc, (i, &a)| {
            if acc.1 < a {
                (i, a)
            } else {
                acc
            }
        });
    assert_ne!(
        ix, dim,
        "failed to execute argmax. array is {arr:?}"
    );
    ix
}

#[cfg(test)]
mod tests {
    use super::*;

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
            assert_eq!(
                e, yh,
                "failed for {i}th example. expected {e}, got {yh}."
            );
        }
    }

    #[test]
    fn test_margins_02() {
        let sample = training_examples_case_02();
        let h = TestHypothesis::new(0f64);
        let margins = margins(&sample, &h);
        let expected = [-1.0, 1.0, -1.0, 1.0];
        for (i, (e, yh)) in expected.into_iter().zip(margins).enumerate() {
            assert_eq!(
                e, yh,
                "failed for {i}th example. expected {e}, got {yh}."
            );
        }
    }
}

