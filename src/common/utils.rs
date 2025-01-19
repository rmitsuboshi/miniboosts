//! This file provides some common functions
//! such as edge calculation.
use rayon::prelude::*;


use crate::{Sample, Classifier};
use crate::common::checker;


/// Returns the edge of a single hypothesis for the given distribution.
/// Here `edge` is the weighted training loss.
/// 
/// Time complexity: `O(m)`, where `m` is the number of training examples.
#[inline(always)]
pub fn edge_of_hypothesis<H>(
    sample: &Sample,
    dist: &[f64],
    h: &H
) -> f64
    where H: Classifier,
{
    margins_of_hypothesis(sample, h)
        .into_iter()
        .zip(dist)
        .map(|(yh, d)| *d * yh)
        .sum::<f64>()
}


/// Returns the margin vector of a single hypothesis
/// for the given distribution.
/// 
/// Time complexity: `O(m)`, where `m` is the number of training examples.
#[inline(always)]
pub fn margins_of_hypothesis<H>(sample: &Sample, h: &H)
    -> Vec<f64>
    where H: Classifier,
{
    let targets = sample.target();

    targets.iter()
        .enumerate()
        .map(|(i, y)| y * h.confidence(sample, i))
        .collect()
}


/// Returns the edge of a weighted hypothesis for the given distribution.
/// 
/// Time complexity: `O(m * n)`, where
/// - `m` is the number of training examples and
/// - `n` is the number of hypotheses.
#[inline(always)]
pub fn edge_of_weighted_hypothesis<H>(
    sample: &Sample,
    dist: &[f64],
    weights: &[f64],
    hypotheses: &[H],
) -> f64
    where H: Classifier,
{
    margins_of_weighted_hypothesis(sample, weights, hypotheses)
        .into_iter()
        .zip(dist)
        .map(|(yh, d)| *d * yh)
        .sum::<f64>()
}


/// Returns the margin vector of a weighted hypothesis
/// for the given distribution.
/// 
/// Time complexity: `O(m * n)`, where
/// - `m` is the number of training examples and
/// - `n` is the number of hypotheses.
#[inline(always)]
pub fn margins_of_weighted_hypothesis<H>(
    sample: &Sample,
    weights: &[f64],
    hypotheses: &[H],
) -> Vec<f64>
    where H: Classifier,
{
    let targets = sample.target();

    targets.iter()
        .enumerate()
        .map(|(i, y)| {
            let fx = weights.iter()
                .copied()
                .zip(hypotheses)
                .map(|(w, h)| w * h.confidence(sample, i))
                .sum::<f64>();
            y * fx
        })
        .collect()
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
    weights: &[f64],
    hypotheses: &[H],
) -> impl Iterator<Item = f64>
    where H: Classifier,
{
    margins_of_weighted_hypothesis(sample, weights, hypotheses)
        .into_iter()
        .map(move |yf| - eta * yf)
}


/// Computes the logarithm of 
/// the exponential distribution for the given combined hypothesis.
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
    weights: &[f64],
    hypotheses: &[H],
) -> Vec<f64>
    where H: Classifier,
{
    let log_dist = log_exp_distribution(eta, sample,weights, hypotheses);

    project_log_distribution_to_capped_simplex(nu, log_dist)
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
    let n_sample = dist.len();

    // Construct a vector of indices `ix.`
    let mut ix = (0..n_sample).collect::<Vec<usize>>();
    // sort `ix` in the descending order of `dist`.
    ix.sort_by(|&i, &j| dist[j].partial_cmp(&dist[i]).unwrap());


    // `logsums[k] = ln( sum_{i=0}^{k-1} exp( -η (Aw)i ) )
    let mut logsums: Vec<f64> = Vec::with_capacity(n_sample);
    ix.iter().rev()
        .copied()
        .for_each(|i| {
            let logsum = logsums.last()
                .map(|&v| {
                    let small = v.min(dist[i]);
                    let large = v.max(dist[i]);
                    large + (1.0 + (small - large).exp()).ln()
                })
                .unwrap_or(dist[i]);
            logsums.push(logsum);
        });

    let logsums = logsums.into_iter().rev();


    let ub = 1.0 / nu;
    let log_nu = nu.ln();

    let mut ix_with_logsum = ix.into_iter().zip(logsums).enumerate();


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
    checker::check_capped_simplex_condition(&dist, nu);
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


pub(crate) fn total_weight_for_label(
    y: f64,
    target: &[f64],
    weight: &[f64],
) -> f64
{
    target.iter()
        .copied()
        .zip(weight)
        .filter_map(|(t, w)| if t == y { Some(w) } else { None })
        .sum::<f64>()
}


pub(crate) fn format_unit(value: f64) -> String {
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
