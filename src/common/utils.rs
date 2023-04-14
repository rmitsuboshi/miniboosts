//! This file provides some common functions
//! such as edge calculation.
use rayon::prelude::*;
use crate::{Sample, Classifier};


/// Returns the edge of a single hypothesis for the given distribution
#[inline(always)]
pub(crate) fn edge_of_hypothesis<H>(
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
/// for the given distribution
#[inline(always)]
pub(crate) fn margins_of_hypothesis<H>(sample: &Sample, h: &H)
    -> Vec<f64>
    where H: Classifier,
{
    let targets = sample.target();

    targets.iter()
        .enumerate()
        .map(|(i, y)| y * h.confidence(sample, i))
        .collect()
}


/// Returns the edge of a weighted hypothesis for the given distribution
#[inline(always)]
pub(crate) fn edge_of_weighted_hypothesis<H>(
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
/// for the given distribution
#[inline(always)]
pub(crate) fn margins_of_weighted_hypothesis<H>(
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


#[inline(always)]
pub(crate) fn log_exp_distribution<H>(
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


#[inline(always)]
pub(crate) fn exp_distribution<H>(
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


#[inline(always)]
pub(crate) fn exp_distribution_from_margins<I>(
    eta: f64,
    nu: f64,
    margins: I,
) -> Vec<f64>
    where I: Iterator<Item = f64>,
{
    let iter = margins.map(|yf| - eta * yf);
    project_log_distribution_to_capped_simplex(nu, iter)
}


#[inline(always)]
pub(crate) fn project_log_distribution_to_capped_simplex<I>(
    nu: f64,
    iter: I,
) -> Vec<f64>
    where I: Iterator<Item = f64>,
{
    let mut dist: Vec<_> = iter.collect();
    let n_sample = dist.len();

    // Construct a vector of indices sorted in descending order of `dist`.
    let mut ix = (0..n_sample).collect::<Vec<usize>>();
    ix.sort_by(|&i, &j| dist[j].partial_cmp(&dist[i]).unwrap());


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

    while let Some((i, (i_sorted, logsum))) = ix_with_logsum.next() {
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
    dist
}


/// Compute the relative entropy from the uniform distribution.
#[inline(always)]
pub(crate) fn entropy_from_uni_distribution(dist: &[f64]) -> f64 {
    let n_dim = dist.len() as f64;
    let e = entropy(dist);

    e + n_dim.ln()
}


/// Compute the entropy of the given distribution.
#[inline(always)]
pub(crate) fn entropy(dist: &[f64]) -> f64 {
    dist.iter()
        .copied()
        .map(|d| if d == 0.0 { 0.0 } else { d * d.ln() })
        .sum::<f64>()
}


/// Compute the inner-product of the given two slices.
#[inline(always)]
pub(crate) fn inner_product(v1: &[f64], v2: &[f64]) -> f64 {
    v1.into_par_iter()
        .zip(v2)
        .map(|(a, b)| a * b)
        .sum::<f64>()
}


#[inline(always)]
pub(crate) fn normalize(items: &mut [f64]) {
    let z = items.iter()
        .map(|it| it.abs())
        .sum::<f64>();

    assert_ne!(z, 0.0);

    items.par_iter_mut()
        .for_each(|item| { *item /= z; });
}


#[inline(always)]
pub(crate) fn hadamard_product(mut m1: Vec<Vec<f64>>, m2: Vec<Vec<f64>>)
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
