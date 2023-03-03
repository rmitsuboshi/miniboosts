//! Defines some functions that compute the distribution.

use super::utils::*;
use crate::{Sample, Classifier};

/// Returns the logarithmic distribution for the given weights.
fn log_dist_at<C>(
    eta: f64,
    sample: &Sample,
    classifiers: &[C],
    weights: &[f64]
) -> Vec<f64>
    where C: Classifier
{
    // Assign the logarithmic distribution in `dist`.
    sample.target()
        .into_iter()
        .enumerate()
        .map(|(i, y)| {
            let p = confidence(i, sample, classifiers, weights);
            - eta * y * p
        })
        .collect::<Vec<_>>()
}


/// Project the logarithmic distribution `dist`
/// onto the capped simplex.
fn projection(nu: f64, dist: &[f64]) -> Vec<f64> {

    let n_sample = dist.len();

    // Sort the indices over `dist` in non-increasing order.
    let mut ix = (0..n_sample).collect::<Vec<_>>();
    ix.sort_by(|&i, &j| dist[j].partial_cmp(&dist[i]).unwrap());


    let mut dist = dist.to_vec();


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
            while let Some((_, (ii, _))) = ix_with_logsum.next() {
                dist[ii] = (log_xi + dist[ii]).exp();
            }
            break;
        }

        dist[i_sorted] = ub;
    }
    dist
}


/// Computes the distribution (gradient) at current weight.
#[inline(always)]
pub(super) fn dist_at<C>(
    eta: f64,
    nu: f64,
    sample: &Sample,
    classifiers: &[C],
    weights: &[f64]
) -> Vec<f64>
    where C: Classifier
{
    let dist = log_dist_at(eta, sample, classifiers, weights);

    projection(nu, &dist[..])
}
