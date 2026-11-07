//! Sample-optimal weak-to-strong learners that take an unweighted majority
//! vote of boosters run on sub-samples: [`MajorityOfFive`] (Høgsgaard,
//! Larsen, and Mathiasen, 2024) and [`LarsenRitzert`] (Larsen and Ritzert,
//! 2022).
//!
//! Both take a closure that trains a voting classifier on a sub-sample, so
//! any booster and weak learner can be plugged in. They do not implement
//! [`Booster`](crate::Booster), whose rounds call a single weak learner on
//! the whole sample.
mod larsen_ritzert;
mod majority_of_five;

pub use larsen_ritzert::LarsenRitzert;
pub use majority_of_five::MajorityOfFive;

use crate::MajorityVote;
use crate::core::{Classifier, Sample};
use crate::optimization::sparsification::SplitMix64;
use rayon::prelude::*;

/// Returns `0, ..., n - 1` in an order drawn uniformly at random with
/// `seed` (Fisher–Yates).
fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut rng = SplitMix64::new(seed);
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        // The modulo bias is below 2^-40 for n < 2^24.
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    order
}

/// Trains `train` on the sample's rows in each of `subsets`, in parallel,
/// and returns the unweighted majority vote of the results, in the order of
/// `subsets`.
fn vote<H, F>(sample: &Sample, subsets: Vec<Vec<usize>>, train: F) -> MajorityVote<H>
where
    H: Classifier + Send,
    F: Fn(&Sample) -> H + Sync,
{
    let hypotheses = subsets
        .into_par_iter()
        .map(|rows| train(&sample.select_rows(&rows)))
        .collect();
    MajorityVote::new(hypotheses)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffled_is_a_seeded_permutation() {
        let order = shuffled(50, 7);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());
        assert_eq!(order, shuffled(50, 7));
        assert_ne!(order, shuffled(50, 8));
        assert!(shuffled(0, 1).is_empty());
    }
}
