//! Sample-optimal weak-to-strong learners that take an unweighted majority
//! vote of boosters run on sub-samples: [`MajorityOfFive`] (Høgsgaard,
//! Larsen, and Mathiasen, 2024) and [`LarsenRitzert`] (Larsen and Ritzert,
//! 2022).
//!
//! Both take a closure that runs a booster on a sub-sample, so any booster
//! can be plugged in. With that closure set by `booster`, they implement
//! [`Booster`](crate::Booster): each round runs the closure on one
//! sub-sample with the weak learner given to `run`. `run_with` takes a
//! closure that trains on a sub-sample by any means and runs the
//! sub-samples in parallel.
mod larsen_ritzert;
mod majority_of_five;

pub use larsen_ritzert::LarsenRitzert;
pub use majority_of_five::MajorityOfFive;

use crate::MajorityVote;
use crate::core::{Classifier, DynWeakLearner, Sample, WeakLearner};
use crate::optimization::sparsification::SplitMix64;
use rayon::prelude::*;
use std::ops::ControlFlow;

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

/// The per-run state of a sub-sample majority used as a
/// [`Booster`](crate::Booster): the closure `train` that runs a booster,
/// the sub-samples of the current run, and the voters trained so far.
struct Voters<F, O> {
    train: F,
    subsets: Vec<Vec<usize>>,
    voters: Vec<O>,
}

impl<F, O> Voters<F, O> {
    fn new(train: F) -> Self {
        Self {
            train,
            subsets: Vec::new(),
            voters: Vec::new(),
        }
    }

    /// Starts a run over `subsets`.
    fn reset(&mut self, subsets: Vec<Vec<usize>>) {
        self.subsets = subsets;
        self.voters.clear();
    }

    /// Round `iteration` (1-based): trains on the rows of the
    /// `iteration`-th subset. Breaks, with the number of subsets, once the
    /// last one is trained.
    fn train<H, W>(
        &mut self,
        sample: &Sample,
        weak_learner: &W,
        iteration: usize,
    ) -> ControlFlow<usize>
    where
        F: Fn(&Sample, &DynWeakLearner<'_, H>) -> O,
        W: WeakLearner<Hypothesis = H>,
    {
        let total = self.subsets.len();
        if self.voters.len() < total {
            debug_assert_eq!(iteration, self.voters.len() + 1);
            let part = sample.select_rows(&self.subsets[self.voters.len()]);
            let voter = (self.train)(&part, &DynWeakLearner::new(weak_learner));
            self.voters.push(voter);
        }
        if self.voters.len() < total {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(total)
        }
    }

    /// The unweighted majority vote of the voters trained so far.
    fn vote(&self) -> MajorityVote<O>
    where
        O: Clone,
    {
        MajorityVote::new(self.voters.clone())
    }
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
