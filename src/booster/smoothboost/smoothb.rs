//! This file defines `SmoothBoost` based on the paper
//! ``Smooth Boosting and Learning with Malicious Noise''
//! by Rocco A. Servedio.


use polars::prelude::*;
use rayon::prelude::*;

use crate::{
    Classifier,
    CombinedClassifier,
    BaseLearner,
    Booster,
};


/// SmoothBoost. See Figure 1
/// in [this paper](https://www.jmlr.org/papers/volume4/servedio03a/servedio03a.pdf).
pub struct SmoothBoost {
    /// Desired accuracy
    kappa: f64,

    /// Desired margin for the final hypothesis.
    /// To guarantee the convergence rate, `theta` should be
    /// `gamma / (2.0 + gamma)`.
    theta: f64,

    /// Weak-learner guarantee;
    /// for any distribution over the training examples,
    /// the weak-learner returns a hypothesis
    /// with edge at least `gamma`.
    gamma: f64,

    /// The number of training examples.
    n_sample: usize,

    /// Terminated iteration.
    terminated: usize,
}


impl SmoothBoost {
    /// Initialize `SmoothBoost`.
    pub fn init(data: &DataFrame, _target: &Series) -> Self {
        let n_sample = data.shape().0;


        Self {
            kappa: 0.5,
            theta: 0.5 / (2.0 + 0.5), // gamma / (2.0 + gamma)
            gamma: 0.5,

            n_sample,

            terminated: usize::MAX,
        }
    }


    /// Set the parameter `kappa`.
    #[inline(always)]
    pub fn tolerance(mut self, kappa: f64) -> Self {
        self.kappa = kappa;

        self
    }


    /// Set the parameter `gamma`.
    #[inline(always)]
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;

        self
    }


    /// Set the parameter `theta`.
    fn theta(&mut self) {
        self.theta = self.gamma / (2.0 + self.gamma);
    }


    /// Returns the maximum iteration
    /// of SmoothBoost to satisfy the stopping criterion.
    fn max_loop(&self) -> usize {
        let denom = self.kappa
            * self.gamma.powi(2)
            * (1.0 - self.gamma).sqrt();


        (2.0 / denom).ceil() as usize
    }


    fn check_preconditions(&self) {
        // Check `kappa`.
        if !(0.0..1.0).contains(&self.kappa) || self.kappa <= 0.0 {
            panic!(
                "Invalid kappa.\
                 The parameter `kappa` must be in (0.0, 1.0)"
            );
        }

        // Check `gamma`.
        if !(self.theta..0.5).contains(&self.gamma) {
            panic!(
                "Invalid gamma.\
                 The parameter `gamma` must be in [self.theta, 0.5)"
            );
        }
    }
}



impl<C> Booster<C> for SmoothBoost
    where C: Classifier,
{
    fn run<B>(
        &mut self,
        base_learner: &B,
        data:         &DataFrame,
        target:       &Series,
    ) -> CombinedClassifier<C>
        where B: BaseLearner<Clf = C>
    {
        // Set the paremeter `theta`.
        self.theta();

        // Check whether the parameter satisfies the pre-conditions.
        self.check_preconditions();


        let max_iter = self.max_loop();
        self.terminated = max_iter;


        let mut m = vec![1.0; self.n_sample];
        let mut n = vec![1.0; self.n_sample];


        let mut clfs = Vec::new();


        for t in 1..max_iter {
            let sum = m.iter().sum::<f64>();
            // Check the stopping criterion.
            if sum < self.n_sample as f64 * self.kappa {
                self.terminated = t - 1;
                break;
            }


            // Compute the distribution.
            let dist = m.iter()
                .map(|mj| *mj / sum)
                .collect::<Vec<_>>();


            // Call weak learner to obtain a hypothesis.
            clfs.push(
                base_learner.produce(data, target, &dist[..])
            );
            let h: &C = clfs.last().unwrap();


            let margins = target.i64()
                .expect("The target is not a dtype i64")
                .into_iter()
                .enumerate()
                .map(|(i, y)| y.unwrap() as f64 * h.confidence(data, i));


            // Update `n`
            n.iter_mut()
                .zip(margins)
                .for_each(|(nj, yh)| {
                    *nj = *nj + yh - self.theta;
                });


            // Update `m`
            m.par_iter_mut()
                .zip(&n[..])
                .for_each(|(mj, nj)| {
                    if *nj <= 0.0 {
                        *mj = 1.0;
                    } else {
                        *mj = (1.0 - self.gamma).powf(*nj * 0.5);
                    }
                });
        }

        // Compute the combined hypothesis
        let weight = 1.0 / self.terminated as f64;
        let clfs = clfs.into_iter()
            .map(|h| (weight, h))
            .collect::<Vec<(f64, C)>>();

        CombinedClassifier::from(clfs)
    }
}
