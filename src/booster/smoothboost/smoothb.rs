//! This file defines `SmoothBoost` based on the paper
//! ``Smooth Boosting and Learning with Malicious Noise''
//! by Rocco A. Servedio.


use polars::prelude::*;
use rayon::prelude::*;

use crate::{
    Booster,
    WeakLearner,

    State,
    Classifier,
    CombinedHypothesis,
};


/// SmoothBoost. See Figure 1
/// in [this paper](https://www.jmlr.org/papers/volume4/servedio03a/servedio03a.pdf).
pub struct SmoothBoost<F> {
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

    max_iter: usize,

    classifiers: Vec<F>,


    m: Vec<f64>,
    n: Vec<f64>,
}


impl<F> SmoothBoost<F> {
    /// Initialize `SmoothBoost`.
    pub fn init(data: &DataFrame, _target: &Series) -> Self {
        let n_sample = data.shape().0;

        let gamma = 0.5;


        Self {
            kappa: 0.5,
            theta: gamma / (2.0 + gamma), // gamma / (2.0 + gamma)
            gamma,

            n_sample,

            terminated: usize::MAX,
            max_iter: usize::MAX,

            classifiers: Vec::new(),

            m: Vec::new(),
            n: Vec::new(),
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
                "Invalid kappa. \
                 The parameter `kappa` must be in (0.0, 1.0)"
            );
        }

        // Check `gamma`.
        if !(self.theta..0.5).contains(&self.gamma) {
            panic!(
                "Invalid gamma. \
                 The parameter `gamma` must be in [self.theta, 0.5)"
            );
        }
    }
}



impl<F> Booster<F> for SmoothBoost<F>
    where F: Classifier + Clone,
{
    fn preprocess<W>(
        &mut self,
        _weak_learner: &W,
        data: &DataFrame,
        _target: &Series,
    )
        where W: WeakLearner<Hypothesis = F>
    {
        self.n_sample = data.shape().0;
        // Set the paremeter `theta`.
        self.theta();

        // Check whether the parameter satisfies the pre-conditions.
        self.check_preconditions();


        self.max_iter = self.max_loop();
        self.terminated = self.max_iter;

        self.classifiers = Vec::new();


        self.m = vec![1.0; self.n_sample];
        self.n = vec![1.0; self.n_sample];
    }


    fn boost<W>(
        &mut self,
        weak_learner: &W,
        data: &DataFrame,
        target: &Series,
        iteration: usize,
    ) -> State
        where W: WeakLearner<Hypothesis = F>
    {

        if self.max_iter < iteration {
            return State::Terminate;
        }


        let sum = self.m.iter().sum::<f64>();
        // Check the stopping criterion.
        if sum < self.n_sample as f64 * self.kappa {
            self.terminated = iteration - 1;
            return State::Terminate;
        }


        // Compute the distribution.
        let dist = self.m.iter()
            .map(|mj| *mj / sum)
            .collect::<Vec<_>>();


        // Call weak learner to obtain a hypothesis.
        self.classifiers.push(
            weak_learner.produce(data, target, &dist[..])
        );
        let h: &F = self.classifiers.last().unwrap();


        let margins = target.i64()
            .expect("The target is not a dtype i64")
            .into_iter()
            .enumerate()
            .map(|(i, y)| y.unwrap() as f64 * h.confidence(data, i));


        // Update `n`
        self.n.iter_mut()
            .zip(margins)
            .for_each(|(nj, yh)| {
                *nj = *nj + yh - self.theta;
            });


        // Update `m`
        self.m.par_iter_mut()
            .zip(&self.n[..])
            .for_each(|(mj, nj)| {
                if *nj <= 0.0 {
                    *mj = 1.0;
                } else {
                    *mj = (1.0 - self.gamma).powf(*nj * 0.5);
                }
            });

        State::Continue
    }


    fn postprocess<W>(
        &mut self,
        _weak_learner: &W,
        _data: &DataFrame,
        _target: &Series,
    ) -> CombinedHypothesis<F>
        where W: WeakLearner<Hypothesis = F>
    {
        let weight = 1.0 / self.terminated as f64;
        let clfs = self.classifiers.clone()
            .into_iter()
            .map(|h| (weight, h))
            .collect::<Vec<(f64, F)>>();

        CombinedHypothesis::from(clfs)
    }
}
