//! This file defines `SmoothBoost` based on the paper
//! ``Smooth Boosting and Learning with Malicious Noise''
//! by Rocco A. Servedio.
use rayon::prelude::*;

use miniboosts_core::{
    Booster,
    WeakLearner,
    Classifier,
    Sample,
};
use logging::CurrentHypothesis;
use hypotheses::WeightedMajority;

use std::ops::ControlFlow;

pub struct SmoothBoost<'a, F> {
    // Training sample
    sample: &'a Sample,

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
    n_examples: usize,

    current: usize,

    /// Terminated iteration.
    terminated: usize,

    max_iter: usize,

    hypotheses: Vec<F>,

    m: Vec<f64>,
    n: Vec<f64>,
}

impl<'a, F> SmoothBoost<'a, F> {
    /// Initialize `SmoothBoost`.
    pub fn init(sample: &'a Sample) -> Self {
        let n_examples = sample.shape().0;

        let gamma = 0.5;

        Self {
            sample,

            kappa: 0.5,
            theta: gamma / (2.0 + gamma), // gamma / (2.0 + gamma)
            gamma,

            n_examples,

            current: 0_usize,

            terminated: usize::MAX,
            max_iter: usize::MAX,

            hypotheses: Vec::new(),

            m: Vec::new(),
            n: Vec::new(),
        }
    }

    /// Set the tolerance parameter `kappa`.
    #[inline(always)]
    pub fn kappa(mut self, kappa: f64) -> Self {
        self.kappa = kappa;

        self
    }

    /// Set the parameter `gamma`.
    /// `gamma` is the weak learner guarantee;  
    /// `SmoothBoost` assumes the weak learner to returns a hypothesis `h`
    /// such that
    /// `0.5 * sum_i D[i] |h(x[i]) - y[i]| <= 0.5 - gamma`
    /// for the given distribution.  
    /// **Note that** this is an extremely assumption.
    #[inline(always)]
    pub fn gamma(mut self, gamma: f64) -> Self {
        // `gamma` must be in [0.0, 0.5)
        assert!((0.0..0.5).contains(&gamma));
        self.gamma = gamma;

        self
    }

    /// Set the parameter `theta`.
    fn theta(&mut self) {
        self.theta = self.gamma / (2.0 + self.gamma);
    }

    /// Returns the maximum iteration
    /// of SmoothBoost to satisfy the stopping split_by.
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

impl<F> Booster<F> for SmoothBoost<'_, F>
    where F: Classifier + Clone,
{
    type Output = WeightedMajority<F>;

    fn name(&self) -> &str {
        "SmoothBoost"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_examples, n_feature) = self.sample.shape();
        let info = Vec::from([
            ("# of examples", format!("{n_examples}")),
            ("# of features", format!("{n_feature}")),
            ("Tolerance (Kappa)", format!("{}", self.kappa)),
            ("Max iteration", format!("{}", self.max_iter)),
            ("Theta", format!("{}", self.theta)),
            ("Gamma (WL guarantee)", format!("{}", self.gamma)),
        ]);
        Some(info)
    }

    fn preprocess(&mut self) {
        self.sample.is_valid_binary_instance();
        self.n_examples = self.sample.shape().0;
        // Set the paremeter `theta`.
        self.theta();

        // Check whether the parameter satisfies the pre-conditions.
        self.check_preconditions();

        self.current = 0_usize;
        self.max_iter = self.max_loop();
        self.terminated = self.max_iter;

        self.hypotheses = Vec::new();

        self.m = vec![1.0; self.n_examples];
        self.n = vec![1.0; self.n_examples];
    }

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize)
        -> ControlFlow<usize>
        where W: WeakLearner<Hypothesis = F>
    {

        if self.max_iter < iteration {
            return ControlFlow::Break(self.max_iter);
        }

        self.current = iteration;

        let sum = self.m.iter().sum::<f64>();
        // Check the stopping split_by.
        if sum < self.n_examples as f64 * self.kappa {
            self.terminated = iteration - 1;
            return ControlFlow::Break(iteration);
        }

        // Compute the distribution.
        let dist = self.m.iter()
            .map(|mj| *mj / sum)
            .collect::<Vec<_>>();

        // Call weak learner to obtain a hypothesis.
        self.hypotheses.push(
            weak_learner.produce(self.sample, &dist[..])
        );
        let h: &F = self.hypotheses.last().unwrap();

        let target = self.sample.target();
        let margins = target.iter()
            .enumerate()
            .map(|(i, y)| y * h.confidence(self.sample, i));

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

        ControlFlow::Continue(())
    }

    fn postprocess(&mut self) -> Self::Output {
        let weight = 1f64 / self.terminated as f64;
        let weights = vec![weight; self.n_examples];
        WeightedMajority::from_slices(&weights[..], &self.hypotheses[..])
    }
}

impl<H> CurrentHypothesis for SmoothBoost<'_, H>
    where H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        let weight = 1f64 / self.terminated as f64;
        let weights = vec![weight; self.n_examples];
        WeightedMajority::from_slices(&weights[..], &self.hypotheses[..])
    }
}

