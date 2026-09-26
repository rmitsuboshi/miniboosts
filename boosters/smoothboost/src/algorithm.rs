//! This file defines `SmoothBoost` based on the paper
//! ``Smooth Boosting and Learning with Malicious Noise''
//! by Rocco A. Servedio.
use rayon::prelude::*;

use hypotheses::WeightedMajority;
use miniboosts_core::CurrentHypothesis;
use miniboosts_core::{Booster, Classifier, Sample, WeakLearner};

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
    /// with advantage at least `gamma` (edge at least `2 * gamma`).
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
    /// Initialize `SmoothBoost` with weak-learner advantage `gamma = 0.25`.
    pub fn init(sample: &'a Sample) -> Self {
        let n_examples = sample.shape().0;

        let gamma = 0.25;

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

    /// Set the weak-learner advantage `gamma` in `(0, 0.5)`.
    /// `gamma` is the weak learner guarantee;  
    /// `SmoothBoost` assumes the weak learner to returns a hypothesis `h`
    /// such that
    /// `0.5 * sum_i D[i] |h(x[i]) - y[i]| <= 0.5 - gamma`
    /// for the given distribution.  
    /// This assumption is required for the iteration bound.
    #[inline(always)]
    pub fn gamma(mut self, gamma: f64) -> Self {
        // A positive advantage is needed for a finite iteration bound.
        assert!(gamma > 0.0 && gamma < 0.5);
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
        let denom = self.kappa * self.gamma.powi(2) * (1.0 - self.gamma).sqrt();

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
where
    F: Classifier + Clone,
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
        // Servedio (2003), Figure 1, line 2: N_0(j) = 0.
        // https://www.jmlr.org/papers/volume4/servedio03a/servedio03a.pdf
        self.n = vec![0.0; self.n_examples];
    }

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize) -> ControlFlow<usize>
    where
        W: WeakLearner<Hypothesis = F>,
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
        let dist = self.m.iter().map(|mj| *mj / sum).collect::<Vec<_>>();

        // Call weak learner to obtain a hypothesis.
        self.hypotheses
            .push(weak_learner.produce(self.sample, &dist[..]));
        let h: &F = self.hypotheses.last().unwrap();

        let target = self.sample.target();
        let margins = target
            .iter()
            .enumerate()
            .map(|(i, y)| y * h.confidence(self.sample, i));

        // Update `n`
        self.n.iter_mut().zip(margins).for_each(|(nj, yh)| {
            *nj = *nj + yh - self.theta;
        });

        // Update `m`
        self.m.par_iter_mut().zip(&self.n[..]).for_each(|(mj, nj)| {
            if *nj <= 0.0 {
                *mj = 1.0;
            } else {
                *mj = (1.0 - self.gamma).powf(*nj * 0.5);
            }
        });

        ControlFlow::Continue(())
    }

    fn postprocess(&mut self) -> Self::Output {
        // Figure 1, line 11: average all T hypotheses, not the sample rows.
        let weights = vec![1f64; self.hypotheses.len()];
        WeightedMajority::from_slices(&weights[..], &self.hypotheses[..])
    }
}

impl<H> CurrentHypothesis for SmoothBoost<'_, H>
where
    H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        // Figure 1, line 11: average all T hypotheses, not the sample rows.
        let weights = vec![1f64; self.hypotheses.len()];
        WeightedMajority::from_slices(&weights[..], &self.hypotheses[..])
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    #[derive(Clone)]
    struct Constant(f64);
    impl Classifier for Constant {
        fn confidence(&self, _: &Sample, _: usize) -> f64 {
            self.0
        }
    }
    impl WeakLearner for Constant {
        type Hypothesis = Self;
        fn produce(&self, _: &Sample, _: &[f64]) -> Self {
            self.clone()
        }
    }

    #[test]
    fn default_training_keeps_every_round_and_starts_at_zero_margin() {
        let sample = Sample::dummy(2);
        let mut booster = SmoothBoost::init(&sample);
        booster.preprocess();
        assert_eq!(booster.n, vec![0.0; 2]);
        // Deliberately poor learner keeps training active for more rounds than rows.
        for iteration in 1..=3 {
            assert!(booster.boost(&Constant(1.0), iteration).is_continue());
        }
        for ensemble in [booster.current_hypothesis(), booster.postprocess()] {
            assert_eq!(ensemble.hypotheses.len(), 3);
            assert!(
                ensemble
                    .weights
                    .iter()
                    .all(|w| (*w - 1.0 / 3.0).abs() < 1e-14)
            );
        }
        assert!((booster.n[0] - 3.0 * (1.0 - booster.theta)).abs() < 1e-14);
    }

    #[test]
    #[should_panic]
    fn zero_advantage_is_rejected() {
        let sample = Sample::dummy(2);
        let _ = SmoothBoost::<Constant>::init(&sample).gamma(0.0);
    }
}
