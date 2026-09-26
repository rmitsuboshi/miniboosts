//! This file defines `SoftBoost` based on the paper
//! "Boosting Algorithms for Maximizing the Soft Margin"
//! by Warmuth et al.
//!
use crate::solver::SoftBoostSolver;
use hypotheses::WeightedMajority;
use miniboosts_core::CurrentHypothesis;
use miniboosts_core::{
    Booster, Classifier, Sample, WeakLearner,
    constants::{DEFAULT_CAPPING, DEFAULT_TOLERANCE},
    tools::checkers,
    tools::helpers,
};
use optimization::soft_margin_optimization;

use std::ops::ControlFlow;

pub struct SoftBoost<'a, H> {
    sample: &'a Sample,

    pub(crate) dist: Vec<f64>,

    // `gamma_hat` corresponds to $\min_{q=1, .., t} P^q (d^{q-1})
    gamma_hat: f64,
    tolerance: f64,
    nu: f64,

    solver: SoftBoostSolver<'a>,

    hypotheses: Vec<H>,

    max_iter: usize,
    terminated: usize,

    weights: Vec<f64>,
    numerical_stop_reason: Option<String>,
}

impl<'a, H> SoftBoost<'a, H>
where
    H: Classifier,
{
    /// Initialize the `SoftBoost`.
    pub fn init(sample: &'a Sample) -> Self {
        let n_examples = sample.shape().0;
        assert_ne!(n_examples, 0);

        SoftBoost {
            sample,

            gamma_hat: 1f64,
            tolerance: DEFAULT_TOLERANCE,
            nu: DEFAULT_CAPPING,

            solver: SoftBoostSolver::new(sample),

            dist: Vec::new(),
            weights: Vec::new(),
            numerical_stop_reason: None,
            hypotheses: Vec::new(),

            max_iter: usize::MAX,
            terminated: usize::MAX,
        }
    }

    /// Numerical reason for early termination, if any. This is not convergence.
    /// The final model is fitted to the hypotheses collected before stopping.
    pub fn numerical_stop_reason(&self) -> Option<&str> {
        self.numerical_stop_reason.as_deref()
    }

    /// Set the capping parameter.
    ///
    /// Time complexity: `O(1)`.
    #[inline(always)]
    pub fn nu(mut self, nu: f64) -> Self {
        let n_examples = self.sample.shape().0;
        checkers::capping_parameter(nu, n_examples);

        self.nu = nu;
        self
    }

    /// Set the tolerance parameter.
    ///
    /// Time complexity: `O(1)`.
    #[inline(always)]
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        assert!(
            tolerance.is_finite() && tolerance > 0.0,
            "SoftBoost tolerance must be finite and positive"
        );
        self.tolerance = tolerance;
        self
    }

    fn initialize_solver(&mut self) {
        self.solver.initialize(self.nu);
    }

    /// Iteration bound for the soft-margin objective gap (Theorem 2 in
    /// Warmuth, Glocer and Raetsch, 2007), not classification error.
    ///
    /// Time complexity: `O(1)`.
    pub fn max_loop(&mut self) -> usize {
        let m = self.sample.shape().0 as f64;

        let ln_m = (m / self.nu).ln();
        // Even at nu=m, obtain one hypothesis before forming the output.
        ((2f64 * ln_m / self.tolerance.powi(2)).ceil() as usize).max(1)
    }
}

impl<H> SoftBoost<'_, H>
where
    H: Classifier,
{
    /// Updates `self.dist`
    /// Returns `None` if the stopping criterion satisfied.
    fn update_params_mut(&mut self) -> Option<()> {
        match self
            .solver
            .solve(self.gamma_hat, self.tolerance, &self.hypotheses[..])
        {
            Ok(Some(())) => {
                self.dist = self.solver.distribution_on_examples();
                if self.dist.iter().any(|&d| d == 0.0) {
                    return None;
                }
                Some(())
            }
            Ok(None) => None,
            Err(reason) => {
                eprintln!(
                    "[WARN] SoftBoost/TotalBoost stopped early: {reason}. Returning a model from collected hypotheses; convergence is not certified."
                );
                self.numerical_stop_reason = Some(reason);
                None
            }
        }
    }
}

impl<H> Booster<H> for SoftBoost<'_, H>
where
    H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;

    fn name(&self) -> &str {
        "SoftBoost"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_examples, n_feature) = self.sample.shape();
        let ratio = self.nu * 100f64 / n_examples as f64;
        let nu = helpers::format_unit(self.nu);
        let info = Vec::from([
            ("# of examples", format!("{n_examples}")),
            ("# of features", format!("{n_feature}")),
            ("Tolerance", format!("{}", self.tolerance)),
            ("Max iteration", format!("{}", self.max_iter)),
            ("Capping (outliers)", format!("{nu} ({ratio: >7.3} %)")),
        ]);
        Some(info)
    }

    fn preprocess(&mut self) {
        self.sample.is_valid_binary_instance();
        let n_examples = self.sample.shape().0;

        let uni = 1.0 / n_examples as f64;

        self.dist = vec![uni; n_examples];

        self.max_iter = self.max_loop();
        self.terminated = self.max_iter;
        self.hypotheses = Vec::new();

        self.gamma_hat = 1.0;
        self.numerical_stop_reason = None;
        self.initialize_solver();
    }

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize) -> ControlFlow<usize>
    where
        W: WeakLearner<Hypothesis = H>,
    {
        if self.max_iter < iteration {
            return ControlFlow::Break(self.max_iter);
        }

        // Receive a hypothesis from the base learner
        let h = weak_learner.produce(self.sample, &self.dist);

        let edge = helpers::edge(self.sample, &self.dist, &h);
        self.gamma_hat = self.gamma_hat.min(edge);

        self.hypotheses.push(h);

        // Update the parameters
        if self.update_params_mut().is_none() {
            self.terminated = iteration;
            return ControlFlow::Break(self.terminated);
        }

        ControlFlow::Continue(())
    }

    fn postprocess(&mut self) -> Self::Output {
        let (_, weights) = soft_margin_optimization(self.nu, self.sample, &self.hypotheses[..]);
        self.weights = weights;
        WeightedMajority::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}

impl<H> CurrentHypothesis for SoftBoost<'_, H>
where
    H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        let (_, weights) = soft_margin_optimization(self.nu, self.sample, &self.hypotheses[..]);
        WeightedMajority::from_slices(&weights[..], &self.hypotheses[..])
    }
}

#[cfg(test)]
mod endpoint_tests {
    use super::*;

    #[derive(Clone)]
    struct Oracle;
    impl Classifier for Oracle {
        fn confidence(&self, sample: &Sample, row: usize) -> f64 {
            [1.0, 0.0, -1.0][row] * sample.target()[row]
        }
    }
    impl WeakLearner for Oracle {
        type Hypothesis = Self;
        fn produce(&self, _: &Sample, dist: &[f64]) -> Self {
            assert!(dist.iter().all(|d| (*d - 1.0 / 3.0).abs() < 1e-12));
            Self
        }
    }
    #[test]
    fn full_capping_obtains_a_hypothesis_before_stopping() {
        let sample = Sample::dummy(3);
        let mut booster = SoftBoost::init(&sample).nu(3.0);
        let output = booster.run(&Oracle);
        assert_eq!(output.hypotheses.len(), 1);
        assert!((output.weights[0] - 1.0).abs() < 1e-7);
        assert!(output.confidence(&sample, 0).is_finite());
    }
}
