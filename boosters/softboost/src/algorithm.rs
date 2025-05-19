//! This file defines `SoftBoost` based on the paper
//! "Boosting Algorithms for Maximizing the Soft Margin"
//! by Warmuth et al.
//! 
use crate::solver::SoftBoostSolver;
use miniboosts_core::{
    Sample,
    Booster,
    WeakLearner,
    Classifier,
    tools::helpers,
    tools::checkers,
    constants::{
        DEFAULT_CAPPING,
        DEFAULT_TOLERANCE,
    },
};
use hypotheses::WeightedMajority;
use optimization::soft_margin_optimization;
use logging::CurrentHypothesis;

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
}

impl<'a, H> SoftBoost<'a, H>
    where H: Classifier
{
    /// Initialize the `SoftBoost`.
    pub fn init(sample: &'a Sample) -> Self {
        let n_examples = sample.shape().0;
        assert_ne!(n_examples, 0);

        SoftBoost {
            sample,

            gamma_hat:  1f64,
            tolerance:  DEFAULT_TOLERANCE,
            nu:         DEFAULT_CAPPING,

            solver:     SoftBoostSolver::new(sample),

            dist:       Vec::new(),
            weights:    Vec::new(),
            hypotheses: Vec::new(),

            max_iter:   usize::MAX,
            terminated: usize::MAX,
        }
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
        self.tolerance = tolerance;
        self
    }

    fn initialize_solver(&mut self) {
        self.solver.initialize(self.nu);
    }

    /// `max_loop` returns the maximum iteration
    /// of the Adaboost to find a combined hypothesis
    /// that has error at most `tolerance`.
    /// 
    /// Time complexity: `O(1)`.
    pub fn max_loop(&mut self) -> usize {
        let m = self.sample.shape().0 as f64;

        let ln_m = (m / self.nu).ln();
        (2f64 * ln_m / self.tolerance.powi(2)).ceil() as usize
    }
}

impl<H> SoftBoost<'_, H>
    where H: Classifier,
{
    /// Updates `self.dist`
    /// Returns `None` if the stopping criterion satisfied.
    fn update_params_mut(&mut self) -> Option<()> {
        let result = self.solver.solve(
            self.gamma_hat,
            self.tolerance,
            &self.hypotheses[..],
        );
        if let Some(_) = result {
            self.dist = self.solver.distribution_on_examples();
            if self.dist.iter().any(|&d| d == 0f64) {
                return None;
            }
        }
        result
    }
}

impl<H> Booster<H> for SoftBoost<'_, H>
    where H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;

    fn name(&self) -> &str { "SoftBoost" }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_examples, n_feature) = self.sample.shape();
        let ratio = self.nu * 100f64 / n_examples as f64;
        let nu = helpers::format_unit(self.nu);
        let info = Vec::from([
            ("# of examples", format!("{n_examples}")),
            ("# of features", format!("{n_feature}")),
            ("Tolerance", format!("{}", self.tolerance)),
            ("Max iteration", format!("{}", self.max_iter)),
            ("Capping (outliers)", format!("{nu} ({ratio: >7.3} %)"))
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
        self.initialize_solver();
    }

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize)
        -> ControlFlow<usize>
        where W: WeakLearner<Hypothesis = H>,
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
        let (_, weights) = soft_margin_optimization(
            self.nu,
            self.sample,
            &self.hypotheses[..],
        );
        self.weights = weights;
        WeightedMajority::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}

impl<H> CurrentHypothesis for SoftBoost<'_, H>
    where H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        let (_, weights) = soft_margin_optimization(
            self.nu,
            self.sample,
            &self.hypotheses[..],
        );
        WeightedMajority::from_slices(&weights[..], &self.hypotheses[..])
    }
}

