//! This file defines `ErlpBoost` based on the paper
//! "Entropy Regularized LPBoost"
//! by Warmuth et al.
//! 
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
use optimization::{
    RowGeneration,
    soft_margin_optimization,
    EntropyRegularizedMaxEdge,
};
use logging::CurrentHypothesis;

use std::ops::ControlFlow;

pub struct ErlpBoost<'a, H> {
    // Training sample
    sample: &'a Sample,

    // `gamma_hat` corresponds to $\min_{q=1, .., t} P^q (d^{q-1})$
    gamma_hat: f64,

    // `gamma_star` corresponds to $P^{t-1} (d^{t-1})$
    gamma_star: f64,
    // regularization parameter defined in the paper
    eta: f64,

    half_tolerance: f64,

    solver: RowGeneration<'a, EntropyRegularizedMaxEdge>,

    hypotheses: Vec<H>,
    weights: Vec<f64>,
    dist: Vec<f64>,

    // an accuracy parameter for the sub-problems
    n_sample: usize,
    nu: f64,

    terminated: usize,

    max_iter: usize,
}

impl<'a, H> ErlpBoost<'a, H> {
    /// Constructs a new instance of `ErlpBoost`.
    /// 
    /// Time complexity: `O(1)`.
    pub fn init(sample: &'a Sample) -> Self {
        let n_sample = sample.shape().0;
        assert!(n_sample != 0);

        // Compute $\ln(n_sample)$ in advance
        let ln_m = (n_sample as f64).ln();

        // Set tolerance
        let half_tolerance = DEFAULT_TOLERANCE / 2f64;

        // Set regularization parameter
        let eta = 0.5f64.max(ln_m / half_tolerance);

        // Set gamma_hat and gamma_star
        let gamma_hat  = 1f64;
        let gamma_star = f64::MIN;

        let objective = EntropyRegularizedMaxEdge::new(eta);

        Self {
            sample,

            dist: Vec::new(),
            gamma_hat,
            gamma_star,
            eta,
            half_tolerance,
            solver: RowGeneration::new(sample, objective),

            hypotheses: Vec::new(),
            weights: Vec::new(),

            n_sample,
            nu: DEFAULT_CAPPING,

            terminated: usize::MAX,
            max_iter: usize::MAX,
        }
    }

    /// Initialize the sequential quadratic programming solver.
    fn init_solver(&mut self) {
        let objective = EntropyRegularizedMaxEdge::new(self.eta);
        self.solver.initialize(self.nu, objective);
    }

    /// Updates the capping parameter.
    /// Time complexity: `O(1)`.
    pub fn nu(mut self, nu: f64) -> Self {
        checkers::capping_parameter(nu, self.n_sample);
        self.nu = nu;

        self
    }

    /// Returns the break iteration.
    /// Time complexity: `O(1)`.
    #[inline(always)]
    pub fn terminated(&self) -> usize {
        self.terminated
    }

    /// Sets the tolerance parameter.
    /// 
    /// Time complexity: `O(1)`.
    #[inline(always)]
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.half_tolerance = tolerance / 2f64;
        self
    }

    /// Sets `self.eta.`
    /// Time complexity: `O(1)`.
    #[inline(always)]
    fn regularization_param(&mut self) {
        let ln_m = (self.n_sample as f64 / self.nu).ln();
        self.eta = 0.5f64.max(ln_m / self.half_tolerance);
    }

    /// `max_loop` returns the maximum iteration
    /// of the Adaboost to find a combined hypothesis
    /// that has error at most `tolerance`.
    /// 
    /// Time complexity: `O(1)`.
    fn max_loop(&mut self) -> usize {
        let n_sample = self.n_sample as f64;

        let max_iter_1 = 4f64 / self.half_tolerance;

        let max_iter_2 = {
            let ln_m = (n_sample / self.nu).ln();
            8f64 * ln_m / self.half_tolerance.powi(2)
        };

        max_iter_1.max(max_iter_2).ceil() as usize
    }
}

impl<H> ErlpBoost<'_, H>
    where H: Classifier
{
    /// Computes the current objective value and set it to `self.gamma_hat`.
    /// Time complexity: `O(m)`, where `m` is the number of training examples.
    #[inline]
    fn update_gamma_hat_mut(&mut self, h: &H) {
        let edge = helpers::edge(self.sample, &self.dist[..], h);
        let entropy = helpers::entropy_from_uni_distribution(&self.dist[..]);

        let obj_val = edge + (entropy / self.eta);

        self.gamma_hat = self.gamma_hat.min(obj_val);
    }

    /// Solve the entropy regularized edge minimization problem 
    /// to update `self.dist.`
    fn update_distribution_mut(&mut self) {
        self.solver.solve(&self.hypotheses[..]);
        self.dist = self.solver.distribution_on_examples();
    }
}

impl<H> Booster<H> for ErlpBoost<'_, H>
    where H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;

    fn name(&self) -> &str { "ErlpBoost" }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_sample, n_feature) = self.sample.shape();
        let ratio = self.nu * 100f64 / n_sample as f64;
        let nu = helpers::format_unit(self.nu);
        let info = Vec::from([
            ("# of examples", format!("{n_sample}")),
            ("# of features", format!("{n_feature}")),
            ("Tolerance", format!("{}", 2f64 * self.half_tolerance)),
            ("Max iteration", format!("{}", self.max_iter)),
            ("Capping (outliers)", format!("{nu} ({ratio: >7.3} %)"))
        ]);
        Some(info)
    }

    fn preprocess(&mut self) {
        self.sample.is_valid_binary_instance();
        let n_sample = self.sample.shape().0;
        let uni = 1.0 / n_sample as f64;

        self.dist = vec![uni; n_sample];

        self.max_iter = self.max_loop();
        self.terminated = self.max_iter;

        self.hypotheses = Vec::new();

        self.gamma_hat = 1.0;
        self.gamma_star = -1.0;

        assert!((0.0..1.0).contains(&self.half_tolerance));
        self.regularization_param();
        self.init_solver();
    }

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize)
        -> ControlFlow<usize>
        where W: WeakLearner<Hypothesis = H>,
    {
        if self.max_iter < iteration {
            println!("reached to the max iteration");
            return ControlFlow::Break(self.max_iter);
        }

        // Receive a hypothesis from the base learner
        let h = weak_learner.produce(self.sample, &self.dist[..]);

        // update `self.gamma_hat`
        self.update_gamma_hat_mut(&h);

        // Check the stopping criterion
        let diff = self.gamma_hat - self.gamma_star;
        if diff <= self.half_tolerance {
            self.terminated = iteration;
            return ControlFlow::Break(iteration);
        }

        // At this point, the stopping criterion is not satisfied.

        // Update the parameters
        self.hypotheses.push(h);
        self.update_distribution_mut();

        // update `self.gamma_star`.
        self.gamma_star = self.solver.optval();
        ControlFlow::Continue(())
    }

    fn postprocess(&mut self) -> Self::Output {
        let (_, weights) = soft_margin_optimization(
            self.nu,
            &self.sample,
            &self.hypotheses[..],
        );
        self.weights = weights;
        WeightedMajority::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}

impl<H> CurrentHypothesis for ErlpBoost<'_, H>
    where H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        let weights = self.solver.weights_on_hypotheses();

        WeightedMajority::from_slices(&weights[..], &self.hypotheses[..])
    }
}

