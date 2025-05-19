//! This file defines `LpBoost` based on the paper
//! ``Boosting algorithms for Maximizing the Soft Margin''
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
        DEFAULT_TOLERANCE,
        DEFAULT_CAPPING,
    },
};
use hypotheses::WeightedMajority;
use optimization::{
    ColumnGeneration,
    soft_margin_optimization,
};
use logging::CurrentHypothesis;

use std::ops::ControlFlow;

pub struct LpBoost<'a, F> {
    // Training sample
    sample: &'a Sample,

    // Distribution over examples
    dist: Vec<f64>,

    // min-max edge of the new hypothesis
    gamma_hat: f64,

    // Tolerance parameter
    tolerance: f64,

    // Number of examples
    n_examples: usize,

    // Capping parameter
    nu: f64,

    solver: ColumnGeneration<'a>,

    hypotheses: Vec<F>,
    weights: Vec<f64>,

    terminated: usize,
}

impl<'a, F> LpBoost<'a, F>
    where F: Classifier
{
    /// Constructs a new instance of `LpBoost`.
    /// 
    /// Time complexity: `O(1)`.
    pub fn init(sample: &'a Sample) -> Self {
        let n_examples = sample.shape().0;

        Self {
            sample,

            dist: Vec::new(),
            gamma_hat: 1.0,
            tolerance: DEFAULT_TOLERANCE,
            n_examples,
            nu: DEFAULT_CAPPING,
            solver: ColumnGeneration::new(&sample),

            hypotheses: Vec::new(),
            weights: Vec::new(),

            terminated: usize::MAX,
        }
    }

    /// This method updates the capping parameter.
    /// This parameter must be in `[1, # of training examples]`.
    /// 
    /// Time complexity: `O(1)`.
    pub fn nu(mut self, nu: f64) -> Self {
        checkers::capping_parameter(nu, self.n_examples);
        self.nu = nu;

        self
    }

    /// Set the tolerance parameter.
    /// LpBoost guarantees the `tolerance`-approximate solution to
    /// the soft margin optimization.  
    /// Default value is `0.01`.
    /// 
    /// Time complexity: `O(1)`.
    #[inline(always)]
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Returns the terminated iteration.
    /// This method returns `usize::MAX` before the boosting step.
    /// 
    /// Time complexity: `O(1)`.
    #[inline(always)]
    pub fn terminated(&self) -> usize {
        self.terminated
    }

    /// This method updates `self.dist` and `self.gamma_hat`
    /// by solving a linear program
    /// over the hypotheses obtained in past rounds.
    /// 
    /// Time complexity depends on the LP solver.
    #[inline(always)]
    fn update_distribution_mut(&mut self, h: &F) -> f64 {
        self.solver.solve(h);
        self.dist = self.solver.distribution_on_examples();

        self.solver.optimal_value()
    }
}

impl<F> Booster<F> for LpBoost<'_, F>
    where F: Classifier + Clone,
{
    type Output = WeightedMajority<F>;

    fn name(&self) -> &str { "LpBoost" }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_examples, n_feature) = self.sample.shape();
        let ratio = self.nu * 100f64 / n_examples as f64;
        let nu = helpers::format_unit(self.nu);
        let info = Vec::from([
            ("# of examples", format!("{n_examples}")),
            ("# of features", format!("{n_feature}")),
            ("Tolerance", format!("{}", self.tolerance)),
            ("Max iteration", "-".to_string()),
            ("Capping (outliers)", format!("{nu} ({ratio: >7.3} %)"))
        ]);
        Some(info)
    }

    fn preprocess(&mut self) {
        self.sample.is_valid_binary_instance();
        let n_examples = self.sample.shape().0;
        let uni = 1.0_f64 / self.n_examples as f64;

        self.n_examples = n_examples;
        self.dist = vec![uni; n_examples];
        self.gamma_hat = 1.0;
        self.solver.initialize(self.n_examples, self.nu);
        self.hypotheses = Vec::new();
        self.terminated = usize::MAX;
    }

    fn boost<W>(
        &mut self,
        weak_learner: &W,
        iteration: usize,
    ) -> ControlFlow<usize>
        where W: WeakLearner<Hypothesis = F>,
    {
        let h = weak_learner.produce(self.sample, &self.dist);

        // Each element in `margins` is the product of
        // the predicted vector and the correct vector
        let ghat = helpers::edge(self.sample, &self.dist[..], &h);

        self.gamma_hat = ghat.min(self.gamma_hat);

        let gamma_star = self.update_distribution_mut(&h);
        self.hypotheses.push(h);
        assert!((-1f64..=1f64).contains(&gamma_star));

        if gamma_star >= self.gamma_hat - self.tolerance {
            self.terminated = self.hypotheses.len();
            return ControlFlow::Break(iteration);
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

impl<H> CurrentHypothesis for LpBoost<'_, H>
    where H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        let weights = self.solver.weights_on_hypotheses();

        WeightedMajority::from_slices(&weights[..], &self.hypotheses[..])
    }
}

