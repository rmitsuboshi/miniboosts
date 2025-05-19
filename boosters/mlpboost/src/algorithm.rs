//! This file defines `MlpBoost` based on the paper
//! [Boosting as Frank-Wolfe](https://arxiv.org/abs/2209.10831).
//! by Mitsuboshi et al.
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
use hypotheses::{
    RefWeightedMajority,
    WeightedMajority,
};
use optimization::{
    ColumnGeneration,
    ErlpSoftMarginObjective,
    FrankWolfe,
    FwUpdateRule,
    ObjectiveFunction,
    StepSize,
    soft_margin_optimization,
};
use logging::CurrentHypothesis;
use corrective_erlpboost::CorrErlpFwObjective;

use std::ops::ControlFlow;

pub struct MlpBoost<'a, F> {
    // Training sample
    sample: &'a Sample,

    // Tolerance parameter
    half_tolerance: f64,

    // Number of examples
    n_examples: usize,

    // Capping parameter
    nu: f64,

    // Regularization parameter.
    eta: f64,

    // Primary (FW) update
    objective:   ErlpSoftMarginObjective,
    update_rule: FwUpdateRule,
    frank_wolfe: FrankWolfe<CorrErlpFwObjective>,

    // Secondary (LpBoost) update
    lpboost: ColumnGeneration<'a>,

    // Weights on hypotheses
    weights: Vec<f64>,

    dist: Vec<f64>,

    // Hypotheses
    hypotheses: Vec<F>,

    terminated: usize,
    max_iter: usize,

    gamma: f64,
}

impl<'a, F> MlpBoost<'a, F> {
    /// Construct a new instance of `MlpBoost`.
    /// 
    /// Time complexity: `O(1)`.
    pub fn init(sample: &'a Sample) -> Self {
        let n_examples = sample.shape().0;
        assert!(n_examples != 0);

        let half_tolerance = DEFAULT_TOLERANCE / 2f64;
        let nu  = DEFAULT_CAPPING;
        let eta = (n_examples as f64 / nu).ln() / half_tolerance;

        let update_rule = FwUpdateRule::ShortStep;
        let objective = ErlpSoftMarginObjective::new(nu, eta);
        let fw_objective = CorrErlpFwObjective::new(objective.clone());
        let frank_wolfe = FrankWolfe::new(fw_objective, update_rule);

        let lpboost = ColumnGeneration::new(&sample);

        Self {
            sample,

            half_tolerance,
            n_examples,
            nu,
            eta,

            objective,
            update_rule,
            frank_wolfe,

            lpboost,

            weights:    Vec::new(),
            hypotheses: Vec::new(),
            dist:       Vec::new(),

            terminated: usize::MAX,
            max_iter:   usize::MAX,

            gamma: 1f64,
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

    /// Set the Frank-Wolfe rule.
    /// See [`FWType`].
    /// 
    /// Time complexity: `O(1)`.
    pub fn update_rule(mut self, update_rule: FwUpdateRule) -> Self {
        self.update_rule = update_rule;
        self
    }

    /// Set the tolerance parameter.
    /// 
    /// Time complexity: `O(1)`.
    #[inline(always)]
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.half_tolerance = tolerance / 2f64;
        self
    }

    /// Set the regularization parameter.
    /// 
    /// Time complexity: `O(1)`.
    #[inline(always)]
    fn eta(&mut self) {
        let ln_m = (self.n_examples as f64 / self.nu).ln();
        self.eta = ln_m / self.half_tolerance;
    }

    fn initialize_objective(&mut self) {
        self.objective = ErlpSoftMarginObjective::new(self.nu, self.eta);
    }

    /// Initialize the LP solver.
    /// 
    /// Time complexity: `O( # of training examples )`.
    fn initialize_solver(&mut self) {
        self.initialize_objective();

        let objective = CorrErlpFwObjective::new(self.objective.clone());
        self.frank_wolfe = FrankWolfe::new(objective, self.update_rule);

        self.lpboost.initialize(self.n_examples, self.nu);
    }

    /// Returns the maximum iterations 
    /// to obtain the solution with accuracy `self.half_tolerance`.
    /// 
    /// Time complexity: `O(1)`.
    pub fn max_loop(&self) -> usize {
        let ln_m = (self.n_examples as f64 / self.nu).ln();
        (8f64 * ln_m / self.half_tolerance.powi(2)).ceil() as usize
    }

    /// Returns the terminated iteration.
    /// This method returns `0` before the boosting step.
    /// 
    /// Time complexity: `O(1)`.
    pub fn terminated(&self) -> usize {
        self.terminated
    }
}

impl<F> MlpBoost<'_, F>
    where F: Classifier,
{
    /// Returns the smoothed objective value 
    /// `-f*` at the current weighting `self.weights`.
    /// 
    /// ```text
    /// - max [ - d^T Aw - sum_i [ di ln( di ) ] ]
    /// s.t. sum_i di = 1, 0 <= di <= 1 / self.nu, for all i <= m.
    ///  ^
    ///  |
    ///  v
    /// min [ d^T Aw + sum_i [ di ln( di ) ] ]
    /// s.t. sum_i di = 1, 0 <= di <= 1 / self.nu, for all i <= m.
    /// ```
    /// 
    /// Time complexity: `O( # of training examples )`.
    fn objval(&self, weights: &[f64]) -> f64 {
        if weights.is_empty() { return -1f64; }
        let f = RefWeightedMajority::new(&weights[..], &self.hypotheses[..]);
        let neg_margins = helpers::margins(self.sample, &f)
            .map(|yf| -yf)
            .collect::<Vec<_>>();
        self.objective.objective_value(&neg_margins[..])
    }

    /// Updates weight on hypotheses and `self.dist` in this order.
    fn update_distribution_mut(&mut self) {
        let f = RefWeightedMajority::new(
            &self.weights[..],
            &self.hypotheses[..],
        );
        let neg_margins = helpers::margins(self.sample, &f)
            .map(|yf| -yf)
            .collect::<Vec<_>>();
        self.dist = self.objective.gradient(&neg_margins[..]);
    }

    fn update_lpboost(&mut self, h: &F) -> (f64, Vec<f64>) {
        self.lpboost.solve(h);
        let weight = self.lpboost.weights_on_hypotheses();
        let objval = self.objval(&weight[..]);
        (objval, weight)
    }

    fn update_frank_wolfe(&mut self, h: F) -> (f64, Vec<f64>) {
        if self.weights.is_empty() {
            self.hypotheses.push(h);
            let weight = vec![1f64];
            let objval = self.objval(&weight[..]);
            return (objval, weight);
        }

        let cur_margins = {
            let f = RefWeightedMajority::new(
                &self.weights,
                &self.hypotheses[..],
            );
            helpers::margins(self.sample, &f)
                .map(|yf| -yf)
                .collect::<Vec<_>>()
        };
        let new_margins = helpers::margins(self.sample, &h)
            .map(|yh| -yh)
            .collect::<Vec<_>>();

        let mut weights = self.weights.clone();

        let stepsize = self.frank_wolfe.get_stepsize_mut(
            &cur_margins[..],
            &new_margins[..],
        );
        match stepsize {
            StepSize::Normal(stepsize) => {
                weights.iter_mut().for_each(|w| { *w *= 1f64 - stepsize; });
                weights.push(stepsize);
                self.hypotheses.push(h);
            },
            StepSize::BpfwMoveWeights { .. } => {
                unimplemented!()
            // StepSize::BpfwMoveWeights { stepsize, .. } => {
            //     let edges = self.hypotheses.iter()
            //         .map(|h| helpers::edge(self.sample, &self.dist[..], h))
            //         .collect::<Vec<_>>();
            //     let n = self.weights.len();
            //     let (bst, wst) = {
            //         let mut ix = (0..n).collect::<Vec<_>>();
            //         ix.sort_by(|&i, &j| edges[i].partial_cmp(&edges[j]).unwrap());
            //         let wst = ix.iter()
            //             .copied()
            //             .filter(|&i| self.weights[i] > 0f64)
            //             .next()
            //             .expect("failed to get a worst hypothesis");
            //         let bst = ix.iter()
            //             .copied()
            //             .rev()
            //             .filter(|&i| self.weights[i] > 0f64)
            //             .next()
            //             .expect("failed to get a best hypothesis");
            //         (bst, wst)
            //     };
            //     self.weights[bst] += stepsize;
            //     self.weights[wst] -= stepsize;
            },
        }
        checkers::capped_simplex_condition(&weights[..], 1f64);

        let objval = self.objval(&weights[..]);
        (objval, weights)
    }
}

impl<F> Booster<F> for MlpBoost<'_, F>
    where F: Classifier + Clone + PartialEq,
{
    type Output = WeightedMajority<F>;

    fn name(&self) -> &str {
        "MlpBoost"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_examples, n_feature) = self.sample.shape();
        let ratio = self.nu * 100f64 / n_examples as f64;
        let nu = helpers::format_unit(self.nu);
        let info = Vec::from([
            ("# of examples", format!("{n_examples}")),
            ("# of features", format!("{n_feature}")),
            ("Tolerance", format!("{}", 2f64 * self.half_tolerance)),
            ("Max iteration", format!("{}", self.max_iter)),
            ("Capping (outliers)", format!("{nu} ({ratio: >7.3} %)")),
            ("Primary", format!("{}", self.frank_wolfe.current_update_rule())),
            ("Secondary", "LpBoost".to_string())
        ]);
        Some(info)
    }

    fn preprocess(&mut self) {
        self.sample.is_valid_binary_instance();
        self.n_examples = self.sample.shape().0;

        self.eta();
        self.initialize_solver();

        self.max_iter = self.max_loop();
        self.terminated = self.max_iter;

        self.hypotheses = Vec::new();
        self.weights = Vec::new();
        self.dist = vec![1f64 / self.n_examples as f64; self.n_examples];

        // Upper-bound of the optimal `edge`.
        self.gamma = 1f64;
    }

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize)
        -> ControlFlow<usize>
        where W: WeakLearner<Hypothesis = F>,
    {

        if self.max_iter < iteration {
            return ControlFlow::Break(self.max_iter);
        }
        let h = weak_learner.produce(self.sample, &self.dist[..]);

        let edge = helpers::edge(self.sample, &self.dist[..], &h);

        // Update the estimation of `edge`.
        self.gamma = self.gamma.min(edge);

        // Compute the smoothed objective value `-f*`.
        let objval = self.objval(&self.weights[..]);

        // If the difference between `gamma` and `objval` is
        // lower than `self.half_tolerance`,
        // optimality guaranteed with the precision.
        if self.gamma - objval <= self.half_tolerance {
            self.terminated = iteration;
            return ControlFlow::Break(self.terminated);
        }

        let (objval_lp, weight_lp) = self.update_lpboost(&h);
        let (objval_fw, weight_fw) = self.update_frank_wolfe(h);

        self.weights = if objval_lp > objval_fw {
            weight_lp
        } else {
            weight_fw
        };

        self.update_distribution_mut();

        // DEBUG
        checkers::capped_simplex_condition(&self.weights[..], 1f64);

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

impl<H> CurrentHypothesis for MlpBoost<'_, H>
    where H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        WeightedMajority::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}


