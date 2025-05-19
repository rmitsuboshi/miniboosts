//! This file defines `CorrectiveErlpBoost` based on the paper
//! "On the Equivalence of Weak Learnability and Linaer Separability:
//!     New Relaxations and Efficient Boosting Algorithms"
//! by Shai Shalev-Shwartz and Yoram Singer.
//! I named this algorithm `CorrectiveErlpBoost`
//! since it is referred as `the Corrective version of CorrectiveErlpBoost`
//! in "Entropy Regularized LPBoost" by Warmuth et al.
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
    FrankWolfe,
    FwUpdateRule,
    ObjectiveFunction,
    ErlpSoftMarginObjective,
    StepSize,
};
use logging::CurrentHypothesis;

use std::ops::ControlFlow;

pub struct CorrectiveErlpBoost<'a, H> {
    // Training sample
    sample: &'a Sample,

    dist: Vec<f64>,
    // A regularization parameter defined in the paper
    eta: f64,

    half_tolerance: f64,
    nu: f64,

    objective: ErlpSoftMarginObjective,
    frank_wolfe: FrankWolfe<CorrErlpFwObjective>,
    update_rule: FwUpdateRule,
    weights: Vec<f64>,
    hypotheses: Vec<H>,

    max_iter: usize,
    terminated: usize,
}

impl<'a, H> CorrectiveErlpBoost<'a, H> {
    /// Construct a new instance of `CorrectiveErlpBoost`.
    /// 
    /// Time complexity: `O(1)`.
    pub fn init(sample: &'a Sample) -> Self {
        let n_examples = sample.shape().0;

        // Set tolerance, sub_tolerance
        let half_tolerance = DEFAULT_TOLERANCE / 2f64;

        // Set regularization parameter
        let nu = DEFAULT_CAPPING;
        let eta = (n_examples as f64 / nu).ln() / half_tolerance;

        let objective = ErlpSoftMarginObjective::new(nu, eta);
        let update_rule = FwUpdateRule::ShortStep;
        let fw_objective = CorrErlpFwObjective::new(objective.clone());
        let frank_wolfe = FrankWolfe::new(fw_objective, update_rule);

        Self {
            sample,

            dist: Vec::new(),
            half_tolerance,
            eta,
            nu,

            objective,
            frank_wolfe,
            update_rule,

            weights: Vec::new(),
            hypotheses: Vec::new(),
            max_iter: usize::MAX,
            terminated: usize::MAX,
        }
    }

    /// This method updates the capping parameter.
    /// 
    /// Time complexity: `O(1)`.
    pub fn nu(mut self, nu: f64) -> Self {
        let n_examples = self.sample.shape().0;
        checkers::capping_parameter(nu, n_examples);
        self.nu = nu;
        self
    }

    /// Update tolerance parameter `half_tolerance`.
    /// 
    /// Time complexity: `O(1)`.
    #[inline(always)]
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.half_tolerance = tolerance / 2.0;
        self
    }

    /// Set an update strategy for the Frank-Wolfe algorithm.
    #[inline(always)]
    pub fn update_rule(mut self, update_rule: FwUpdateRule) -> Self {
        self.update_rule = update_rule;
        self
    }

    /// Update regularization parameter.
    /// (the regularization parameter on `self.tolerance` and `self.nu`.)
    /// 
    /// Time complexity: `O(1)`.
    #[inline(always)]
    fn regularization_param(&mut self) {
        let m = self.dist.len() as f64;
        let ln_part = (m / self.nu).ln();
        self.eta = ln_part / self.half_tolerance;
    }

    /// returns the maximum iteration of the CorrectiveErlpBoost
    /// to find a combined hypothesis that has error at most `tolerance`.
    /// 
    /// Time complexity: `O(1)`.
    pub fn max_loop(&mut self) -> usize {

        let m = self.dist.len() as f64;

        let ln_m = (m / self.nu).ln();
        let max_iter = 8.0 * ln_m / self.half_tolerance.powi(2);

        max_iter.ceil() as usize
    }

    fn initialize_objective(&mut self) {
        self.objective = ErlpSoftMarginObjective::new(self.nu, self.eta);
    }

    /// Initializes the Frank-Wolfe algorithm.
    fn initialize_solver(&mut self) {
        self.initialize_objective();
        let objective = CorrErlpFwObjective::new(self.objective.clone());
        self.frank_wolfe = FrankWolfe::new(objective, self.update_rule);
    }
}

impl<H> CorrectiveErlpBoost<'_, H>
    where H: Classifier + PartialEq,
{
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

    fn update_params_mut(
        &mut self,
        h: H,
        mut cur_margins: Vec<f64>,
        mut new_margins: Vec<f64>,
    )
    {
        if self.weights.is_empty() {
            self.weights.push(1f64);
            self.hypotheses.push(h);
            self.update_distribution_mut();
            return;
        }

        cur_margins.iter_mut()
            .for_each(|yf| { *yf *= -1f64; });

        new_margins.iter_mut()
            .for_each(|yh| { *yh *= -1f64; });

        let stepsize = self.frank_wolfe.get_stepsize_mut(
            &cur_margins[..],
            &new_margins[..],
        );
        match stepsize {
            StepSize::Normal(stepsize) => {
                self.weights.iter_mut()
                    .for_each(|w| { *w *= 1f64 - stepsize; });
                self.weights.push(stepsize);
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
        checkers::capped_simplex_condition(&self.weights[..], 1f64);

        self.update_distribution_mut();
    }
}

impl<H> Booster<H> for CorrectiveErlpBoost<'_, H>
    where H: Classifier + Clone + PartialEq + std::fmt::Debug,
{
    type Output = WeightedMajority<H>;

    fn name(&self) -> &str {
        "Corrective ErlpBoost"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_examples, n_feature) = self.sample.shape();
        let ratio = self.nu / n_examples as f64;
        let nu = helpers::format_unit(self.nu);
        let fw = self.frank_wolfe.current_update_rule();
        let info = Vec::from([
            ("# of examples", format!("{n_examples}")),
            ("# of features", format!("{n_feature}")),
            ("Tolerance", format!("{}", 2f64 * self.half_tolerance)),
            ("Max iteration", format!("{}", self.max_iter)),
            ("Capping (outliers)", format!("{nu} ({ratio: >7.3} %)")),
            ("Frank-Wolfe", format!("{fw}")),
        ]);
        Some(info)
    }

    fn preprocess(&mut self) {
        self.sample.is_valid_binary_instance();
        let n_examples = self.sample.shape().0;
        let uni = 1f64 / n_examples as f64;

        self.dist = vec![uni; n_examples];

        self.regularization_param();
        self.max_iter = self.max_loop();
        self.terminated = self.max_iter;

        // self.classifiers = Vec::new();
        self.weights = Vec::new();
        self.hypotheses = Vec::new();

        self.initialize_solver();
    }

    fn boost<W>(
        &mut self,
        weak_learner: &W,
        iteration: usize,
    ) -> ControlFlow<usize>
    where W: WeakLearner<Hypothesis = H>,
    {
        if self.max_iter < iteration {
            return ControlFlow::Break(self.max_iter);
        }

        // Receive a hypothesis from the base learner
        let h = weak_learner.produce(self.sample, &self.dist);

        let cur_margins = {
            let f = RefWeightedMajority::new(
                &self.weights,
                &self.hypotheses[..],
            );
            helpers::margins(self.sample, &f).collect::<Vec<_>>()
        };
        let new_margins = helpers::margins(self.sample, &h)
            .collect::<Vec<_>>();

        let duality_gap = {
            let e1 = helpers::inner_product(&new_margins[..], &self.dist[..]);
            let e2 = helpers::inner_product(&cur_margins[..], &self.dist[..]);
            e1 - e2
        };

        // Update the parameters
        if duality_gap <= self.half_tolerance {
            self.terminated = iteration;
            return ControlFlow::Break(iteration);
        }

        self.update_params_mut(h, cur_margins, new_margins);

        ControlFlow::Continue(())
    }

    fn postprocess(&mut self) -> Self::Output {
        WeightedMajority::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}

impl<H> CurrentHypothesis for CorrectiveErlpBoost<'_, H>
    where H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        WeightedMajority::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}

pub struct CorrErlpFwObjective(ErlpSoftMarginObjective);

impl CorrErlpFwObjective {
    pub fn new(objective: ErlpSoftMarginObjective) -> Self {
        Self(objective)
    }
}

impl ObjectiveFunction for CorrErlpFwObjective {
    fn name(&self) -> &str {
        "The dual objective for ErlpBoost (minimization form)"
    }

    fn smooth(&self) -> (f64, f64) {
        self.0.smooth()
    }

    /// returns the objective value `f^*(θ)` at given point `θ = -Aw.`
    fn objective_value(&self, point: &[f64]) -> f64 {
        - self.0.objective_value(point)
    }

    /// returns the gradient `∇f^*(θ)` at given point `θ = -Aw.`
    fn gradient(&self, point: &[f64]) -> Vec<f64> {
        self.0.gradient(point)
            .into_iter()
            .collect()
    }
}

