//! This file defines `MlpBoost` based on the paper
//! [Boosting as Frank-Wolfe](https://arxiv.org/abs/2209.10831).
//! by Mitsuboshi et al.
//!
use corrective_erlpboost::CorrErlpFwObjective;
use hypotheses::{RefWeightedMajority, WeightedMajority};
use miniboosts_core::CurrentHypothesis;
use miniboosts_core::{
    Booster, Classifier, Sample, WeakLearner,
    constants::{DEFAULT_CAPPING, DEFAULT_TOLERANCE},
    tools::checkers,
    tools::helpers,
};
use optimization::{
    ColumnGeneration, ErlpSoftMarginObjective, FrankWolfe, FwUpdateRule, ObjectiveFunction,
    StepSize, soft_margin_optimization,
};

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
    objective: ErlpSoftMarginObjective,
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
        let nu = DEFAULT_CAPPING;
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

            weights: Vec::new(),
            hypotheses: Vec::new(),
            dist: Vec::new(),

            terminated: usize::MAX,
            max_iter: usize::MAX,

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
    /// Supports classic, short-step, and line-search updates.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Panics
    /// Blended-pairwise updates require coefficient-space atoms and are not
    /// supported by this booster's margin-space implementation.
    pub fn update_rule(mut self, update_rule: FwUpdateRule) -> Self {
        assert!(
            !matches!(update_rule, FwUpdateRule::BlendedPairwise),
            "BlendedPairwise is not supported by this booster",
        );
        self.update_rule = update_rule;
        self
    }

    /// Set the soft-margin objective tolerance. The stopping gap uses half
    /// this value; it is not a classification-error threshold.
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
        // At nu=m the feasible distribution is uniform and its relative
        // entropy is zero. Any positive eta represents the same objective.
        self.eta = if ln_m == 0.0 {
            1.0
        } else {
            ln_m / self.half_tolerance
        };
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

    /// Returns the iteration budget for the configured soft-margin objective
    /// tolerance under the weak-learning assumption.
    ///
    /// Time complexity: `O(1)`.
    pub fn max_loop(&self) -> usize {
        let ln_m = (self.n_examples as f64 / self.nu).ln();
        ((8f64 * ln_m / self.half_tolerance.powi(2)).ceil() as usize).max(1)
    }

    /// Returns the terminated iteration.
    /// Returns `usize::MAX` before preprocessing.
    ///
    /// Time complexity: `O(1)`.
    pub fn terminated(&self) -> usize {
        self.terminated
    }
}

impl<F> MlpBoost<'_, F>
where
    F: Classifier,
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
        assert_eq!(weights.len(), self.hypotheses.len());
        if weights.is_empty() {
            return -1f64;
        }
        let f = RefWeightedMajority::new(&weights[..], &self.hypotheses[..]);
        let neg_margins = helpers::margins(self.sample, &f)
            .map(|yf| -yf)
            .collect::<Vec<_>>();
        self.objective.objective_value(&neg_margins[..])
    }

    /// Updates weight on hypotheses and `self.dist` in this order.
    fn update_distribution_mut(&mut self) {
        let f = RefWeightedMajority::new(&self.weights[..], &self.hypotheses[..]);
        let neg_margins = helpers::margins(self.sample, &f)
            .map(|yf| -yf)
            .collect::<Vec<_>>();
        self.dist = self.objective.gradient(&neg_margins[..]);
    }

    fn update_lpboost(&mut self) -> (f64, Vec<f64>) {
        self.lpboost.solve(self.hypotheses.last().unwrap());
        let weight = self.lpboost.weights_on_hypotheses();
        let objval = self.objval(&weight[..]);
        (objval, weight)
    }

    fn update_frank_wolfe(&mut self) -> (f64, Vec<f64>) {
        if self.weights.is_empty() {
            let weight = vec![1f64];
            let objval = self.objval(&weight[..]);
            return (objval, weight);
        }

        let cur_margins = {
            let f = RefWeightedMajority::new(&self.weights, &self.hypotheses[..self.weights.len()]);
            helpers::margins(self.sample, &f)
                .map(|yf| -yf)
                .collect::<Vec<_>>()
        };
        let new_margins = helpers::margins(self.sample, self.hypotheses.last().unwrap())
            .map(|yh| -yh)
            .collect::<Vec<_>>();

        let mut weights = self.weights.clone();

        let stepsize = self
            .frank_wolfe
            .get_stepsize_mut(&cur_margins[..], &new_margins[..]);
        match stepsize {
            StepSize::Normal(stepsize) => {
                weights.iter_mut().for_each(|w| {
                    *w *= 1f64 - stepsize;
                });
                weights.push(stepsize);
            }
            StepSize::BpfwMoveWeights { .. } => {
                unreachable!("BlendedPairwise is rejected by update_rule")
            }
        }
        checkers::capped_simplex_condition(&weights[..], 1f64);

        let objval = self.objval(&weights[..]);
        (objval, weights)
    }
}

impl<F> Booster<F> for MlpBoost<'_, F>
where
    F: Classifier + Clone + PartialEq,
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
            (
                "Primary",
                format!("{}", self.frank_wolfe.current_update_rule()),
            ),
            ("Secondary", "LpBoost".to_string()),
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

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize) -> ControlFlow<usize>
    where
        W: WeakLearner<Hypothesis = F>,
    {
        if self.max_iter < iteration {
            return ControlFlow::Break(self.max_iter);
        }
        let h = weak_learner.produce(self.sample, &self.dist[..]);

        // With nu=m the dual feasible set is the singleton uniform distribution.
        // One oracle call suffices; retain it even when its edge is nonpositive.
        if self.nu == self.sample.shape().0 as f64 {
            self.hypotheses.push(h);
            self.weights = vec![1.0];
            self.terminated = iteration;
            return ControlFlow::Break(iteration);
        }

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

        // Boosting as Frank-Wolfe, Algorithm 1, lines 11-13:
        // evaluate both candidates on the same atoms, including the new hypothesis.
        self.hypotheses.push(h);
        let (objval_lp, weight_lp) = self.update_lpboost();
        let (objval_fw, weight_fw) = self.update_frank_wolfe();

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
        let (_, weights) = soft_margin_optimization(self.nu, &self.sample, &self.hypotheses[..]);
        self.weights = weights;
        WeightedMajority::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}

impl<H> CurrentHypothesis for MlpBoost<'_, H>
where
    H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        WeightedMajority::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct Margins([f64; 3]);
    impl Classifier for Margins {
        fn confidence(&self, sample: &Sample, row: usize) -> f64 {
            self.0[row] * sample.target()[row]
        }
    }
    impl WeakLearner for Margins {
        type Hypothesis = Self;
        fn produce(&self, _: &Sample, _: &[f64]) -> Self {
            self.clone()
        }
    }

    #[test]
    #[should_panic(expected = "BlendedPairwise is not supported")]
    fn unsupported_rule_is_rejected_at_configuration() {
        let sample = Sample::dummy(3);
        let _ = MlpBoost::<Margins>::init(&sample).update_rule(FwUpdateRule::BlendedPairwise);
    }

    #[test]
    fn supported_rules_produce_finite_simplex_weights_over_multiple_rounds() {
        let sample = Sample::dummy(3);
        for rule in [
            FwUpdateRule::Classic,
            FwUpdateRule::ShortStep,
            FwUpdateRule::LineSearch,
        ] {
            let mut booster = MlpBoost::init(&sample).update_rule(rule);
            booster.preprocess();
            assert!(booster.boost(&Margins([1.0, 1.0, -1.0]), 1).is_continue());
            assert!(booster.boost(&Margins([-1.0, 1.0, 1.0]), 2).is_continue());
            assert_eq!(booster.hypotheses.len(), 2);
            assert_eq!(booster.weights.len(), 2);
            assert!(booster.weights.iter().all(|w| w.is_finite() && *w >= 0.0));
            assert!((booster.weights.iter().sum::<f64>() - 1.0).abs() < 1e-7);
        }
    }

    #[test]
    fn chosen_objective_is_the_better_full_candidate() {
        let sample = Sample::dummy(3);
        let first = Margins([1.0, 1.0, -1.0]);
        let second = Margins([-1.0, 1.0, 1.0]);
        let mut candidates = MlpBoost::init(&sample);
        candidates.preprocess();
        assert!(candidates.boost(&first, 1).is_continue());
        candidates.hypotheses.push(second.clone());
        let (lp_score, lp_weights) = candidates.update_lpboost();
        let (fw_score, fw_weights) = candidates.update_frank_wolfe();
        let explicit = |weights: &[f64]| {
            let margins: Vec<_> = (0..3)
                .map(|i| -(weights[0] * first.0[i] + weights[1] * second.0[i]))
                .collect();
            candidates.objective.objective_value(&margins)
        };
        assert!(lp_weights[1] > 0.1);
        assert!((lp_score - explicit(&lp_weights)).abs() < 1e-12);
        assert!((fw_score - explicit(&fw_weights)).abs() < 1e-12);
        let mut actual = MlpBoost::init(&sample);
        actual.preprocess();
        assert!(actual.boost(&first, 1).is_continue());
        assert!(actual.boost(&second, 2).is_continue());
        assert!((actual.objval(&actual.weights) - lp_score.max(fw_score)).abs() < 1e-10);
    }
    #[test]
    fn repeated_runs_match_a_fresh_booster() {
        struct Oracle;
        impl WeakLearner for Oracle {
            type Hypothesis = Margins;
            fn produce(&self, _: &Sample, dist: &[f64]) -> Margins {
                if dist[0] < dist[2] {
                    Margins([-1.0, 1.0, 1.0])
                } else {
                    Margins([1.0, 1.0, -1.0])
                }
            }
        }
        let sample = Sample::dummy(3);
        let mut reused = MlpBoost::init(&sample).tolerance(0.1);
        let first = reused.run(&Oracle);
        let second = reused.run(&Oracle);
        let fresh = MlpBoost::init(&sample).tolerance(0.1).run(&Oracle);
        for row in 0..3 {
            assert!((first.confidence(&sample, row) - fresh.confidence(&sample, row)).abs() < 1e-8);
            assert!(
                (second.confidence(&sample, row) - fresh.confidence(&sample, row)).abs() < 1e-8
            );
        }
    }

    #[test]
    fn full_capping_keeps_the_uniform_oracle_hypothesis() {
        let sample = Sample::dummy(3);
        let hypothesis = Margins([-1.0, 0.0, 1.0]);
        let mut booster = MlpBoost::init(&sample).nu(3.0);
        let output = booster.run(&hypothesis);
        assert_eq!(output.hypotheses.len(), 1);
        assert!((output.weights[0] - 1.0).abs() < 1e-7);
        let objective = booster.objective.objective_value(&[1.0, 0.0, -1.0]);
        assert!(objective.is_finite() && objective.abs() < 1e-12);
        assert!(booster.dist.iter().all(|d| (*d - 1.0 / 3.0).abs() < 1e-12));
        assert_eq!(booster.terminated, 1);
    }
}
