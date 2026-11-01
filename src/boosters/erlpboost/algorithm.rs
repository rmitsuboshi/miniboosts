//! Implementation of [`ErlpBoost`] (Warmuth, Glocer, and Vishwanathan, 2008).
//!
//! See the [`ErlpBoost`] struct for the algorithm, guarantees, and an example.
use crate::core::CurrentHypothesis;
use crate::core::{
    Booster, Classifier, Sample, WeakLearner,
    constants::{DEFAULT_CAPPING, DEFAULT_TOLERANCE},
    tools::checkers,
    tools::helpers,
};
use crate::hypotheses::WeightedMajority;
use crate::optimization::{EntropyRegularizedMaxEdge, RowGeneration, soft_margin_optimization};

use std::ops::ControlFlow;

/// ERLPBoost: totally corrective soft-margin boosting with entropy
/// regularization.
///
/// # Algorithm
///
/// ERLPBoost replaces LPBoost's edge minimization by an entropy-regularized
/// problem over the capped simplex
/// `Δ_{m,ν} = { d : Σ_i d_i = 1, 0 ≤ d_i ≤ 1/ν }`:
/// ```txt
/// P^t(d) = max_{q ≤ t} Σ_i d_i y_i h_q(x_i) + (1/η) Δ(d),
/// d_t    = argmin_{d ∈ Δ_{m,ν}} P^t(d),
/// ```
/// where `Δ(d) = Σ_i d_i ln(m d_i)` is the relative entropy to the uniform
/// distribution and, with `ε = tolerance`,
/// ```txt
/// η = max(1/2, (2/ε) ln(m/ν)).
/// ```
/// Starting from the uniform `d_0`, `γ̂ = 1`, and `γ* = -1`, round `t`:
///
/// 1. obtains `h_t` from the weak learner on `d_{t-1}`;
/// 2. updates `γ̂ ← min(γ̂, edge_{d_{t-1}}(h_t) + Δ(d_{t-1})/η)`, where
///    `edge_d(h) = Σ_i d_i y_i h(x_i)`;
/// 3. stops if `γ̂ - γ* ≤ ε/2`; the hypothesis `h_t` is then discarded;
/// 4. otherwise adds `h_t`, solves for `d_t`, and sets `γ*` to the solver's
///    entropy-dual lower bound on `min_d P^t(d)`.
///
/// After stopping, [`Booster::postprocess`] solves the unregularized
/// soft-margin LP over all collected hypotheses (as in LPBoost) and returns
/// the resulting [`WeightedMajority`].
///
/// Implementation choices: the regularized subproblem is solved with
/// exponential cones by the Clarabel interior-point solver
/// ([`RowGeneration`] with [`EntropyRegularizedMaxEdge`]). The algorithm
/// requires the exact restricted optimum for `γ*`; a numerically feasible
/// primal value would overestimate it, so the dual lower bound is used to
/// avoid stopping early on an inaccurate inner solution. The run is also
/// capped at an iteration budget
/// ```txt
/// T_max = ceil(max(8/ε, 32 ln(m/ν)/ε²)).
/// ```
///
/// # Theoretical guarantee
///
/// Warmuth, Glocer, and Vishwanathan (2008) show that, if the weak learner
/// returns a hypothesis with edge at least `g` on every distribution it
/// receives and the subproblems are solved exactly, ERLPBoost stops within
/// `O(ln(m/ν)/ε²)` rounds with a combined hypothesis whose soft-margin
/// objective is at least `g - ε`. The regularization accounts for half of
/// `ε` and the stopping gap for the other half. Re-solving the LP in
/// postprocessing cannot decrease the soft-margin objective over the same
/// hypotheses (up to solver accuracy). In contrast, LPBoost can need `Ω(m)`
/// rounds (see [`LpBoost`](crate::LpBoost)).
///
/// # Limitations
///
/// - The weak-learning assumption is not checked at runtime; a decision tree
///   does not by itself ensure it.
/// - `tolerance` is an optimization accuracy on the soft-margin objective,
///   not a classification-error threshold. It must be smaller than `2`.
/// - `nu` caps every example weight at `1 / nu`; it does not guarantee a
///   particular number of margin errors.
/// - Labels must be `-1` or `+1` with both classes present; preprocessing
///   panics otherwise.
/// - Inner solver failures **panic** rather than count as convergence; this
///   includes a missing entropy-dual lower bound and a failed final LP.
///   Extremely small tolerances can exceed the inner solver's accuracy.
/// - When the iteration budget is reached, training stops, prints
///   `reached to the max iteration` to standard output, and returns the LP
///   fit over the hypotheses collected so far.
/// - Each round solves a conic program whose size grows with the number of
///   hypotheses, so rounds are much more expensive than LPBoost's.
/// - [`CurrentHypothesis::current_hypothesis`] (used by loggers) uses the
///   weights of the regularized subproblem, which can differ from the
///   final LP weights returned by [`Booster::run`].
///
/// # Parameters
///
/// | Builder method | Meaning | Default |
/// | --- | --- | --- |
/// | [`nu`](ErlpBoost::nu) | capping parameter `ν ∈ [1, m]` | `1` ([`DEFAULT_CAPPING`]) |
/// | [`tolerance`](ErlpBoost::tolerance) | accuracy `ε ∈ (0, 2)`; sets `η`, the stopping gap `ε/2`, and the budget | `1e-3` ([`DEFAULT_TOLERANCE`]) |
///
/// # Reference
///
/// Manfred K. Warmuth, Karen A. Glocer, and S. V. N. Vishwanathan.
/// [Entropy Regularized LPBoost](https://www.stat.purdue.edu/~vishy/papers/WarGloVis08.pdf).
/// Algorithmic Learning Theory (ALT), 2008.
///
/// # Examples
/// Train on a held-out split with decision stumps and evaluate on the test
/// part.
/// ```
/// use std::io::BufReader;
/// use miniboosts::{Sample, prelude::*};
///
/// // Labels follow the sign of `x + y`, so no single stump separates them.
/// let csv = b"x,y,class\n\
///     -2,1,-1\n-1,-2,-1\n1,-3,-1\n-3,2,-1\n0,-1,-1\n-2,-1,-1\n\
///     2,1,1\n1,2,1\n-1,3,1\n3,-1,1\n0,2,1\n2,2,1\n";
/// let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
///     .unwrap()
///     .set_target("class");
///
/// // A fixed permutation keeps the split reproducible:
/// // rows ix[0..4] form the test set, the other 8 rows the training set.
/// let ix = [0, 6, 4, 10, 1, 2, 3, 5, 7, 8, 9, 11];
/// let (train, test) = sample.split(&ix, 0, 4);
///
/// // The weak learner sees only the training split.
/// let learner = DecisionTreeBuilder::new(&train).max_depth(1).build();
///
/// // `nu = 2` caps every example weight at 1/2; `tolerance = 0.1` sets the
/// // regularization, the stopping gap 0.05, and the iteration budget.
/// let (nu, eps) = (2.0, 0.1);
/// let mut booster = ErlpBoost::init(&train).nu(nu).tolerance(eps);
/// let f = booster.run(&learner);
///
/// let predictions = f.predict_all(&test);
/// assert!(predictions.iter().all(|&p| p == -1 || p == 1));
/// let correct = predictions
///     .iter()
///     .zip(test.target())
///     .filter(|&(&p, &y)| p as f64 == y)
///     .count();
/// let accuracy = correct as f64 / test.shape().0 as f64;
/// println!("ErlpBoost test accuracy: {accuracy}");
///
/// // Budget ceil(max(8/ε, 32 ln(m/ν)/ε²)) for m = 8 training examples.
/// let budget = (8.0 / eps).max(32.0 * (8.0 / nu).ln() / (eps * eps)).ceil() as usize;
/// // `terminated` is below the budget when the gap criterion stopped training.
/// let rounds = booster.terminated();
/// assert!(rounds >= 1 && rounds <= budget);
/// println!("stopped at round {rounds} (budget {budget})");
/// // The final LP weights are nonnegative and normalized.
/// assert!(f.weights.iter().all(|&w| w > 0.0));
/// assert!((f.weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
/// ```
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
    /// Constructs a new instance of `ErlpBoost` for `sample` with the default
    /// capping parameter `nu = 1` and `tolerance = 1e-3`.
    ///
    /// The regularization parameter and iteration budget are computed when
    /// training starts.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Panics
    /// Panics if `sample` has no examples.
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{Sample, prelude::*};
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    /// // A coarser tolerance than the default keeps the subproblems cheap.
    /// let mut booster = ErlpBoost::init(&sample).tolerance(0.1);
    /// let f = booster.run(&learner);
    /// assert_eq!(f.predict_all(&sample).len(), 6);
    /// ```
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
        let gamma_hat = 1f64;
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

    /// Initialize the entropy-cone solver.
    fn init_solver(&mut self) {
        let objective = EntropyRegularizedMaxEdge::new(self.eta);
        self.solver.initialize(self.nu, objective);
    }

    /// Sets the capping parameter `nu`.
    ///
    /// Every example weight is capped at `1 / nu`; `nu` must lie in `[1, m]`,
    /// where `m` is the number of training examples. Default: `1`.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Panics
    /// Panics if `nu` is not in `[1, m]`.
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{Sample, prelude::*};
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    /// // m = 6: every example weight is at most 1/3.
    /// let mut booster = ErlpBoost::init(&sample).nu(3.0).tolerance(0.1);
    /// let f = booster.run(&learner);
    /// assert!(!f.hypotheses.is_empty());
    /// ```
    pub fn nu(mut self, nu: f64) -> Self {
        checkers::capping_parameter(nu, self.n_sample);
        self.nu = nu;

        self
    }

    /// Returns the round at which training stopped.
    ///
    /// Returns `usize::MAX` before training. During training it holds the
    /// iteration budget until the stopping rule fires; if the budget is
    /// reached, it remains equal to the budget.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{Sample, prelude::*};
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    /// let mut booster = ErlpBoost::init(&sample).tolerance(0.1);
    /// assert_eq!(booster.terminated(), usize::MAX); // not trained yet
    /// let _f = booster.run(&learner);
    /// assert!(booster.terminated() >= 1 && booster.terminated() != usize::MAX);
    /// ```
    #[inline(always)]
    pub fn terminated(&self) -> usize {
        self.terminated
    }

    /// Sets the optimization tolerance `ε`.
    ///
    /// Half of `ε` is spent on the entropy regularization
    /// (`η = max(1/2, (2/ε) ln(m/ν))`) and half on the stopping gap
    /// `γ̂ - γ* ≤ ε/2`; `ε` also determines the iteration budget. It is not a
    /// classification-error threshold. Default: `1e-3`.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Panics
    /// Panics if `tolerance` is not finite and positive or if `tolerance / 2`
    /// underflows to zero. Training later panics if `tolerance ≥ 2` or if the
    /// resulting `η` is not finite.
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{Sample, prelude::*};
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    /// // ε = 0.2: stopping gap 0.1, η = max(1/2, ln(6) / 0.1).
    /// let mut booster = ErlpBoost::init(&sample).tolerance(0.2);
    /// let f = booster.run(&learner);
    /// assert!(!f.hypotheses.is_empty());
    /// ```
    #[inline(always)]
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        assert!(
            tolerance.is_finite() && tolerance > 0.0,
            "tolerance must be finite and positive"
        );
        assert!(
            tolerance / 2.0 > 0.0,
            "half tolerance must be representable"
        );
        self.half_tolerance = tolerance / 2f64;
        self
    }

    /// Sets `self.eta = max(1/2, ln(m/ν) / (ε/2))`.
    ///
    /// Time complexity: `O(1)`.
    #[inline(always)]
    fn regularization_param(&mut self) {
        let ln_m = (self.n_sample as f64 / self.nu).ln();
        self.eta = 0.5f64.max(ln_m / self.half_tolerance);
        assert!(
            self.eta.is_finite(),
            "regularization parameter must be finite"
        );
    }

    /// Returns the iteration budget
    /// `ceil(max(8/ε, 32 ln(m/ν) / ε²))` for the configured tolerance `ε`.
    ///
    /// The budget corresponds to the `O(ln(m/ν)/ε²)` bound under the
    /// weak-learning assumption; the tolerance concerns optimization
    /// accuracy, not classification error.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{Sample, prelude::*};
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// use miniboosts::DecisionTreeClassifier;
    ///
    /// let booster = ErlpBoost::<DecisionTreeClassifier>::init(&sample)
    ///     .nu(2.0)
    ///     .tolerance(0.2);
    /// // m = 6: max(8/0.2, 32 ln(6/2) / 0.2²) = max(40, 878.9) -> 879.
    /// assert_eq!(booster.max_loop(), 879);
    /// ```
    pub fn max_loop(&self) -> usize {
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
where
    H: Classifier,
{
    /// Updates `self.gamma_hat` with the edge of `h` on the current
    /// distribution plus its relative entropy divided by `eta`.
    ///
    /// Time complexity: `O(m)`, where `m` is the number of training examples.
    #[inline]
    fn update_gamma_hat_mut(&mut self, h: &H) {
        let edge = helpers::edge(self.sample, &self.dist[..], h);
        let entropy = helpers::entropy_from_uni_distribution(&self.dist[..]);

        let obj_val = edge + (entropy / self.eta);

        self.gamma_hat = self.gamma_hat.min(obj_val);
    }

    /// Solves the entropy-regularized edge minimization problem
    /// to update `self.dist`.
    fn update_distribution_mut(&mut self) {
        self.solver.solve(&self.hypotheses[..]);
        self.dist = self.solver.distribution_on_examples();
    }
}

impl<H> Booster<H> for ErlpBoost<'_, H>
where
    H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;

    fn name(&self) -> &str {
        "ErlpBoost"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_sample, n_feature) = self.sample.shape();
        let ratio = self.nu * 100f64 / n_sample as f64;
        let nu = helpers::format_unit(self.nu);
        let info = Vec::from([
            ("# of examples", format!("{n_sample}")),
            ("# of features", format!("{n_feature}")),
            ("Tolerance", format!("{}", 2f64 * self.half_tolerance)),
            ("Max iteration", format!("{}", self.max_iter)),
            ("Capping (outliers)", format!("{nu} ({ratio: >7.3} %)")),
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

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize) -> ControlFlow<usize>
    where
        W: WeakLearner<Hypothesis = H>,
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
        // Algorithm 1 requires the restricted optimum. A numerically feasible
        // primal value is an upper bound, so use the entropy dual lower bound
        // to avoid stopping early on an inaccurate inner solution.
        self.gamma_star = self
            .solver
            .objective_lower_bound()
            .expect("ERLPBoost requires an entropy dual lower bound");
        ControlFlow::Continue(())
    }

    fn postprocess(&mut self) -> Self::Output {
        let (_, weights) = soft_margin_optimization(self.nu, &self.sample, &self.hypotheses[..]);
        self.weights = weights;
        WeightedMajority::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}

impl<H> CurrentHypothesis for ErlpBoost<'_, H>
where
    H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        let weights = self.solver.weights_on_hypotheses();

        WeightedMajority::from_slices(&weights[..], &self.hypotheses[..])
    }
}

#[cfg(test)]
mod parameter_tests {
    use super::*;
    #[test]
    fn rejects_invalid_tolerance() {
        let sample = Sample::dummy(3);
        for tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::from_bits(1)] {
            assert!(
                std::panic::catch_unwind(|| ErlpBoost::<()>::init(&sample).tolerance(tolerance))
                    .is_err()
            );
        }
        let mut booster = ErlpBoost::<()>::init(&sample).tolerance(f64::from_bits(2));
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || booster.regularization_param()
            ))
            .is_err()
        );
    }
}
