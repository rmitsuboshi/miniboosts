//! Implementation of [`CorrectiveErlpBoost`] (Shalev-Shwartz and Singer,
//! 2010) and its Frank-Wolfe objective [`CorrErlpFwObjective`].
//!
//! The name follows Warmuth, Glocer, and Vishwanathan (2008), who call this
//! algorithm the corrective version of ERLPBoost. See the
//! [`CorrectiveErlpBoost`] struct for the algorithm, guarantees, and an
//! example.
use crate::core::CurrentHypothesis;
use crate::core::{
    Booster, Classifier, Sample, WeakLearner,
    constants::{DEFAULT_CAPPING, DEFAULT_TOLERANCE},
    tools::checkers,
    tools::helpers,
};
use crate::hypotheses::WeightedMajority;
use crate::optimization::{
    ErlpSoftMarginObjective, FrankWolfe, FwUpdateRule, ObjectiveFunction, StepSize,
};

use std::ops::ControlFlow;

/// Corrective ERLPBoost: soft-margin boosting by Frank-Wolfe steps on the
/// entropy-regularized objective.
///
/// # Algorithm
///
/// Let `A = (y_i h_j(x_i))_{i,j}` be the margin matrix, `ε = tolerance`, and
/// `Δ_{m,ν} = { d : Σ_i d_i = 1, 0 ≤ d_i ≤ 1/ν }`. The booster maximizes the
/// entropy-regularized soft-margin objective
/// ```txt
/// -f̃*(-Aw) = min_{d ∈ Δ_{m,ν}} [ d·Aw + (1/η) Δ(d) ],   η = (2/ε) ln(m/ν),
/// ```
/// over nonnegative weights `w` with `Σ_j w_j ≤ 1`, where
/// `Δ(d) = Σ_i d_i ln(m d_i)` is the relative entropy to the uniform
/// distribution. Starting from `w_1 = 0` (so `d_1` is uniform), round `t`:
///
/// 1. obtains `h_t` from the weak learner on `d_t` and its margins
///    `a_i = y_i h_t(x_i)`, which must be finite and in `[-1, 1]`;
/// 2. stops if the Frank-Wolfe gap `d_t·a - d_t·Aw_t ≤ ε/2`; `h_t` is then
///    discarded;
/// 3. otherwise chooses a step `λ_t ∈ [0, 1]` and updates
///    ```txt
///    w_{t+1} = (1 - λ_t) w_t + λ_t e_{h_t},
///    ```
///    where the step rule is set by [`update_rule`](Self::update_rule):
///    - `ShortStep` (default, Figure 1 of the paper):
///      `λ_t = clip_[0,1]( d_t·(a - Aw_t) / (η ‖a - Aw_t‖_∞²) )`;
///    - `Classic`: `λ_t = 2/(k + 2)` after `k` previous steps (so the first
///      step is `1`);
///    - `LineSearch`: bisection for the maximizer of the objective on the
///      segment, to an interval width of `1e-10`;
/// 4. sets `d_{t+1}` to the minimizer above at `w_{t+1}`, i.e.
///    `d_{t+1,i} ∝ exp(-η (Aw_{t+1})_i)` capped at `1/ν`.
///
/// The output is `Σ_j w_j h_j` with these weights; they are not re-fitted or
/// normalized, so their sum can be below one and confidence values keep that
/// scale. A hypothesis returned in several rounds appears several times.
///
/// Implementation choices: the paper runs Figure 1 for a fixed number of
/// rounds; this implementation additionally stops on the Frank-Wolfe gap
/// above and caps the run at `ceil(32 ln(m/ν)/ε²)` rounds (see
/// [`max_loop`](Self::max_loop)). Training margins `Aw` are cached and
/// updated as `(1 - λ) Aw + λ a`, so earlier hypotheses are never evaluated
/// again. If `ν = m`, the only feasible distribution is uniform; the first
/// hypothesis is returned with weight `1` after one round, even if its edge is
/// nonpositive.
///
/// # Theoretical guarantee
///
/// Shalev-Shwartz and Singer show that, when the weak learner returns a
/// hypothesis of maximum edge on every distribution, the algorithm of their
/// Figure 1 reaches an `ε`-accurate solution of the relaxed-margin problem in
/// `O(ln(m)/ε²)` rounds; for the soft margin with capping `ν` the entropy
/// term is bounded by `ln(m/ν)`, giving the `O(ln(m/ν)/ε²)` bound stated in
/// Mitsuboshi, Hatano, and Takimoto, "Boosting as Frank-Wolfe", which also
/// covers the Classic step and notes that line search makes at least as much
/// progress per round as the short step. Under the same assumption, a
/// Frank-Wolfe gap of at most `ε/2` bounds the suboptimality of the
/// regularized objective, and the regularization costs at most another
/// `ε/2`.
///
/// # Limitations
///
/// - The maximum-edge (or weak-learning) assumption is not checked at
///   runtime; a decision tree does not by itself ensure it, and the gap
///   test is then only a heuristic stopping rule.
/// - `tolerance` is an optimization accuracy on the soft-margin objective,
///   not a classification-error threshold.
/// - `nu` caps every example weight at `1 / nu`; it does not guarantee a
///   particular number of margin errors.
/// - Labels must be `-1` or `+1` with both classes present; preprocessing
///   panics otherwise. Weak hypotheses whose margins are non-finite or
///   outside `[-1, 1]` cause a panic.
/// - Custom hypotheses must predict consistently on a fixed sample, because
///   training margins are cached.
/// - `FwUpdateRule::BlendedPairwise` is not supported.
/// - Many more rounds than LPBoost or ERLPBoost are typically needed, though
///   each round costs only `O(m ln m)` plus one weak-learner call.
///
/// # Parameters
///
/// | Builder method | Meaning | Default |
/// | --- | --- | --- |
/// | [`nu`](CorrectiveErlpBoost::nu) | capping parameter `ν ∈ [1, m]` | `1` ([`DEFAULT_CAPPING`]) |
/// | [`tolerance`](CorrectiveErlpBoost::tolerance) | accuracy `ε`; sets `η`, the stopping gap `ε/2`, and the budget | `1e-3` ([`DEFAULT_TOLERANCE`]) |
/// | [`update_rule`](CorrectiveErlpBoost::update_rule) | Frank-Wolfe step rule | [`FwUpdateRule::ShortStep`] |
///
/// # Reference
///
/// Shai Shalev-Shwartz and Yoram Singer.
/// [On the equivalence of weak learnability and linear separability: new relaxations and efficient boosting algorithms](https://link.springer.com/article/10.1007/s10994-010-5173-z).
/// Machine Learning 80, 2010.
///
/// # Examples
/// Train on a held-out split with decision stumps and evaluate on the test
/// part.
/// ```
/// use std::io::BufReader;
/// use miniboosts::{FwUpdateRule, Sample, prelude::*};
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
/// // regularization, the Frank-Wolfe gap threshold 0.05, and the budget.
/// let mut booster = CorrectiveErlpBoost::init(&train)
///     .nu(2.0)
///     .tolerance(0.1)
///     .update_rule(FwUpdateRule::ShortStep);
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
/// println!("CorrectiveErlpBoost test accuracy: {accuracy}");
///
/// // Budget ceil(32 ln(m/ν) / ε²) = ceil(32 ln 4 / 0.01) = 4437 rounds,
/// // available once training has set up the sample size.
/// assert_eq!(booster.max_loop(), 4437);
/// assert!(f.hypotheses.len() <= 4437);
/// // Frank-Wolfe weights are nonnegative; they are not renormalized, so
/// // their sum can be below one.
/// assert!(f.weights.iter().all(|&w| w >= 0.0));
/// assert!(f.weights.iter().sum::<f64>() <= 1.0 + 1e-9);
/// ```
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
    // Cached training margins Aw; no historical hypothesis inference per step.
    margins: Vec<f64>,
    hypotheses: Vec<H>,

    max_iter: usize,
    terminated: usize,
}

impl<'a, H> CorrectiveErlpBoost<'a, H> {
    /// Constructs a new instance of `CorrectiveErlpBoost` for `sample` with
    /// the default capping parameter `nu = 1`, `tolerance = 1e-3`, and the
    /// ShortStep update rule.
    ///
    /// The regularization parameter and iteration budget are computed when
    /// training starts.
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
    /// // A coarse tolerance keeps the budget small; the default `1e-3` allows
    /// // about 3.2 · 10⁷ · ln(m/ν) rounds.
    /// let mut booster = CorrectiveErlpBoost::init(&sample).tolerance(0.2);
    /// let f = booster.run(&learner);
    /// assert_eq!(f.predict_all(&sample).len(), 6);
    /// ```
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
            margins: Vec::new(),
            hypotheses: Vec::new(),
            max_iter: usize::MAX,
            terminated: usize::MAX,
        }
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
    /// let mut booster = CorrectiveErlpBoost::init(&sample).nu(3.0).tolerance(0.2);
    /// let f = booster.run(&learner);
    /// assert!(!f.hypotheses.is_empty());
    /// ```
    pub fn nu(mut self, nu: f64) -> Self {
        let n_examples = self.sample.shape().0;
        checkers::capping_parameter(nu, n_examples);
        self.nu = nu;
        self
    }

    /// Sets the soft-margin objective tolerance `ε`.
    ///
    /// Half of `ε` is spent on the entropy regularization
    /// (`η = (2/ε) ln(m/ν)`) and half on the Frank-Wolfe gap stopping test;
    /// `ε` also determines the iteration budget. It is not a
    /// classification-error threshold. Default: `1e-3`.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Panics
    /// Panics unless `tolerance` is finite and positive and `tolerance / 2`
    /// is nonzero. Training later panics if the resulting `η` is not finite
    /// and positive.
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
    /// let mut booster = CorrectiveErlpBoost::init(&sample).tolerance(0.2);
    /// let f = booster.run(&learner);
    /// // Budget ceil(32 ln(6) / 0.2²) = 1434 rounds.
    /// assert_eq!(booster.max_loop(), 1434);
    /// assert!(f.hypotheses.len() <= 1434);
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
        self.half_tolerance = tolerance / 2.0;
        self
    }

    /// Sets the Frank-Wolfe step-size rule.
    ///
    /// Supports [`FwUpdateRule::Classic`], [`FwUpdateRule::ShortStep`]
    /// (default; the step of Shalev-Shwartz and Singer), and
    /// [`FwUpdateRule::LineSearch`].
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Panics
    /// Panics on [`FwUpdateRule::BlendedPairwise`]: blended-pairwise updates
    /// require coefficient-space atoms and are not supported by this
    /// booster's margin-space implementation.
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{Sample, prelude::*};
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// use miniboosts::FwUpdateRule;
    ///
    /// let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    /// // Exact line search instead of the default short step.
    /// let mut booster = CorrectiveErlpBoost::init(&sample)
    ///     .tolerance(0.2)
    ///     .update_rule(FwUpdateRule::LineSearch);
    /// let f = booster.run(&learner);
    /// assert!(!f.hypotheses.is_empty());
    /// ```
    #[inline(always)]
    pub fn update_rule(mut self, update_rule: FwUpdateRule) -> Self {
        assert!(
            !matches!(update_rule, FwUpdateRule::BlendedPairwise),
            "BlendedPairwise is not supported by this booster",
        );
        self.update_rule = update_rule;
        self
    }

    /// Sets `self.eta = ln(m/ν) / (ε/2)`, or `1` when `ν = m`.
    ///
    /// Time complexity: `O(1)`.
    #[inline(always)]
    fn regularization_param(&mut self) {
        let m = self.dist.len() as f64;
        let ln_part = (m / self.nu).ln();
        // At nu=m the feasible distribution is uniform and its relative
        // entropy is zero. Any positive eta represents the same objective.
        self.eta = if ln_part == 0.0 {
            1.0
        } else {
            ln_part / self.half_tolerance
        };
        assert!(
            self.eta.is_finite() && self.eta > 0.0,
            "regularization parameter must be finite and positive"
        );
    }

    /// Returns the iteration budget `max(1, ceil(32 ln(m/ν) / ε²))` for the
    /// configured tolerance `ε`.
    ///
    /// The budget corresponds to the `O(ln(m/ν)/ε²)` bound under the
    /// weak-learning assumption; it is not a training-error guarantee.
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
    /// let mut booster = CorrectiveErlpBoost::init(&sample).nu(2.0).tolerance(0.2);
    /// // m = 6: ceil(32 ln(6/2) / 0.2²) = ceil(878.9) = 879, before and
    /// // after training.
    /// assert_eq!(booster.max_loop(), 879);
    /// let _f = booster.run(&learner);
    /// assert_eq!(booster.max_loop(), 879);
    /// ```
    pub fn max_loop(&self) -> usize {
        let m = self.sample.shape().0 as f64;

        let ln_m = (m / self.nu).ln();
        let max_iter = 8.0 * ln_m / self.half_tolerance.powi(2);

        (max_iter.ceil() as usize).max(1)
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
where
    H: Classifier,
{
    /// Sets `self.dist` to the regularized minimizer at the cached margins.
    fn update_distribution_mut(&mut self) {
        let neg_margins: Vec<_> = self.margins.iter().map(|m| -m).collect();
        self.dist = self.objective.gradient(&neg_margins[..]);
    }

    fn update_params_mut(&mut self, h: H, new_margins: Vec<f64>) {
        // Shalev-Shwartz and Singer (2008), Fig. 1, starts at w=0 and
        // applies the same update on every round. Forcing the first weight
        // to one skips the short step and restarts Classic's schedule.
        let negative_current: Vec<_> = self.margins.iter().map(|m| -m).collect();
        let negative_new: Vec<_> = new_margins.iter().map(|m| -m).collect();
        let stepsize = self
            .frank_wolfe
            .get_stepsize_mut(&negative_current, &negative_new);
        match stepsize {
            StepSize::Normal(stepsize) => {
                // Fig. 1: A((1-a)w + a e_h) = (1-a)Aw + a Ah.
                // This uses O(m) state instead of re-evaluating all past trees.
                for (margin, new) in self.margins.iter_mut().zip(new_margins) {
                    *margin = (1.0 - stepsize) * *margin + stepsize * new;
                }
                self.weights.iter_mut().for_each(|w| {
                    *w *= 1f64 - stepsize;
                });
                self.weights.push(stepsize);
                self.hypotheses.push(h);
            }
            StepSize::BpfwMoveWeights { .. } => {
                unreachable!("BlendedPairwise is rejected by update_rule")
            }
        }
        // Fig. 1 optimizes over the l1 ball, not the simplex: a short step
        // from zero can have total mass below one.
        assert!(self.weights.iter().all(|w| w.is_finite() && *w >= 0.0));
        assert!(self.weights.iter().sum::<f64>() <= 1.0 + 1e-10);

        self.update_distribution_mut();
    }
}

impl<H> Booster<H> for CorrectiveErlpBoost<'_, H>
where
    H: Classifier + Clone,
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
        self.margins = vec![0.0; n_examples];

        self.regularization_param();
        self.max_iter = self.max_loop();
        self.terminated = self.max_iter;

        // self.classifiers = Vec::new();
        self.weights = Vec::new();
        self.hypotheses = Vec::new();

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

        let new_margins: Vec<_> = helpers::margins(self.sample, &h).collect();
        assert!(
            new_margins
                .iter()
                .all(|m| m.is_finite() && (-1.0..=1.0).contains(m)),
            "weak hypothesis margins must be finite and lie in [-1, 1]"
        );

        // With nu=m the dual feasible set is the singleton uniform distribution.
        // One oracle call suffices; retain it even when its edge is nonpositive.
        if self.nu == self.sample.shape().0 as f64 {
            self.margins = new_margins;
            self.hypotheses.push(h);
            self.weights = vec![1.0];
            self.terminated = iteration;
            return ControlFlow::Break(iteration);
        }

        let duality_gap = {
            let e1 = helpers::inner_product(&new_margins[..], &self.dist[..]);
            let e2 = helpers::inner_product(&self.margins, &self.dist[..]);
            e1 - e2
        };

        // Update the parameters
        if duality_gap <= self.half_tolerance {
            self.terminated = iteration;
            return ControlFlow::Break(iteration);
        }

        self.update_params_mut(h, new_margins);

        ControlFlow::Continue(())
    }

    fn postprocess(&mut self) -> Self::Output {
        WeightedMajority {
            weights: self.weights.clone(),
            hypotheses: self.hypotheses.clone(),
        }
    }
}

impl<H> CurrentHypothesis for CorrectiveErlpBoost<'_, H>
where
    H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        WeightedMajority {
            weights: self.weights.clone(),
            hypotheses: self.hypotheses.clone(),
        }
    }
}

/// The entropy-regularized soft-margin objective in the minimization form
/// expected by [`FrankWolfe`].
///
/// Wraps an [`ErlpSoftMarginObjective`] with capping `ν` and regularization
/// `η`. At a point `θ = -Aw` (negated training margins) it evaluates the
/// Fenchel conjugate
/// ```txt
/// f̃*(θ) = max_{d ∈ Δ_{m,ν}} [ d·θ - (1/η) Δ(d) ],
/// ```
/// where `Δ(d)` is the relative entropy to the uniform distribution. Its
/// gradient is the maximizing distribution `d`, which is the booster's
/// distribution over examples. `f̃*` is `η`-smooth with respect to the
/// `ℓ∞` norm, so [`ObjectiveFunction::smooth`] returns `(η, f64::MAX)`;
/// the short-step rule uses this constant.
///
/// Minimizing `f̃*(-Aw)` is equivalent to maximizing the regularized soft
/// margin `-f̃*(-Aw)`. Used by [`CorrectiveErlpBoost`] and
/// [`MlpBoost`](crate::MlpBoost).
pub struct CorrErlpFwObjective(ErlpSoftMarginObjective);

impl CorrErlpFwObjective {
    /// Wraps `objective` (which fixes `ν` and `η`) for use with
    /// [`FrankWolfe`].
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::boosters::corrective_erlpboost::CorrErlpFwObjective;
    /// use miniboosts::{ErlpSoftMarginObjective, ObjectiveFunction};
    ///
    /// // m = 4 examples, capping ν = 2 (weights ≤ 1/2), regularization η = 10.
    /// let (nu, eta) = (2.0, 10.0);
    /// let objective = CorrErlpFwObjective::new(ErlpSoftMarginObjective::new(nu, eta));
    ///
    /// // `η`-smooth with respect to the ℓ∞ norm.
    /// assert_eq!(objective.smooth(), (eta, f64::MAX));
    ///
    /// // At θ = -Aw = 0 (all margins zero) the maximizer is uniform and
    /// // f̃*(0) = 0, since the relative entropy of the uniform distribution is 0.
    /// let d = objective.gradient(&[0.0; 4]);
    /// assert!(d.iter().all(|&di| (di - 0.25).abs() < 1e-9));
    /// assert!(objective.objective_value(&[0.0; 4]).abs() < 1e-9);
    ///
    /// // Margins y_i f(x_i) = [1, 0.5, -0.5, -1], i.e. θ = [-1, -0.5, 0.5, 1]:
    /// // the gradient is a distribution in the capped simplex that puts more
    /// // weight on examples with smaller margins.
    /// let d = objective.gradient(&[-1.0, -0.5, 0.5, 1.0]);
    /// assert!((d.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    /// assert!(d.iter().all(|&di| (0.0..=0.5 + 1e-9).contains(&di)));
    /// assert!(d.windows(2).all(|w| w[0] <= w[1] + 1e-12));
    /// ```
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

    /// Returns the objective value `f̃*(θ)` at the point `θ = -Aw`.
    fn objective_value(&self, point: &[f64]) -> f64 {
        -self.0.objective_value(point)
    }

    /// Returns the gradient `∇f̃*(θ)`, the maximizing distribution over
    /// examples, at the point `θ = -Aw`.
    fn gradient(&self, point: &[f64]) -> Vec<f64> {
        self.0.gradient(point).into_iter().collect()
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
    fn rejects_nonfinite_oracle_and_unrepresentable_eta() {
        let sample = Sample::dummy(3);
        let mut tiny = CorrectiveErlpBoost::<Margins>::init(&sample).tolerance(f64::from_bits(2));
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tiny.preprocess())).is_err()
        );
        for nu in [1.0, 3.0] {
            let mut booster = CorrectiveErlpBoost::init(&sample).nu(nu);
            booster.preprocess();
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || booster.boost(&Margins([f64::NAN, 0.0, 0.0]), 1)
                ))
                .is_err()
            );
        }
    }

    #[test]
    fn rejects_invalid_tolerance() {
        let sample = Sample::dummy(3);
        for tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::from_bits(1)] {
            assert!(
                std::panic::catch_unwind(
                    || CorrectiveErlpBoost::<Margins>::init(&sample).tolerance(tolerance)
                )
                .is_err()
            );
        }
    }

    #[test]
    fn cached_margins_match_model_and_do_not_reinfer_old_hypotheses() {
        use std::cell::Cell;
        use std::rc::Rc;
        #[derive(Clone, Debug, PartialEq)]
        struct Counted {
            values: [f64; 3],
            calls: Rc<Cell<usize>>,
        }
        impl Classifier for Counted {
            fn confidence(&self, sample: &Sample, row: usize) -> f64 {
                self.calls.set(self.calls.get() + 1);
                self.values[row] * sample.target()[row]
            }
        }
        impl WeakLearner for Counted {
            type Hypothesis = Self;
            fn produce(&self, _: &Sample, _: &[f64]) -> Self {
                self.clone()
            }
        }
        let sample = Sample::dummy(3);
        let calls = Rc::new(Cell::new(0));
        let atoms = [[1.0, 1.0, -1.0], [-1.0, 1.0, 1.0], [1.0, -1.0, 1.0]];
        for rule in [
            FwUpdateRule::Classic,
            FwUpdateRule::ShortStep,
            FwUpdateRule::LineSearch,
        ] {
            let mut booster = CorrectiveErlpBoost::init(&sample)
                .update_rule(rule)
                .tolerance(0.01);
            // Reusing the booster must reset all cached margins and schedules.
            for _ in 0..2 {
                booster.preprocess();
                for round in 1..=20 {
                    let values = *atoms
                        .iter()
                        .max_by(|a, b| {
                            helpers::edge_from_margins(*a, &booster.dist)
                                .total_cmp(&helpers::edge_from_margins(*b, &booster.dist))
                        })
                        .unwrap();
                    calls.set(0);
                    let flow = booster.boost(
                        &Counted {
                            values,
                            calls: calls.clone(),
                        },
                        round,
                    );
                    assert!(calls.get() <= 3, "old hypotheses were evaluated again");
                    let model = booster.current_hypothesis();
                    let direct: Vec<_> = helpers::margins(&sample, &model).collect();
                    let cached: Vec<f64> = booster.margins.clone();
                    for (a, b) in direct.iter().zip(&cached) {
                        // At most 20 convex updates of bounded margins; covers
                        // accumulated rounding, far below the stopping gap.
                        assert!((a - b).abs() < 1e-12, "{a} != {b}");
                    }
                    let expected = booster
                        .objective
                        .gradient(&direct.iter().map(|m| -m).collect::<Vec<_>>());
                    for (a, b) in expected.iter().zip(&booster.dist) {
                        assert!((a - b).abs() < 1e-10);
                    }
                    if flow.is_break() {
                        break;
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "BlendedPairwise is not supported")]
    fn unsupported_rule_is_rejected_at_configuration() {
        let sample = Sample::dummy(3);
        let _ = CorrectiveErlpBoost::<Margins>::init(&sample)
            .update_rule(FwUpdateRule::BlendedPairwise);
    }

    #[test]
    fn supported_rules_produce_finite_l1_ball_weights_over_multiple_rounds() {
        let sample = Sample::dummy(3);
        for rule in [
            FwUpdateRule::Classic,
            FwUpdateRule::ShortStep,
            FwUpdateRule::LineSearch,
        ] {
            let mut booster = CorrectiveErlpBoost::init(&sample).update_rule(rule);
            booster.preprocess();
            assert!(booster.boost(&Margins([1.0, 1.0, -1.0]), 1).is_continue());
            assert!(booster.boost(&Margins([-1.0, 1.0, 1.0]), 2).is_continue());
            assert_eq!(booster.hypotheses.len(), 2);
            assert_eq!(booster.weights.len(), 2);
            assert!(booster.weights.iter().all(|w| w.is_finite() && *w >= 0.0));
            assert!(booster.weights.iter().sum::<f64>() <= 1.0 + 1e-10);
        }
    }
    #[test]
    fn first_short_step_follows_the_bound_from_zero() {
        let sample = Sample::dummy(3);
        let mut booster = CorrectiveErlpBoost::init(&sample);
        booster.preprocess();
        let expected = (1.0 / 3.0) / booster.eta;
        assert!(booster.boost(&Margins([1.0, 1.0, -1.0]), 1).is_continue());
        assert!((booster.weights[0] - expected).abs() < 1e-15);
        let output = booster.current_hypothesis();
        assert_eq!(output.weights, booster.weights);
        assert_eq!(booster.postprocess().weights, booster.weights);
        assert!((output.confidence(&sample, 0).abs() - expected).abs() < 1e-15);
    }

    #[test]
    fn classic_counts_the_first_update() {
        let sample = Sample::dummy(3);
        let mut booster = CorrectiveErlpBoost::init(&sample).update_rule(FwUpdateRule::Classic);
        booster.preprocess();
        assert!(booster.boost(&Margins([1.0, 1.0, -1.0]), 1).is_continue());
        assert!(booster.boost(&Margins([-1.0, 1.0, 1.0]), 2).is_continue());
        assert!((booster.weights[0] - 1.0 / 3.0).abs() < 1e-15);
        assert!((booster.weights[1] - 2.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn zero_edge_returns_the_zero_model() {
        let sample = Sample::dummy(3);
        let mut booster = CorrectiveErlpBoost::init(&sample);
        let output = booster.run(&Margins([0.0, 0.0, 0.0]));
        assert!(output.weights.is_empty());
        assert_eq!(output.confidence(&sample, 0), 0.0);
    }

    #[test]
    fn minimization_gradient_matches_finite_differences() {
        for nu in [1.0, 2.0] {
            let objective = CorrErlpFwObjective::new(ErlpSoftMarginObjective::new(nu, 2.0));
            let point = [-0.7, 0.2, 0.9];
            let gradient = objective.gradient(&point);
            for i in 0..point.len() {
                let mut lower = point;
                let mut upper = point;
                lower[i] -= 1e-6;
                upper[i] += 1e-6;
                let derivative =
                    (objective.objective_value(&upper) - objective.objective_value(&lower)) / 2e-6;
                assert!((derivative - gradient[i]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn short_steps_decrease_the_regularized_training_objective() {
        let sample = Sample::dummy(3);
        let mut booster = CorrectiveErlpBoost::init(&sample);
        booster.preprocess();
        let atoms = [
            Margins([1.0, 1.0, -1.0]),
            Margins([-1.0, 1.0, 1.0]),
            Margins([1.0, -1.0, 1.0]),
        ];
        let mut previous = 0.0;
        for round in 1..=100 {
            let best = atoms
                .iter()
                .max_by(|a, b| {
                    helpers::inner_product(&a.0, &booster.dist)
                        .total_cmp(&helpers::inner_product(&b.0, &booster.dist))
                })
                .unwrap();
            let flow = booster.boost(best, round);
            let model = booster.current_hypothesis();
            let negative_margins: Vec<_> = helpers::margins(&sample, &model).map(|m| -m).collect();
            let value = -booster.objective.objective_value(&negative_margins);
            assert!(value <= previous + 1e-12);
            previous = value;
            if flow.is_break() {
                break;
            }
        }
        assert!(previous < -1e-4);
    }

    #[test]
    fn full_capping_keeps_the_uniform_oracle_hypothesis() {
        let sample = Sample::dummy(3);
        let hypothesis = Margins([-1.0, 0.0, 1.0]);
        let mut booster = CorrectiveErlpBoost::init(&sample).nu(3.0);
        let output = booster.run(&hypothesis);
        assert_eq!(output.hypotheses.len(), 1);
        assert!((output.weights[0] - 1.0).abs() < 1e-7);
        let objective = booster.objective.objective_value(&[1.0, 0.0, -1.0]);
        assert!(objective.is_finite() && objective.abs() < 1e-12);
        assert!(booster.dist.iter().all(|d| (*d - 1.0 / 3.0).abs() < 1e-12));
        assert_eq!(booster.terminated, 1);
    }
}
