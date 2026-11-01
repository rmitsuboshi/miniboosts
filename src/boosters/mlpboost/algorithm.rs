//! Implementation of [`MlpBoost`] from
//! [Boosting as Frank-Wolfe](https://arxiv.org/abs/2209.10831)
//! by Mitsuboshi, Hatano, and Takimoto.
//!
//! See the [`MlpBoost`] struct for the algorithm, guarantees, and an example.
use crate::boosters::corrective_erlpboost::CorrErlpFwObjective;
use crate::core::CurrentHypothesis;
use crate::core::{
    Booster, Classifier, Sample, WeakLearner,
    constants::{DEFAULT_CAPPING, DEFAULT_TOLERANCE},
    tools::checkers,
    tools::helpers,
};
use crate::hypotheses::WeightedMajority;
use crate::optimization::{
    ColumnGeneration, ErlpSoftMarginObjective, FrankWolfe, FwUpdateRule, ObjectiveFunction,
    StepSize, soft_margin_optimization,
};

use std::ops::ControlFlow;

/// MLPBoost: Frank-Wolfe soft-margin boosting with an LPBoost secondary
/// update.
///
/// # Algorithm
///
/// This is Algorithm 1 of "Boosting as Frank-Wolfe" with the LPBoost rule
/// (Algorithm 2) as the secondary update. Let `A = (y_i h_j(x_i))_{i,j}`,
/// `ε = tolerance`, and `Δ_{m,ν} = { d : Σ_i d_i = 1, 0 ≤ d_i ≤ 1/ν }`. The
/// booster maximizes the entropy-regularized soft-margin objective
/// ```txt
/// -f̃*(-Aw) = min_{d ∈ Δ_{m,ν}} [ d·Aw + (1/η) Δ(d) ],   η = (2/ε) ln(m/ν),
/// ```
/// over weights `w` in the simplex, where `Δ(d) = Σ_i d_i ln(m d_i)` is the
/// relative entropy to the uniform distribution. Starting from the uniform
/// `d_1` and `γ̂ = 1`, round `t`:
///
/// 1. obtains `h_t` from the weak learner on `d_t` and its margins
///    `a_i = y_i h_t(x_i)`, which must be finite and in `[-1, 1]`;
/// 2. updates `γ̂ ← min(γ̂, d_t·a)`, the smallest edge returned so far;
/// 3. stops if `γ̂ - (-f̃*(-Aw_t)) ≤ ε/2`; `h_t` is then discarded (before
///    the first hypothesis, the objective is taken to be `-1`);
/// 4. otherwise adds `h_t` and computes two candidates:
///    - Frank-Wolfe: `w_t + λ_t (e_{h_t} - w_t)` with the step rule set by
///      [`update_rule`](Self::update_rule) (`LineSearch` by default,
///      `ShortStep`, or `Classic` with `λ_t = 2/(t + 2)`); the first round
///      sets `w = e_{h_1}`;
///    - LPBoost: the optimal weights of the soft-margin LP over all
///      hypotheses collected so far;
/// 5. keeps the candidate with the larger `-f̃*(-Aw)` (ties go to
///    Frank-Wolfe) and sets `d_{t+1}` to the minimizer above, i.e.
///    `d_{t+1,i} ∝ exp(-η (Aw_{t+1})_i)` capped at `1/ν`.
///
/// After stopping, [`Booster::postprocess`] solves the soft-margin LP over
/// all collected hypotheses and returns the resulting [`WeightedMajority`].
///
/// Implementation choices: the paper's output is `Σ_j w_{T,j} h_j`; this
/// implementation instead re-fits the LP, which cannot decrease the
/// soft-margin objective over the same hypotheses (up to solver accuracy).
/// The run is capped at `max(1, ceil(32 ln(m/ν)/ε²))` rounds (see
/// [`max_loop`](Self::max_loop)). The LP subproblems are solved with the
/// Clarabel interior-point solver through [`ColumnGeneration`]. One margin
/// column per accepted hypothesis is cached (`O(mT)` memory) so earlier
/// hypotheses are never evaluated again. Line search uses bisection to an
/// interval width of `1e-10`. If `ν = m`, the only feasible distribution is
/// uniform; the first hypothesis is kept after one round, even if its edge is
/// nonpositive.
///
/// # Theoretical guarantee
///
/// Theorem 3 of "Boosting as Frank-Wolfe" (arXiv numbering): if the weak
/// learner returns, on every distribution `d_t`, a hypothesis with edge at
/// least some unknown `g`, and the Frank-Wolfe step is Classic or ShortStep,
/// then Algorithm 1 reaches an `ε`-accurate solution of the soft-margin
/// problem (soft-margin objective at least `g - ε`) within
/// `O(ln(m/ν)/ε²)` rounds, for any secondary update. The paper notes that
/// line search makes at least as much progress per round as the short step,
/// so the same bound holds for `LineSearch`. The stopping test
/// `γ̂ + f̃*(-Aw_t) ≤ ε/2` certifies this accuracy under the same assumption.
/// This matches the bounds of ERLPBoost and Corrective ERLPBoost, whereas
/// LPBoost can need `Ω(m)` rounds.
///
/// # Limitations
///
/// - The weak-learning assumption is not checked at runtime; a decision tree
///   does not by itself ensure it.
/// - `tolerance` is an optimization accuracy on the soft-margin objective,
///   not a classification-error threshold.
/// - `nu` caps every example weight at `1 / nu`; it does not guarantee a
///   particular number of margin errors.
/// - Labels must be `-1` or `+1` with both classes present; preprocessing
///   panics otherwise. Weak hypotheses whose margins are non-finite or
///   outside `[-1, 1]` cause a panic.
/// - Custom hypotheses must predict consistently on a fixed sample, because
///   training margins are cached.
/// - LP solver failures **panic**. Each round solves an LP whose size grows
///   with the number of hypotheses, and the margin cache uses `O(mT)`
///   memory for `T` accepted hypotheses.
/// - `FwUpdateRule::BlendedPairwise` is not supported.
/// - [`CurrentHypothesis::current_hypothesis`] (used by loggers) uses the
///   chosen Frank-Wolfe or LPBoost weights, which can differ from the final
///   LP weights returned by [`Booster::run`].
///
/// # Parameters
///
/// | Builder method | Meaning | Default |
/// | --- | --- | --- |
/// | [`nu`](MlpBoost::nu) | capping parameter `ν ∈ [1, m]` | `1` ([`DEFAULT_CAPPING`]) |
/// | [`tolerance`](MlpBoost::tolerance) | accuracy `ε`; sets `η`, the stopping gap `ε/2`, and the budget | `1e-3` ([`DEFAULT_TOLERANCE`]) |
/// | [`update_rule`](MlpBoost::update_rule) | Frank-Wolfe step rule | [`FwUpdateRule::LineSearch`] |
///
/// # Reference
///
/// Ryotaro Mitsuboshi, Kohei Hatano, and Eiji Takimoto.
/// [Boosting as Frank-Wolfe](https://arxiv.org/abs/2209.10831).
/// arXiv:2209.10831.
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
/// // regularization, the stopping gap 0.05, and the budget.
/// let mut booster = MlpBoost::init(&train)
///     .nu(2.0)
///     .tolerance(0.1)
///     .update_rule(FwUpdateRule::LineSearch);
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
/// println!("MlpBoost test accuracy: {accuracy}");
///
/// // Budget ceil(32 ln(m/ν) / ε²) = ceil(32 ln 4 / 0.01) = 4437 rounds;
/// // `terminated` is the round at which training stopped.
/// assert_eq!(booster.max_loop(), 4437);
/// let rounds = booster.terminated();
/// assert!(rounds >= 1 && rounds <= booster.max_loop());
/// assert!(f.hypotheses.len() <= rounds);
/// println!("stopped at round {rounds}");
/// // The final LP weights are nonnegative and normalized.
/// assert!(f.weights.iter().all(|&w| w > 0.0));
/// assert!((f.weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
/// ```
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
    // One training margin column per accepted hypothesis (O(mT) memory).
    margin_columns: Vec<Vec<f64>>,

    terminated: usize,
    max_iter: usize,

    gamma: f64,
}

impl<'a, F> MlpBoost<'a, F> {
    /// Constructs a new instance of `MlpBoost` for `sample` with the default
    /// capping parameter `nu = 1`, `tolerance = 1e-3`, and line-search
    /// Frank-Wolfe updates.
    ///
    /// See the discussion after Theorem 3 in
    /// <https://arxiv.org/abs/2209.10831>: exact line search is at least as
    /// good as the short-step candidate on the same segment. This
    /// implementation uses the shared numerical line-search tolerance
    /// (`1e-10`). Classic and ShortStep remain available through
    /// [`Self::update_rule`].
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
    /// let mut booster = MlpBoost::init(&sample).tolerance(0.1);
    /// let f = booster.run(&learner);
    /// assert_eq!(f.predict_all(&sample).len(), 6);
    /// ```
    pub fn init(sample: &'a Sample) -> Self {
        let n_examples = sample.shape().0;
        assert!(n_examples != 0);

        let half_tolerance = DEFAULT_TOLERANCE / 2f64;
        let nu = DEFAULT_CAPPING;
        let eta = (n_examples as f64 / nu).ln() / half_tolerance;

        let update_rule = FwUpdateRule::LineSearch;
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
            margin_columns: Vec::new(),
            dist: Vec::new(),

            terminated: usize::MAX,
            max_iter: usize::MAX,

            gamma: 1f64,
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
    /// let mut booster = MlpBoost::init(&sample).nu(3.0).tolerance(0.1);
    /// let f = booster.run(&learner);
    /// assert!(!f.hypotheses.is_empty());
    /// ```
    pub fn nu(mut self, nu: f64) -> Self {
        checkers::capping_parameter(nu, self.n_examples);
        self.nu = nu;

        self
    }

    /// Sets the Frank-Wolfe step-size rule of the primary update.
    ///
    /// Supports [`FwUpdateRule::Classic`], [`FwUpdateRule::ShortStep`], and
    /// [`FwUpdateRule::LineSearch`] (default).
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
    /// // The classic step size 2 / (t + 2) instead of line search.
    /// let mut booster = MlpBoost::init(&sample)
    ///     .tolerance(0.1)
    ///     .update_rule(FwUpdateRule::Classic);
    /// let f = booster.run(&learner);
    /// assert!(!f.hypotheses.is_empty());
    /// ```
    pub fn update_rule(mut self, update_rule: FwUpdateRule) -> Self {
        assert!(
            !matches!(update_rule, FwUpdateRule::BlendedPairwise),
            "BlendedPairwise is not supported by this booster",
        );
        self.update_rule = update_rule;
        self
    }

    /// Sets the soft-margin objective tolerance `ε`.
    ///
    /// Half of `ε` is spent on the entropy regularization
    /// (`η = (2/ε) ln(m/ν)`) and half on the stopping gap; `ε` also
    /// determines the iteration budget. It is not a classification-error
    /// threshold. Default: `1e-3`.
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
    /// let mut booster = MlpBoost::init(&sample).tolerance(0.2);
    /// // Budget ceil(32 ln(6) / 0.2²) = 1434 rounds.
    /// assert_eq!(booster.max_loop(), 1434);
    /// let f = booster.run(&learner);
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
        self.half_tolerance = tolerance / 2f64;
        self
    }

    /// Sets `self.eta = ln(m/ν) / (ε/2)`, or `1` when `ν = m`.
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
        assert!(
            self.eta.is_finite() && self.eta > 0.0,
            "regularization parameter must be finite and positive"
        );
    }

    fn initialize_objective(&mut self) {
        self.objective = ErlpSoftMarginObjective::new(self.nu, self.eta);
    }

    /// Initializes the Frank-Wolfe objective and the LP solver.
    ///
    /// Time complexity: `O( # of training examples )`.
    fn initialize_solver(&mut self) {
        self.initialize_objective();

        let objective = CorrErlpFwObjective::new(self.objective.clone());
        self.frank_wolfe = FrankWolfe::new(objective, self.update_rule);

        self.lpboost.initialize(self.n_examples, self.nu);
    }

    /// Returns the iteration budget `max(1, ceil(32 ln(m/ν) / ε²))` for the
    /// configured `nu` and tolerance `ε`.
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
    /// use miniboosts::DecisionTreeClassifier;
    ///
    /// // The hypothesis type must be known before training.
    /// let booster = MlpBoost::<DecisionTreeClassifier>::init(&sample)
    ///     .nu(2.0)
    ///     .tolerance(0.2);
    /// // m = 6: ceil(32 ln(6/2) / 0.2²) = ceil(878.9) = 879.
    /// assert_eq!(booster.max_loop(), 879);
    /// ```
    pub fn max_loop(&self) -> usize {
        let ln_m = (self.n_examples as f64 / self.nu).ln();
        ((8f64 * ln_m / self.half_tolerance.powi(2)).ceil() as usize).max(1)
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
    /// let mut booster = MlpBoost::init(&sample).tolerance(0.1);
    /// assert_eq!(booster.terminated(), usize::MAX); // not trained yet
    /// let _f = booster.run(&learner);
    /// // Below the budget iff the gap criterion stopped training.
    /// assert!(booster.terminated() >= 1 && booster.terminated() <= booster.max_loop());
    /// ```
    pub fn terminated(&self) -> usize {
        self.terminated
    }
}

impl<F> MlpBoost<'_, F>
where
    F: Classifier,
{
    // Preserve hypothesis order when summing, as in RefWeightedMajority.
    // Algorithm 1 of Boosting as Frank-Wolfe uses Aw; cached columns of A
    // avoid repeated calls to old hypotheses without changing that expression.
    fn negative_margins(&self, weights: &[f64]) -> Vec<f64> {
        assert!(weights.len() <= self.margin_columns.len());
        (0..self.n_examples)
            .map(|row| {
                -weights
                    .iter()
                    .zip(&self.margin_columns)
                    .map(|(w, column)| w * column[row])
                    .sum::<f64>()
            })
            .collect()
    }

    /// Returns the regularized soft-margin objective `-f̃*(-Aw)` at
    /// `weights`, or `-1` if there are no hypotheses yet.
    ///
    /// ```text
    /// -f̃*(-Aw) = min_d [ d^T Aw + (1/eta) sum_i d_i ln(m d_i) ]
    /// s.t. sum_i d_i = 1, 0 <= d_i <= 1 / nu, for all i <= m.
    /// ```
    ///
    /// Time complexity: `O(examples * hypotheses)` with no hypothesis inference.
    fn objval(&self, weights: &[f64]) -> f64 {
        assert_eq!(weights.len(), self.hypotheses.len());
        if weights.is_empty() {
            return -1f64;
        }
        let neg_margins = self.negative_margins(weights);
        self.objective.objective_value(&neg_margins[..])
    }

    /// Sets `self.dist` to the regularized minimizer at the current weights.
    fn update_distribution_mut(&mut self) {
        let neg_margins = self.negative_margins(&self.weights);
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
            // Algorithm 1 initializes w_1 with one atom; its first subsequent
            // Classic update is t=1, lambda=2/(t+2)=2/3, not lambda=1.
            self.frank_wolfe.iteration(1);
            let weight = vec![1f64];
            let objval = self.objval(&weight[..]);
            return (objval, weight);
        }

        let cur_margins = self.negative_margins(&self.weights);
        let new_margins: Vec<_> = self
            .margin_columns
            .last()
            .unwrap()
            .iter()
            .map(|m| -m)
            .collect();

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
    F: Classifier + Clone,
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
        self.margin_columns = Vec::new();
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
            self.margin_columns.push(new_margins);
            self.hypotheses.push(h);
            self.weights = vec![1.0];
            self.terminated = iteration;
            return ControlFlow::Break(iteration);
        }

        let edge = helpers::edge_from_margins(&new_margins, &self.dist);

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
        self.margin_columns.push(new_margins);
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
    fn default_line_search_is_no_worse_than_short_step_on_the_same_segment() {
        let sample = Sample::dummy(3);
        let mut line = MlpBoost::init(&sample);
        let mut short = MlpBoost::init(&sample).update_rule(FwUpdateRule::ShortStep);
        assert!(matches!(line.update_rule, FwUpdateRule::LineSearch));
        for booster in [&mut line, &mut short] {
            booster.preprocess();
            assert!(booster.boost(&Margins([1.0, 1.0, -1.0]), 1).is_continue());
            booster.hypotheses.push(Margins([-1.0, 1.0, 1.0]));
            booster.margin_columns.push(vec![-1.0, 1.0, 1.0]);
        }
        let before = line
            .objective
            .objective_value(&line.negative_margins(&line.weights));
        let (line_score, line_weights) = line.update_frank_wolfe();
        let (short_score, _) = short.update_frank_wolfe();
        // The binary line search locates lambda within 1e-10. A bounded
        // margin segment has objective slope at most 2, so allow 3e-10.
        assert!(line_score + 3e-10 >= short_score);
        assert!(line_score + 3e-10 >= before);
        assert!(line_weights.iter().all(|w| w.is_finite() && *w >= 0.0));
        assert!((line_weights.iter().sum::<f64>() - 1.0).abs() < 1e-7);
    }

    #[test]
    fn rejects_nonfinite_oracle_and_unrepresentable_eta() {
        let sample = Sample::dummy(3);
        let mut tiny = MlpBoost::<Margins>::init(&sample).tolerance(f64::from_bits(2));
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tiny.preprocess())).is_err()
        );
        for nu in [1.0, 3.0] {
            let mut booster = MlpBoost::init(&sample).nu(nu);
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
    fn classic_schedule_starts_after_the_initial_atom() {
        let sample = Sample::dummy(3);
        let mut booster = MlpBoost::init(&sample).update_rule(FwUpdateRule::Classic);
        booster.preprocess();
        assert!(booster.boost(&Margins([1.0, 1.0, -1.0]), 1).is_continue());
        let previous_weight = booster.weights[0];
        booster.hypotheses.push(Margins([-1.0, 1.0, 1.0]));
        booster.margin_columns.push(vec![-1.0, 1.0, 1.0]);
        let (_, weights) = booster.update_frank_wolfe();
        assert!((weights[0] - previous_weight / 3.0).abs() < 1e-15);
        assert!((weights[1] - 2.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn rejects_invalid_tolerance() {
        let sample = Sample::dummy(3);
        for tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::from_bits(1)] {
            assert!(std::panic::catch_unwind(|| MlpBoost::<Margins>::init(&sample).tolerance(tolerance)).is_err());
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
            let mut booster = MlpBoost::init(&sample).update_rule(rule).tolerance(0.01);
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
                    assert!(calls.get() <= 6, "old hypotheses were evaluated again");
                    let model = booster.current_hypothesis();
                    let direct: Vec<_> = helpers::margins(&sample, &model).collect();
                    let cached: Vec<f64> = booster
                        .negative_margins(&booster.weights)
                        .iter()
                        .map(|m| -m)
                        .collect();
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
        candidates.margin_columns.push(second.0.to_vec());
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
