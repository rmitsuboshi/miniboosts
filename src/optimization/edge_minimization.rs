//! Edge minimization over the capped simplex and the row-generation solver of
//! ERLPBoost.
//!
//! For `m` examples, hypotheses `h_1, ..., h_T`, and capping parameter
//! `ν ∈ [1, m]`, let `Δ_{m,ν} = { d : Σ_i d_i = 1, 0 ≤ d_i ≤ 1/ν }` and
//! `edge_j(d) = Σ_i d_i y_i h_j(x_i)`.
//!
//! - [`edge_minimization`] solves the linear program
//!   `min_{d ∈ Δ_{m,ν}} max_j edge_j(d)`, the LP dual of the soft-margin
//!   problem in [`soft_margin_optimization`](super::soft_margin_optimization()).
//! - [`RowGeneration`] solves the regularized restricted problem
//!   `min_{d ∈ Δ_{m,ν}} max_j edge_j(d) + R(d)` for a
//!   [`RowGenerationObjective`]. With [`EntropyRegularizedMaxEdge`],
//!   `R(d) = KL(d ‖ u)/η` with `u` the uniform distribution; this is the
//!   subproblem of ERLPBoost (Warmuth, Glocer and Vishwanathan, 2008,
//!   "Entropy regularized LPBoost", Algorithm 1,
//!   <https://mwarmuth.bitbucket.io/pubs/C83.pdf>).
use crate::core::{
    Classifier, Sample,
    constants::{DEFAULT_CAPPING, PERTURBATION, SQP_TOLERANCE},
    tools::checkers,
    tools::helpers,
};

use clarabel::{algebra::*, solver::*};

use std::iter;

/// Solves the (unregularized) edge-minimization LP and returns
/// `(optimal value γ, distribution d on examples)`.
///
/// ```txt
/// min γ
/// γ,d
/// s.t. Σ_i d_i y_i h_j (x_i) ≤ γ,   ∀j = 1, 2, ..., T
///      Σ_i d_i = 1,
///      d_1, d_2, ..., d_m ≤ 1/ν
///      d_1, d_2, ..., d_m ≥ 0.
/// ```
/// This is the LP dual of the soft-margin problem, so its optimal value equals
/// that of [`soft_margin_optimization`](super::soft_margin_optimization()) for
/// the same inputs. The returned `γ` is Clarabel's primal objective value
/// (not recomputed from `d`). The returned `d` is the solver's distribution
/// with negative roundoff clipped to zero (not renormalized).
///
/// # Panics
/// - if `ν ∉ [1, m]`;
/// - unless Clarabel returns exactly `Solved` (`AlmostSolved` is rejected);
///   in particular, an empty `hypotheses` makes `γ` unbounded below, which is
///   not `Solved`;
/// - if the clipped `d` is not in `Δ_{m,ν}` up to an absolute sum tolerance
///   of `1e-5` and an upper-bound tolerance of `1e-5`.
///
/// No residual or duality-gap check is performed beyond the solver status.
/// No error value is returned.
///
/// # Formulation
/// To solve the problem we build the constraint matrix
/// ```txt
/// # of   
/// rows    γ        d1        ...     dm
///       ┏    ┃                                 ┓   ┏     ┓
///   1   ┃  0 ┃      1        ...      1        ┃ = ┃  1  ┃
///      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
///       ┃  0 ┃                                 ┃ ≤ ┃  0  ┃
///       ┃  0 ┃                                 ┃ ≤ ┃  0  ┃
///       ┃  . ┃     (-1) * Identity matrix      ┃ . ┃  .  ┃
///   m   ┃  . ┃              m x m              ┃ . ┃  .  ┃
///       ┃  . ┃                                 ┃ . ┃  .  ┃
///       ┃  0 ┃                                 ┃ ≤ ┃  0  ┃
///      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
///       ┃  0 ┃                                 ┃ ≤ ┃ 1/ν ┃
///       ┃  0 ┃                                 ┃ ≤ ┃ 1/ν ┃
///   m   ┃  . ┃         Identity matrix         ┃ . ┃  .  ┃
///       ┃  . ┃              m x m              ┃ . ┃  .  ┃
///       ┃  . ┃                                 ┃ . ┃  .  ┃
///       ┃  0 ┃                                 ┃ ≤ ┃ 1/ν ┃
///      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
///       ┃ -1 ┃  y_1 h_1(x_1) ...  y_m h_1(x_m) ┃ ≤ ┃  0  ┃
///       ┃ -1 ┃  y_1 h_2(x_1) ...  y_m h_2(x_m) ┃ ≤ ┃  0  ┃
///       ┃  . ┃      .        ...      .        ┃ . ┃  .  ┃
///   H   ┃  . ┃      .        ...      .        ┃ . ┃  .  ┃
///       ┃  . ┃      .        ...      .        ┃ . ┃  .  ┃
///       ┃ -1 ┃  y_1 h_T(x_1) ...  y_m h_T(x_m) ┃ ≤ ┃  0  ┃
///       ┗    ┃                                 ┛   ┗     ┛
///
/// # of
/// cols     1 ┃               m
/// ```
///
/// # Examples
/// The edge of `h_j` is `1 - 2 d_k`, where `k` is the example `h_j`
/// misclassifies. Requiring all three edges to be at most `γ` forces
/// `d_2, d_3, d_4 ≥ (1 - γ)/2`, so the optimum is `γ = 1/3` at the unique
/// `d = (0, 1/3, 1/3, 1/3)`, matching
/// [`soft_margin_optimization`](super::soft_margin_optimization()) by LP
/// duality. The cap `1/ν = 1/2` is inactive, so `ν = 2` gives the same
/// answer.
/// ```
/// use miniboosts::{Classifier, Sample};
///
/// /// A hypothesis with fixed margins `y_i h(x_i)`, one per example.
/// struct Fixed([f64; 4]);
///
/// impl Classifier for Fixed {
///     fn confidence(&self, sample: &Sample, row: usize) -> f64 {
///         self.0[row] * sample.target()[row]
///     }
/// }
///
/// // Four examples; the features are ignored by `Fixed`.
/// let sample = Sample::dummy(4);
/// // Each hypothesis misclassifies exactly one of examples 2, 3, 4.
/// let hypotheses = [
///     Fixed([1.0, 1.0, 1.0, -1.0]),
///     Fixed([1.0, 1.0, -1.0, 1.0]),
///     Fixed([1.0, -1.0, 1.0, 1.0]),
/// ];
///
/// use miniboosts::{edge_minimization, soft_margin_optimization};
///
/// for nu in [1.0, 2.0] {
///     let (gamma, d) = edge_minimization(nu, &sample, &hypotheses);
///     // Errors are about 1e-9; 1e-6 allows for the interior-point accuracy.
///     assert!((gamma - 1.0 / 3.0).abs() < 1e-6);
///     for (di, expected) in d.iter().zip([0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]) {
///         assert!((di - expected).abs() < 1e-6);
///     }
///
///     // Strong duality with the soft-margin LP.
///     let (rho, _) = soft_margin_optimization(nu, &sample, &hypotheses);
///     assert!((gamma - rho).abs() < 1e-6);
/// }
/// ```
pub fn edge_minimization<T, H>(nu: f64, sample: &Sample, hypotheses: T) -> (f64, Vec<f64>)
where
    T: AsRef<[H]>,
    H: Classifier,
{
    let hypotheses = hypotheses.as_ref();
    let n_hypotheses = hypotheses.len();
    let n_examples = sample.shape().0;

    checkers::capping_parameter(nu, n_examples);

    let obj = build_objective(n_examples + 1);
    let mat = build_constraint_matrix(sample, hypotheses);
    let sns = build_sns(n_examples, n_hypotheses);
    let rhs = build_rhs(nu, n_examples, n_hypotheses);

    let mut solver = build_solver(obj, mat, sns, rhs);
    solver.solve();

    assert!(
        matches! { solver.solution.status, SolverStatus::Solved },
        "unexped solver status. got {}",
        solver.solution.status,
    );

    let objval = solver.solution.obj_val;
    let dist = solver.solution.x[1..]
        .iter()
        .map(|w| w.max(0f64))
        .collect::<Vec<f64>>();

    checkers::capped_simplex_condition(&dist[..], nu);

    (objval, dist)
}

fn build_objective(n_variables: usize) -> Vec<f64> {
    let mut objective = vec![0f64; n_variables];
    objective[0] = 1f64;
    objective
}

/// Build the constraint matrix:
/// ```txt
/// # of   
/// rows    γ        d1        ...     dm
///       ┏    ┃                                 ┓
///   1   ┃  0 ┃      1        ...      1        ┃
///      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
///       ┃  0 ┃                                 ┃
///       ┃  0 ┃                                 ┃
///       ┃  . ┃     (-1) * Identity matrix      ┃
///   m   ┃  . ┃              m x m              ┃
///       ┃  . ┃                                 ┃
///       ┃  0 ┃                                 ┃
///      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
///       ┃  0 ┃                                 ┃
///       ┃  0 ┃                                 ┃
///   m   ┃  . ┃         Identity matrix         ┃
///       ┃  . ┃              m x m              ┃
///       ┃  . ┃                                 ┃
///       ┃  0 ┃                                 ┃
///      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
///       ┃ -1 ┃  y_1 h_1(x_1) ...  y_m h_1(x_m) ┃
///       ┃ -1 ┃  y_1 h_2(x_1) ...  y_m h_2(x_m) ┃
///       ┃  . ┃      .        ...      .        ┃
///   H   ┃  . ┃      .        ...      .        ┃
///       ┃  . ┃      .        ...      .        ┃
///       ┃ -1 ┃  y_1 h_T(x_1) ...  y_m h_T(x_m) ┃
///       ┗    ┃                                 ┛
///
/// # of
/// cols     1 ┃               m
/// ```
fn build_constraint_matrix<H>(sample: &Sample, hypotheses: &[H]) -> CscMatrix<f64>
where
    H: Classifier,
{
    let n_hypotheses = hypotheses.len();
    let n_examples = sample.shape().0;

    let start = 1 + 2 * n_examples;
    let end = 1 + 2 * n_examples + n_hypotheses;

    let mut col_ptr = vec![0];
    let mut row_idx = (start..end).collect::<Vec<_>>();
    let mut nonzero = vec![-1f64; n_hypotheses];

    let offset = 1 + 2 * n_examples;
    let y = sample.target();
    for (i, yi) in y.iter().enumerate() {
        col_ptr.push(row_idx.len());

        // `Σ d[i] = 1.`
        row_idx.push(0);
        nonzero.push(1f64);

        // `-d[i] ≤ 0`, i.e., `d[i] ≥ 0.`
        row_idx.push(1 + i);
        nonzero.push(-1f64);

        // `d[i] ≤ 1/ν.`
        row_idx.push(1 + n_examples + i);
        nonzero.push(1f64);

        for (j, h) in hypotheses.iter().enumerate() {
            let hx = h.confidence(sample, i);
            let yh = yi * hx;

            row_idx.push(offset + j);
            nonzero.push(yh);
        }
    }
    col_ptr.push(row_idx.len());

    assert_eq!(col_ptr.len(), n_examples + 2);

    let n_cols = n_examples + 1;
    let n_rows = 1 + 2 * n_examples + n_hypotheses;
    CscMatrix::new(n_rows, n_cols, col_ptr, row_idx, nonzero)
}

fn build_sns(n_examples: usize, n_hypotheses: usize) -> Vec<SupportedConeT<f64>> {
    vec![
        ZeroConeT(1),
        NonnegativeConeT(2 * n_examples + n_hypotheses),
    ]
}

fn build_rhs(nu: f64, n_examples: usize, n_hypotheses: usize) -> Vec<f64> {
    iter::once(1f64)
        .chain(iter::repeat_n(0f64, n_examples))
        .chain(iter::repeat_n(1f64 / nu, n_examples))
        .chain(iter::repeat_n(0f64, n_hypotheses))
        .collect()
}

fn build_solver(
    obj: Vec<f64>,
    mat: CscMatrix<f64>,
    sns: Vec<SupportedConeT<f64>>,
    rhs: Vec<f64>,
) -> DefaultSolver<f64> {
    let settings = DefaultSettingsBuilder::default()
        .equilibrate_enable(true)
        .verbose(false)
        .build()
        .unwrap();

    let n_variables = obj.len();
    let zmx = CscMatrix::<f64>::zeros((n_variables, n_variables));

    DefaultSolver::new(&zmx, &obj, &mat, &rhs, &sns, settings)
        .expect("failed to construct the LP solver: invalid dimensions or settings")
}

/// Solver for the regularized restricted problem of row-generation boosting
/// (ERLPBoost) over the capped simplex `Δ_{m,ν}`.
///
/// For the hypotheses passed to [`solve`](Self::solve) it computes
/// ```txt
/// min_{γ, d}  γ + R(d)
/// s.t.        Σ_i d_i y_i h_j(x_i) ≤ γ,   j = 1, ..., T,
///             d ∈ Δ_{m,ν},
/// ```
/// i.e. `min_{d ∈ Δ_{m,ν}} max_j edge_j(d) + R(d)`, where `R` is defined by
/// the [`RowGenerationObjective`] `T`. The primal solution is a distribution
/// on examples; the multipliers of the edge constraints are weights on
/// hypotheses.
///
/// # Solution methods
/// **Entropy path.** If [`RowGenerationObjective::entropy_parameter`]
/// returns `Some(η)` (as for [`EntropyRegularizedMaxEdge`]), then
/// `R(d) = KL(d ‖ u)/η` is modeled exactly with exponential cones
/// (`t_i ≥ d_i ln(m d_i)`) and solved once by Clarabel; the objective's
/// `gradient` and `hessian` are not used. The result is accepted only if
/// - Clarabel returns `Solved` or `AlmostSolved`;
/// - all solution values and diagnostics are finite and the primal and dual
///   residuals are at most `1e-7`;
/// - the raw distribution satisfies `-1e-7 ≤ d_i ≤ 1/ν + 1e-7` and
///   `|Σ_i d_i - 1| ≤ 1e-7`, and the raw weights are `≥ -1e-7` with
///   `|Σ_j w_j - 1| ≤ 1e-7`;
/// - after repairing `d` into `Δ_{m,ν}` and normalizing `w`, the recomputed
///   primal value `max_j edge_j(d) + KL(d ‖ u)/η` exceeds an independent
///   dual lower bound by a gap in `[-1e-10, 1e-7]`.
///
/// The dual bound is exposed by
/// [`objective_lower_bound`](Self::objective_lower_bound). If `ν == m`
/// exactly, the feasible set is the uniform distribution and it is returned
/// without calling Clarabel, with all weight on a hypothesis of maximal edge.
///
/// **Generic SQP path.** Otherwise a sequence of QPs is solved, starting from
/// the uniform distribution `d_0`. At iterate `d_k` the QP minimizes
/// `g'x + x'Hx/2` over `x = (γ, d)` subject to the constraints above, with
/// `g = gradient(d_k)` and `H = hessian(d_k)` applied to the variables
/// themselves (not to the displacement `x - x_k`). A full step is accepted if
/// it does not increase `objective_value` (up to a roundoff allowance when the
/// QP is stationary); otherwise up to 50 step halvings are tried. Iteration
/// stops when an undamped step decreases the objective by less than `1e-10`.
/// Each QP must return `Solved` or `AlmostSolved` with finite values,
/// residuals at most `1e-7`, a relative QP duality gap at most `1e-7`, and a
/// distribution and edge constraints satisfied within `1e-7`. This path
/// checks only these QP diagnostics; it does not certify optimality of the
/// original objective, and it is a numerical method, not the exact
/// subproblem solution assumed by the ERLPBoost analysis.
///
/// # Solver failure
/// All failures **panic**; no error value is returned. This includes
/// non-finite margins, an empty hypothesis set, a rejected solver status or
/// failed check above, a failed line search, and more than 1000 SQP
/// iterations. [`solve`](Self::solve) clears the previous solution first, so
/// if a panic is caught the solver exposes no solution
/// ([`optval`](Self::optval) is `f64::MAX`).
///
/// # Examples
/// With a single hypothesis the edge term is linear, so the minimizer over
/// the simplex (`ν = 1`, no active cap) is the Gibbs distribution
/// `d_i ∝ exp(-η y_i h(x_i))` and the optimal value is
/// `-(1/η) ln((1/m) Σ_i exp(-η y_i h(x_i)))`.
///
/// The solver certifies the objective gap (about `1e-11` here) rather than
/// each coordinate. Because `KL(· ‖ u)/η` is `(1/η)`-strongly convex in the
/// ℓ1-norm, a gap `g` bounds `‖d - d*‖₁ ≤ sqrt(2ηg)`, so coordinates are
/// compared with tolerance `1e-5`, the value with `1e-6`.
/// ```
/// use miniboosts::{Classifier, Sample};
///
/// /// A hypothesis with fixed margins `y_i h(x_i)`, one per example.
/// struct Fixed([f64; 4]);
///
/// impl Classifier for Fixed {
///     fn confidence(&self, sample: &Sample, row: usize) -> f64 {
///         self.0[row] * sample.target()[row]
///     }
/// }
///
/// // Four examples; the features are ignored by `Fixed`.
/// let sample = Sample::dummy(4);
/// // Each hypothesis misclassifies exactly one of examples 2, 3, 4.
/// let hypotheses = [
///     Fixed([1.0, 1.0, 1.0, -1.0]),
///     Fixed([1.0, 1.0, -1.0, 1.0]),
///     Fixed([1.0, -1.0, 1.0, 1.0]),
/// ];
///
/// use miniboosts::{EntropyRegularizedMaxEdge, RowGeneration};
///
/// let eta = 1.0;
/// let mut solver = RowGeneration::new(&sample, EntropyRegularizedMaxEdge::new(eta));
/// solver.solve(&hypotheses[..1]);
///
/// let margins = [1.0, 1.0, 1.0, -1.0];
/// let exps: Vec<f64> = margins.iter().map(|m: &f64| (-eta * m).exp()).collect();
/// let total: f64 = exps.iter().sum();
/// let value = -(total / 4.0).ln() / eta;
/// assert!((solver.optval() - value).abs() < 1e-6);
/// // The dual lower bound certifies the value.
/// let lower = solver.objective_lower_bound().unwrap();
/// assert!(lower <= solver.optval() + 1e-10 && solver.optval() - lower < 1e-7);
///
/// let d = solver.distribution_on_examples();
/// for (di, e) in d.iter().zip(&exps) {
///     assert!((di - e / total).abs() < 1e-5);
/// }
/// assert_eq!(solver.weights_on_hypotheses(), vec![1.0]);
///
/// // `objval` re-evaluates the objective at any distribution.
/// assert!((solver.objval(&hypotheses[..1], &d) - solver.optval()).abs() < 1e-9);
/// ```
pub struct RowGeneration<'a, T> {
    nu: f64,
    sample: &'a Sample,

    objective: T,

    optval: f64,
    lower_bound: Option<f64>,
    primal: Vec<f64>,
    dual: Vec<f64>,
}

impl<'a, T> RowGeneration<'a, T> {
    /// Creates a solver for `sample` with the given objective and capping
    /// parameter `ν = 1` (the default capping). Use
    /// [`initialize`](Self::initialize) to set `ν`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{EntropyRegularizedMaxEdge, RowGeneration, Sample};
    ///
    /// let sample = Sample::dummy(4);
    /// let solver = RowGeneration::new(&sample, EntropyRegularizedMaxEdge::new(10.0));
    /// // Nothing is solved yet.
    /// assert_eq!(solver.optval(), f64::MAX);
    /// assert_eq!(solver.objective_lower_bound(), None);
    /// assert!(solver.distribution_on_examples().is_empty());
    /// ```
    pub fn new(sample: &'a Sample, objective: T) -> Self {
        Self {
            nu: DEFAULT_CAPPING,
            sample,
            objective,

            optval: f64::MAX,
            lower_bound: None,
            primal: Vec::new(),
            dual: Vec::new(),
        }
    }

    /// Sets the capping parameter `ν`, replaces the objective, and discards
    /// any previous solution.
    ///
    /// # Panics
    /// Panics if `ν ∉ [1, m]`, where `m` is the number of examples.
    ///
    /// # Examples
    /// With `ν = 2` the Gibbs weight `e/(3e⁻¹ + e) ≈ 0.71` of example 4 (see
    /// [`RowGeneration`]) exceeds the cap `1/2`. The KL projection fixes it
    /// at `1/2` and spreads the rest uniformly over the tied examples 1-3, so
    /// `d = (1/6, 1/6, 1/6, 1/2)`, with value `0 + KL(d ‖ u) = ln(4/3)/2`.
    /// ```
    /// # use miniboosts::{Classifier, Sample};
    /// # struct Fixed([f64; 4]);
    /// # impl Classifier for Fixed {
    /// #     fn confidence(&self, sample: &Sample, row: usize) -> f64 {
    /// #         self.0[row] * sample.target()[row]
    /// #     }
    /// # }
    /// # let sample = Sample::dummy(4);
    /// # let hypotheses = [
    /// #     Fixed([1.0, 1.0, 1.0, -1.0]),
    /// #     Fixed([1.0, 1.0, -1.0, 1.0]),
    /// #     Fixed([1.0, -1.0, 1.0, 1.0]),
    /// # ];
    /// use miniboosts::{EntropyRegularizedMaxEdge, RowGeneration};
    ///
    /// let mut solver = RowGeneration::new(&sample, EntropyRegularizedMaxEdge::new(1.0));
    /// solver.initialize(2.0, EntropyRegularizedMaxEdge::new(1.0));
    /// solver.solve(&hypotheses[..1]);
    ///
    /// assert!((solver.optval() - (4f64 / 3.0).ln() / 2.0).abs() < 1e-6);
    /// let d = solver.distribution_on_examples();
    /// // Coordinate tolerance as justified on `RowGeneration`.
    /// for (di, expected) in d.iter().zip([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.5]) {
    ///     assert!((di - expected).abs() < 1e-5);
    /// }
    /// ```
    pub fn initialize(&mut self, nu: f64, objective: T) {
        let n_examples = self.sample.shape().0;
        checkers::capping_parameter(nu, n_examples);
        self.nu = nu;
        self.objective = objective;
        self.optval = f64::MAX;
        self.lower_bound = None;
        self.primal.clear();
        self.dual.clear();
    }
}

/// An objective `max_j edge_j(d) + R(d)` minimized by [`RowGeneration`].
///
/// The variable vector seen by [`gradient`](Self::gradient) and
/// [`hessian`](Self::hessian) is `x = (γ, d_1, ..., d_m)`, where `γ` is the
/// epigraph variable of the maximum edge. These two methods are used only by
/// the generic SQP path of [`RowGeneration::solve`].
///
/// Implementing [`hessian`](Self::hessian) requires the `clarabel` crate's
/// `CscMatrix` type, which `miniboosts` does not re-export.
///
/// # Examples
/// Evaluating the provided objective directly: at the uniform distribution
/// the entropy term vanishes and the value is the largest edge.
/// ```
/// # use miniboosts::{Classifier, Sample};
/// # struct Fixed([f64; 4]);
/// # impl Classifier for Fixed {
/// #     fn confidence(&self, sample: &Sample, row: usize) -> f64 {
/// #         self.0[row] * sample.target()[row]
/// #     }
/// # }
/// # let sample = Sample::dummy(4);
/// # let hypotheses = [
/// #     Fixed([1.0, 1.0, 1.0, -1.0]),
/// #     Fixed([1.0, 1.0, -1.0, 1.0]),
/// #     Fixed([1.0, -1.0, 1.0, 1.0]),
/// # ];
/// use miniboosts::{EntropyRegularizedMaxEdge, RowGenerationObjective};
///
/// let objective = EntropyRegularizedMaxEdge::new(5.0);
/// assert_eq!(objective.entropy_parameter(), Some(5.0));
///
/// let uniform = [0.25; 4];
/// // Every edge is (1 + 1 + 1 - 1) / 4 = 1/2.
/// let value = objective.objective_value(&sample, &hypotheses, &uniform);
/// assert!((value - 0.5).abs() < 1e-12);
///
/// // `gradient` has one entry for γ followed by one per example.
/// assert_eq!(objective.gradient(&uniform).len(), 5);
/// ```
pub trait RowGenerationObjective {
    /// Opt into the exact Shannon-entropy cone formulation. Return `Some(η)`
    /// only if the objective is exactly `max_j edge_j(d) + KL(d ‖ u)/η` with
    /// `η > 0` finite; [`RowGeneration`] then ignores
    /// [`gradient`](Self::gradient) and [`hessian`](Self::hessian). The
    /// default `None` selects the generic SQP path.
    fn entropy_parameter(&self) -> Option<f64> {
        None
    }

    /// Objective value at distribution `dist` (length `m`) for the given
    /// hypotheses.
    fn objective_value<H: Classifier>(
        &self,
        sample: &Sample,
        hypotheses: &[H],
        dist: &[f64],
    ) -> f64;

    /// Linear coefficients of the QP model at `dist`, a vector of length
    /// `m + 1` whose first entry is the coefficient of `γ` (normally `1`).
    fn gradient(&self, dist: &[f64]) -> Vec<f64>;

    /// Quadratic coefficients of the QP model at `dist`, an
    /// `(m + 1) × (m + 1)` positive semidefinite matrix in CSC format whose
    /// first row and column correspond to `γ`. Clarabel reads only the upper
    /// triangle.
    fn hessian(&self, dist: &[f64]) -> CscMatrix<f64>;
}

impl<T> RowGeneration<'_, T>
where
    T: RowGenerationObjective,
{
    /// Solves the restricted problem over `hypotheses`.
    ///
    /// Objectives with [`RowGenerationObjective::entropy_parameter`] use the
    /// exact exponential-cone formulation; other objectives use the
    /// safeguarded SQP implementation. See [`RowGeneration`] for the accepted
    /// solver statuses and checks.
    ///
    /// # Panics
    /// Panics if `hypotheses` is empty, on non-finite margins or objective
    /// values, on a rejected solver status or failed numerical check, or on
    /// exhausted iteration/line-search limits. Such failures leave no
    /// published solution. The entropy path checks the original primal/dual
    /// gap and exposes its lower bound. Generic SQP only checks its
    /// quadratic-model diagnostics.
    ///
    /// # Examples
    /// Solving the restricted problem over all three hypotheses and checking
    /// the guaranteed structure of the solution:
    /// ```
    /// # use miniboosts::{Classifier, Sample};
    /// # struct Fixed([f64; 4]);
    /// # impl Classifier for Fixed {
    /// #     fn confidence(&self, sample: &Sample, row: usize) -> f64 {
    /// #         self.0[row] * sample.target()[row]
    /// #     }
    /// # }
    /// # let sample = Sample::dummy(4);
    /// # let hypotheses = [
    /// #     Fixed([1.0, 1.0, 1.0, -1.0]),
    /// #     Fixed([1.0, 1.0, -1.0, 1.0]),
    /// #     Fixed([1.0, -1.0, 1.0, 1.0]),
    /// # ];
    /// use miniboosts::{EntropyRegularizedMaxEdge, RowGeneration};
    ///
    /// let mut solver = RowGeneration::new(&sample, EntropyRegularizedMaxEdge::new(1.0));
    /// solver.initialize(2.0, EntropyRegularizedMaxEdge::new(1.0));
    /// solver.solve(&hypotheses);
    ///
    /// let d = solver.distribution_on_examples();
    /// assert!((d.iter().sum::<f64>() - 1.0).abs() < 1e-7);
    /// assert!(d.iter().all(|&di| (0.0..=0.5 + 1e-7).contains(&di)));
    ///
    /// let w = solver.weights_on_hypotheses();
    /// assert_eq!(w.len(), 3);
    /// assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-5);
    ///
    /// // The reported value is certified by the dual lower bound.
    /// let gap = solver.optval() - solver.objective_lower_bound().unwrap();
    /// assert!((-1e-10..=1e-7).contains(&gap));
    /// ```
    pub fn solve<H>(&mut self, hypotheses: &[H])
    where
        H: Classifier,
    {
        let n_hypotheses = hypotheses.len();
        let n_examples = self.sample.shape().0;
        // A failed solve must not expose a previous round's solution.
        self.optval = f64::MAX;
        self.lower_bound = None;
        self.primal.clear();
        self.dual.clear();
        assert!(n_hypotheses > 0, "row generation requires a hypothesis");
        checkers::capping_parameter(self.nu, n_examples);

        let mat = build_constraint_matrix(self.sample, hypotheses);
        assert!(
            mat.nzval.iter().all(|v| v.is_finite()),
            "non-finite hypothesis margin"
        );
        let sns = build_sns(n_examples, n_hypotheses);
        let rhs = build_rhs(self.nu, n_examples, n_hypotheses);

        if let Some(eta) = self.objective.entropy_parameter() {
            let result = crate::optimization::entropy_solver::solve(
                eta,
                self.nu,
                mat,
                rhs,
                sns,
                n_examples,
                n_hypotheses,
            );
            self.optval = result.value;
            self.lower_bound = Some(result.lower_bound);
            self.primal = result.distribution;
            self.dual = result.weights;
            return;
        }

        let mut current = vec![1f64 / n_examples as f64; n_examples];
        let mut objval = self.objval(hypotheses, &current[..]);
        assert!(objval.is_finite(), "non-finite row-generation objective");
        // Warmuth, Glocer and Vishwanathan (2008), Algorithm 1, requires
        // solving the entropy-regularized subproblem. The finite iteration
        // limit and descent safeguard below are numerical safeguards for SQP,
        // not alternative convergence criteria from that paper.
        // https://mwarmuth.bitbucket.io/pubs/C83.pdf
        const MAX_SQP_ITERATIONS: usize = 1000;
        for _ in 0..MAX_SQP_ITERATIONS {
            let set = DefaultSettingsBuilder::default()
                .equilibrate_enable(true)
                .verbose(false)
                .tol_gap_abs(1e-9)
                .tol_gap_rel(1e-9)
                .tol_feas(1e-9)
                .build()
                .unwrap();
            let g = self.objective.gradient(&current[..]);
            let h = self.objective.hessian(&current[..]);
            assert!(
                g.iter().chain(&h.nzval).all(|v| v.is_finite()),
                "non-finite SQP coefficients"
            );
            let mut solver = DefaultSolver::new(&h, &g, &mat, &rhs, &sns, set)
                .expect("failed to construct the row-generation QP solver");

            solver.solve();

            require_candidate_status(solver.solution.status);
            assert!(
                solver
                    .solution
                    .x
                    .iter()
                    .chain(&solver.solution.z)
                    .all(|v| v.is_finite())
                    && solver.solution.obj_val.is_finite()
                    && solver.solution.obj_val_dual.is_finite()
                    && solver.solution.r_prim.is_finite()
                    && solver.solution.r_dual.is_finite(),
                "non-finite SQP solution or diagnostics"
            );
            const FEASIBILITY_TOLERANCE: f64 = 1e-7;
            assert!(
                solver.solution.r_prim <= FEASIBILITY_TOLERANCE
                    && solver.solution.r_dual <= FEASIBILITY_TOLERANCE,
                "SQP residual exceeds numerical tolerance"
            );
            assert!(
                solver.solution.x[1..]
                    .iter()
                    .all(|d| *d >= -FEASIBILITY_TOLERANCE
                        && *d <= 1.0 / self.nu + FEASIBILITY_TOLERANCE),
                "SQP distribution violates bounds"
            );

            let mut solution = solver.solution.x[1..]
                .iter()
                .map(|d| d.clamp(0f64, 1f64 / self.nu))
                .collect::<Vec<_>>();
            checkers::capped_simplex_condition(&solution[..], self.nu);
            assert!(
                (solution.iter().sum::<f64>() - 1.0).abs() <= FEASIBILITY_TOLERANCE,
                "corrected SQP distribution does not sum to one"
            );
            // Reuse the already evaluated margin columns; validation must not
            // re-run every tree on every example during each SQP iteration.
            let first_edge_row = 1 + 2 * n_examples;
            let mut edges = vec![0.0; n_hypotheses];
            let mut current_edges = vec![0.0; n_hypotheses];
            for (i, d) in solution.iter().enumerate() {
                let column = i + 1; // column zero is the epigraph variable
                for k in mat.colptr[column]..mat.colptr[column + 1] {
                    let row = mat.rowval[k];
                    if row >= first_edge_row {
                        edges[row - first_edge_row] += mat.nzval[k] * d;
                        current_edges[row - first_edge_row] += mat.nzval[k] * current[i];
                    }
                }
            }
            for edge in edges {
                assert!(
                    edge.is_finite() && edge <= solver.solution.x[0] + FEASIBILITY_TOLERANCE,
                    "corrected SQP solution violates an edge constraint"
                );
            }
            let mut optval = self.objval(hypotheses, &solution[..]);
            assert!(optval.is_finite(), "non-finite SQP objective");
            // Near stationarity the objective comparison can be dominated by
            // QP solve accuracy, especially at a max-edge kink. Changes within
            // the measured QP duality gap (with a roundoff floor) are numerical
            // equality only if the current point is already optimal for the
            // QP model to that same accuracy. Keep the
            // matched QP primal/dual pair. This allowance is not an exact
            // monotonicity or original entropy-objective gap certificate.
            let qp_gap = (solver.solution.obj_val - solver.solution.obj_val_dual).abs();
            assert!(
                qp_gap <= FEASIBILITY_TOLERANCE * (1.0 + solver.solution.obj_val.abs()),
                "SQP duality gap exceeds numerical tolerance"
            );
            let roundoff = qp_gap.max(64.0 * f64::EPSILON * (1.0 + objval.abs()));
            // Q(x) - Q_dual bounds the remaining model improvement. A fixed
            // coordinate cutoff is inappropriate near a simplex boundary,
            // where objective accuracy and coordinate accuracy differ.
            let current_point: Vec<_> =
                iter::once(current_edges.into_iter().fold(f64::NEG_INFINITY, f64::max))
                    .chain(current.iter().copied())
                    .collect();
            let mut model_value = helpers::inner_product(&g, &current_point);
            // Clarabel reads the upper triangle of H in Q(x)=g'x+x'Hx/2.
            for col in 0..h.n {
                for k in h.colptr[col]..h.colptr[col + 1] {
                    let row = h.rowval[k];
                    if row <= col {
                        let factor = if row == col { 0.5 } else { 1.0 };
                        model_value +=
                            factor * h.nzval[k] * current_point[row] * current_point[col];
                    }
                }
            }
            let model_gap = model_value - solver.solution.obj_val_dual;
            let stationary_roundoff = optval > objval
                && optval - objval <= roundoff
                && model_gap.is_finite()
                && model_gap >= -roundoff
                && model_gap <= roundoff;
            let full_step = optval <= objval || stationary_roundoff;
            if !full_step {
                // A full Newton step can increase the true entropy objective.
                // Backtracking stays in the capped simplex, unlike accepting
                // a negative objective improvement as convergence.
                let next = solution;
                let mut step = 1.0;
                solution = current.clone();
                let mut descended = false;
                for _ in 0..50 {
                    step *= 0.5;
                    solution
                        .iter_mut()
                        .zip(&current)
                        .zip(&next)
                        .for_each(|((out, cur), next)| {
                            *out = cur + step * (next - cur);
                        });
                    optval = self.objval(hypotheses, &solution);
                    assert!(optval.is_finite(), "non-finite SQP line-search objective");
                    if optval <= objval {
                        descended = true;
                        break;
                    }
                }
                assert!(
                    descended,
                    "SQP line search could not find a nonincreasing step"
                );
            }

            // A tiny damped step alone does not certify that the QP's full
            // direction is stationary; only an undamped step may terminate.
            if full_step && objval - optval < SQP_TOLERANCE {
                let start = 1 + 2 * n_examples;
                assert!(
                    solver.solution.z[start..]
                        .iter()
                        .all(|w| *w >= -FEASIBILITY_TOLERANCE),
                    "SQP hypothesis weights violate nonnegativity"
                );
                let dual = solver.solution.z[start..]
                    .iter()
                    .map(|w| w.max(0.0))
                    .collect::<Vec<_>>();

                checkers::capped_simplex_condition(&solution[..], self.nu);
                checkers::capped_simplex_condition(&dual[..], 1f64);
                self.optval = optval;
                self.primal = solution;
                self.dual = dual;
                return;
            }

            objval = optval;
            current = solution;
        }
        panic!("row-generation SQP exceeded {MAX_SQP_ITERATIONS} iterations without convergence");
    }

    /// Returns the optimal distribution `d ∈ Δ_{m,ν}` on examples from the
    /// last successful solve, or an empty vector.
    pub fn distribution_on_examples(&self) -> Vec<f64> {
        self.primal.clone()
    }

    /// Returns the weights on hypotheses (edge-constraint multipliers, in the
    /// order of the `hypotheses` slice) from the last successful solve, or an
    /// empty vector. They are nonnegative and sum to one up to `1e-5`.
    pub fn weights_on_hypotheses(&self) -> Vec<f64> {
        self.dual.clone()
    }

    /// Lower bound from the entropy dual, when the objective supports it.
    /// Unlike the primal objective, this can be used conservatively in an
    /// outer optimality-gap stopping condition (up to floating-point error).
    /// `None` before a successful solve and for objectives solved by the
    /// generic SQP path.
    pub fn objective_lower_bound(&self) -> Option<f64> {
        self.lower_bound
    }

    /// Returns the objective value of the last successful solve, evaluated at
    /// the returned distribution, or `f64::MAX` if there is none.
    pub fn optval(&self) -> f64 {
        self.optval
    }

    /// Evaluates [`RowGenerationObjective::objective_value`] at `dist` for
    /// `hypotheses`.
    pub fn objval<H>(&self, hypotheses: &[H], dist: &[f64]) -> f64
    where
        H: Classifier,
    {
        self.objective
            .objective_value(self.sample, hypotheses, dist)
    }
}

fn require_candidate_status(status: SolverStatus) {
    assert!(
        matches!(status, SolverStatus::Solved | SolverStatus::AlmostSolved),
        "row-generation QP did not solve: {status}"
    );
}

#[cfg(test)]
mod row_generation_tests {
    use super::*;

    /// Test-only Tsallis (deformed) entropy objective, `t ∈ [0, 1)`. It exercises
    /// the generic SQP path with a non-Shannon regularizer; no booster uses it.
    struct DeformedEntropyRegularizedMaxEdge {
        t: f64,
        eta: f64,
    }

    impl DeformedEntropyRegularizedMaxEdge {
        fn new(t: f64, eta: f64) -> Self {
            assert!((0.0..1.0).contains(&t) && eta > 0.0);
            Self { t, eta }
        }
    }

    impl RowGenerationObjective for DeformedEntropyRegularizedMaxEdge {
        fn objective_value<H: Classifier>(
            &self,
            sample: &Sample,
            hypotheses: &[H],
            dist: &[f64],
        ) -> f64 {
            assert_eq!(dist.len(), sample.shape().0);
            let max_edge = hypotheses
                .iter()
                .map(|h| helpers::edge(sample, &dist, h))
                .reduce(f64::max)
                .expect("Failed to compute the max-edge");
            // sum_i d_i ln_t(d_i), with ln_t(x) = (x^{1-t} - 1) / (1-t).
            let entropy = dist
                .iter()
                .map(|&d| d * (d.powf(1.0 - self.t) - 1.0) / (1.0 - self.t))
                .sum::<f64>();
            max_edge + (entropy / self.eta)
        }

        /// outputs the gradient vector at a given point `d`:
        /// ```txt
        /// ┏                                          ┓
        /// ┃ 1 c*d₁^{1-t}  c*d₂^{1-t}  ... c*dm^{1-t} ┃
        /// ┗                                          ┛
        /// ```
        /// where `c = t * (2-t) / (η * (1-t)).`
        fn gradient(&self, dist: &[f64]) -> Vec<f64> {
            let constant = self.t * (2f64 - self.t) / (1f64 - self.t);
            iter::once(1f64)
                .chain(
                    dist.iter()
                        .map(|&d| constant * (d + PERTURBATION).powf(1f64 - self.t) / self.eta),
                )
                .collect()
        }

        /// outputs the hessian at a given point `d`:
        /// ```txt
        ///      γ   d₁     d₂    ...    dm
        ///    ┏                               ┓
        /// γ  ┃ 0   0      0     ...    0     ┃
        /// d₁ ┃ 0  1/d₁^t  0     ...    0     ┃ * (2-t)
        /// d₂ ┃ 0   0     1/d₂^t  0           ┃
        ///  . ┃ .   .      0     1/d₃^t       ┃
        ///  . ┃ .   .                         ┃
        ///  . ┃ .   .                         ┃
        /// dm ┃ 0   0     ...    ...   1/dm^t ┃
        ///    ┗                               ┛
        /// ```
        fn hessian(&self, dist: &[f64]) -> CscMatrix<f64> {
            let n_rows = dist.len() + 1;
            let n_cols = n_rows;

            let n_examples = dist.len();

            let col_ptr = iter::once(0).chain(0..=n_examples).collect();
            let row_idx = (1..=n_examples).collect();

            let constant = (2f64 - self.t) / self.eta;
            let nonzero = dist
                .iter()
                .map(|&d| constant / (d + PERTURBATION).powf(self.t))
                .collect();
            CscMatrix::new(n_rows, n_cols, col_ptr, row_idx, nonzero)
        }
    }

    #[test]
    fn deformed_entropy_converges_near_a_simplex_boundary() {
        let sample = Sample::dummy(3);
        let mut solver =
            RowGeneration::new(&sample, DeformedEntropyRegularizedMaxEdge::new(0.5, 2.0));
        solver.solve(&[Margins([1.0, 0.0, -1.0])]);
        // Minimize d0-d2 + sum(d_i^(3/2))-1 on the simplex.
        // KKT gives d0=0 and sqrt(d2)-sqrt(d1)=2/3.
        let middle = ((14.0_f64.sqrt() - 2.0) / 6.0).powi(2);
        let optimum = -(1.0 - middle) + middle.powf(1.5) + (1.0 - middle).powf(1.5) - 1.0;
        // Absolute objective tolerance covers the QP numerical gap;
        // coordinate accuracy is weaker near a constrained optimum.
        assert!((solver.optval() - optimum).abs() < 1e-7);
        for (actual, expected) in solver.primal.iter().zip([0.0, middle, 1.0 - middle]) {
            assert!((actual - expected).abs() < 1e-5);
        }
        assert_eq!(solver.objective_lower_bound(), None);
    }

    struct Margins([f64; 3]);
    impl Classifier for Margins {
        fn confidence(&self, sample: &Sample, row: usize) -> f64 {
            self.0[row] * sample.target()[row]
        }
    }

    #[test]
    fn reported_455_example_high_eta_problem_has_a_verified_entropy_gap() {
        struct Column(Vec<f64>);
        impl Classifier for Column {
            fn confidence(&self, sample: &Sample, row: usize) -> f64 {
                self.0[row] * sample.target()[row]
            }
        }
        // Histogram of the first entropy stump on the reported breast-cancer
        // split. No external dataset is needed to reproduce its subproblem.
        let values: Vec<_> = [
            (0.7853107344632764, 158),
            (0.8992805755395681, 264),
            (-0.7853107344632764, 19),
            (-0.8992805755395681, 14),
        ]
        .into_iter()
        .flat_map(|(v, n)| std::iter::repeat_n(v, n))
        .collect();
        let sample = Sample::dummy(455);
        let eta = 100.0_f64.ln() / 0.0005;
        let mut solver = RowGeneration::new(&sample, EntropyRegularizedMaxEdge::new(eta));
        solver.initialize(4.55, EntropyRegularizedMaxEdge::new(eta));
        solver.solve(&[Column(values)]);
        // Other margin groups have <exp(-1000) relative mass. The cap is
        // inactive on the 14 equal minimal margins (1/14 < 1/4.55).
        let exact = -0.8992805755395681 + (455.0_f64 / 14.0).ln() / eta;
        let lower = solver.objective_lower_bound().unwrap();
        assert!(lower <= exact + 1e-12 && exact - lower < 1e-9);
        assert!(solver.optval() - lower < 1e-7);
        assert!(
            solver
                .primal
                .iter()
                .all(|d| d.is_finite() && *d >= 0.0 && *d <= 1.0 / 4.55)
        );
        assert!((solver.primal.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(solver.dual, vec![1.0]);
    }

    #[test]
    fn entropy_solution_matches_gibbs_distribution_and_capping() {
        let sample = Sample::dummy(3);
        let hypotheses = [Margins([1.0, 0.0, -1.0])];
        for eta in [0.5, 2.0, 10.0, 100.0] {
            let mut solver = RowGeneration::new(&sample, EntropyRegularizedMaxEdge::new(eta));
            solver.solve(&hypotheses);
            let expected = helpers::softmax(&[-eta, 0.0, eta]);
            let optimum = solver.objval(&hypotheses, &expected);
            // Check absolute distribution accuracy; the tiny tail mass
            // cannot be expected to have accurate relative error at eta=100.
            assert!((solver.optval() - optimum).abs() < 1e-7);
            for (actual, expected) in solver.primal.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-6);
            }
        }
        let mut solver = RowGeneration::new(&sample, EntropyRegularizedMaxEdge::new(2.0));
        solver.initialize(2.0, EntropyRegularizedMaxEdge::new(2.0));
        solver.solve(&hypotheses);
        // KKT: the last row saturates its cap; the first two retain the
        // Gibbs ratio exp(-2):1 and together have mass 1/2.
        let first = 0.5 / (1.0 + 2.0_f64.exp());
        for (actual, expected) in solver.primal.iter().zip([first, 0.5 - first, 0.5]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn initialize_and_failed_solve_discard_previous_solution() {
        let sample = Sample::dummy(3);
        let mut solver = RowGeneration::new(&sample, EntropyRegularizedMaxEdge::new(2.0));
        solver.solve(&[Margins([1.0, 0.0, -1.0])]);
        solver.initialize(1.0, EntropyRegularizedMaxEdge::new(2.0));
        assert!(solver.primal.is_empty() && solver.dual.is_empty());
        assert_eq!(solver.objective_lower_bound(), None);
        assert_eq!(solver.optval, f64::MAX);
        solver.solve(&[Margins([1.0, 0.0, -1.0])]);
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            solver.solve(&[Margins([f64::NAN, 0.0, -1.0])]);
        }));
        assert!(failed.is_err());
        assert!(solver.primal.is_empty() && solver.dual.is_empty());
        assert_eq!(solver.objective_lower_bound(), None);
        assert_eq!(solver.optval, f64::MAX);
        solver.solve(&[Margins([1.0, 0.0, -1.0])]);
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            solver.solve::<Margins>(&[]);
        }));
        assert!(failed.is_err());
        assert!(solver.primal.is_empty() && solver.dual.is_empty());
        assert_eq!(solver.objective_lower_bound(), None);
        assert_eq!(solver.optval, f64::MAX);
    }

    #[test]
    fn symmetric_max_edge_kink_has_matching_primal_and_dual() {
        let sample = Sample::dummy(3);
        let mut solver = RowGeneration::new(&sample, EntropyRegularizedMaxEdge::new(2.0));
        let hypotheses = [Margins([1.0, 0.0, -1.0]), Margins([-1.0, 0.0, 1.0])];
        solver.solve(&hypotheses);
        // Pinsker/strong convexity: KL/eta bounds the squared l1 error.
        // An objective-accurate conic solution need not have 1e-7 coordinate
        // accuracy on a flat entropy objective. Check the certified bound.
        let gap = solver.optval - solver.objective_lower_bound().unwrap();
        let error: f64 = solver.primal.iter().map(|d| (d - 1.0 / 3.0).abs()).sum();
        assert!(gap >= -1e-12 && gap < 1e-8);
        assert!(error <= (4.0 * gap.max(0.0)).sqrt() + 1e-10);
        assert!(solver.dual.iter().all(|w| (w - 0.5).abs() < 1e-7));
        assert!(solver.optval.abs() < 1e-8);
        // With uniform d and equal atom weights, every example has zero
        // ensemble margin and hence the entropy KKT gradient is constant.
        let margins: Vec<_> = (0..3)
            .map(|i| solver.dual[0] * hypotheses[0].0[i] + solver.dual[1] * hypotheses[1].0[i])
            .collect();
        assert!(margins.iter().all(|m| m.abs() < 1e-7));
    }

    #[test]
    fn incomplete_solver_status_is_not_convergence() {
        require_candidate_status(SolverStatus::Solved);
        require_candidate_status(SolverStatus::AlmostSolved);
        for status in [
            SolverStatus::InsufficientProgress,
            SolverStatus::MaxIterations,
            SolverStatus::NumericalError,
            SolverStatus::PrimalInfeasible,
        ] {
            assert!(std::panic::catch_unwind(|| require_candidate_status(status)).is_err());
        }
    }
}

/// The ERLPBoost subproblem objective
/// ```txt
/// max_j Σ_i d_i y_i h_j(x_i) + KL(d ‖ u)/η,   KL(d ‖ u) = Σ_i d_i ln(m d_i).
/// ```
///
/// [`RowGeneration`] solves it with the exact exponential-cone formulation
/// (its [`entropy_parameter`](RowGenerationObjective::entropy_parameter) is
/// `Some(η)`), so the [`gradient`](RowGenerationObjective::gradient) and
/// [`hessian`](RowGenerationObjective::hessian) below are not used by
/// [`RowGeneration`]. They are a perturbed model: `ln(d_i + 1e-10)/η` and
/// `1/((d_i + 1e-10) η)`, omitting the additive constants `(1 + ln m)/η` of
/// the exact gradient (a constant shift on the simplex).
///
/// Its `objective_value` panics if `hypotheses` is empty, if `dist` does not
/// have one entry per example, or if the entropy term is not finite.
pub struct EntropyRegularizedMaxEdge(f64);

impl EntropyRegularizedMaxEdge {
    /// Takes the regularization parameter `η` and returns `Self`.
    ///
    /// # Panics
    /// Panics unless `η` is finite and positive.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{EntropyRegularizedMaxEdge, RowGenerationObjective};
    ///
    /// let objective = EntropyRegularizedMaxEdge::new(10.0);
    /// // Selects the exact exponential-cone path of `RowGeneration`.
    /// assert_eq!(objective.entropy_parameter(), Some(10.0));
    /// ```
    pub fn new(eta: f64) -> Self {
        assert!(
            eta.is_finite() && eta > 0.0,
            "eta must be finite and positive"
        );
        Self(eta)
    }
}

impl RowGenerationObjective for EntropyRegularizedMaxEdge {
    fn entropy_parameter(&self) -> Option<f64> {
        Some(self.0)
    }

    fn objective_value<H: Classifier>(
        &self,
        sample: &Sample,
        hypotheses: &[H],
        dist: &[f64],
    ) -> f64 {
        assert_eq!(dist.len(), sample.shape().0);

        let eta = self.0;
        let max_edge = hypotheses
            .iter()
            .map(|h| helpers::edge(sample, dist, h))
            .reduce(f64::max)
            .expect("failed to compute the max-edge.");
        let entropy = helpers::entropy_from_uni_distribution(dist);
        assert!(entropy.is_finite(), "not a finite entropy. dist = {dist:?}",);
        max_edge + (entropy / eta)
    }

    /// outputs the gradient vector at a given point `d`:
    /// ```txt
    /// ┏                                    ┓
    /// ┃ 1 ln(d₁)/η  ln(d₂)/η  ... ln(dm)/η ┃
    /// ┗                                    ┛
    /// ```
    fn gradient(&self, dist: &[f64]) -> Vec<f64> {
        let eta = self.0;
        iter::once(1f64)
            .chain(dist.iter().map(|&d| (d + PERTURBATION).ln() / eta))
            .collect()
    }

    /// outputs the hessian at a given point `d`:
    /// ```txt
    ///      γ   d₁   d₂  ...  dm
    ///    ┏                       ┓
    /// γ  ┃ 0   0    0   ...  0   ┃
    /// d₁ ┃ 0  1/d₁  0   ...  0   ┃ * (1/η)
    /// d₂ ┃ 0   0   1/d₂  0       ┃
    ///  . ┃ .   .    0   1/d₃     ┃
    ///  . ┃ .   .                 ┃
    ///  . ┃ .   .                 ┃
    /// dm ┃ 0   0   ...  ... 1/dm ┃
    ///    ┗                       ┛
    /// ```
    fn hessian(&self, dist: &[f64]) -> CscMatrix<f64> {
        let eta = self.0;
        let n_rows = dist.len() + 1;
        let n_cols = n_rows;

        let n_examples = dist.len();

        let col_ptr = iter::once(0).chain(0..=n_examples).collect();
        let row_idx = (1..=n_examples).collect();
        let nonzero = dist
            .iter()
            .map(|&d| 1f64 / ((d + PERTURBATION) * eta))
            .collect();
        CscMatrix::new(n_rows, n_cols, col_ptr, row_idx, nonzero)
    }
}
