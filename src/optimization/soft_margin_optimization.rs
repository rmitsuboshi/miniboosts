//! The soft-margin linear program of LPBoost.
//!
//! For `m` examples, hypotheses `h_1, ..., h_T`, and capping parameter
//! `ν ∈ [1, m]`, the primal problem is
//! ```txt
//! max_{ρ, w, ξ}  ρ - (1/ν) Σ_i ξ_i
//! s.t.           y_i Σ_j w_j h_j(x_i) ≥ ρ - ξ_i,   i = 1, ..., m,
//!                Σ_j w_j = 1,  w ≥ 0,  ξ ≥ 0,
//! ```
//! and its LP dual is the edge minimization over the capped simplex
//! ```txt
//! min_{γ, d}  γ
//! s.t.        Σ_i d_i y_i h_j(x_i) ≤ γ,   j = 1, ..., T,
//!             d ∈ Δ_{m,ν} = { d : Σ_i d_i = 1, 0 ≤ d_i ≤ 1/ν }.
//! ```
//! Both have the same optimal value. The dual variables `d` of the margin
//! constraints form the distribution on examples used by LPBoost.
//!
//! [`soft_margin_optimization`] solves the primal once for a fixed hypothesis
//! set; [`ColumnGeneration`] adds one hypothesis (column) per call and
//! re-solves. See Demiriz, Bennett and Shawe-Taylor (2002), "Linear
//! programming boosting via column generation", and Warmuth, Glocer and
//! Rätsch (2008), "Boosting algorithms for maximizing the soft margin".
use crate::core::{
    Classifier, Sample,
    tools::{checkers, helpers},
};

use clarabel::{algebra::*, solver::*};

use std::iter;

/// Solves the soft-margin LP for a fixed set of hypotheses and returns
/// `(optimal value, weights on hypotheses)`.
///
/// ```txt
/// max ρ - (1/ν) Σ_i ξ_i
/// s.t. y_i Σ_j w_j h_j (x_i) ≥ ρ - ξ_i,   ∀i = 1, 2, ..., m
///      Σ_j w_j = 1,
///      w_1, w_2, ..., w_T ≥ 0,
///      ξ_1, ξ_2, ..., ξ_m ≥ 0.
/// ```
/// The returned value is `ρ - (1/ν) Σ_i ξ_i` at the solution (the negated
/// Clarabel objective). The returned weights are the solver's `w`, with
/// entries in `[-1e-7, 0)` clipped to zero and then renormalized to sum to
/// one; their order matches `hypotheses`. Boosters use this to compute final
/// weights for a soft-margin ensemble.
///
/// Time complexity: building the matrix evaluates every hypothesis on every
/// example (`O(mT)` confidence calls); the LP has `1 + m + T` variables and
/// is solved from scratch by an interior-point method.
///
/// # Panics
/// - if `ν ∉ [1, m]`;
/// - if `hypotheses` is empty (the constraint `Σ_j w_j = 1` is infeasible);
/// - unless Clarabel returns `Solved` or `AlmostSolved` **and** the solution
///   and diagnostics are finite, the primal and dual residuals are at most
///   `1e-7`, and the absolute primal-dual objective gap is at most `1e-7`;
/// - if a raw weight is below `-1e-7` or the raw weights' sum differs from
///   one by more than `1e-7`.
///
/// `AlmostSolved` is accepted only together with these explicit checks. No
/// error value is returned.
///
/// # Formulation
/// To solve the problem we build the constraint matrix
/// ```txt
/// # of   
/// rows    ρ   ξ1 ξ2  ... ξm       w1        ...    wT
///       ┏   ┃               ┃                                 ┓   ┏   ┓
///       ┃ 1 ┃ -1  0  ...  0 ┃ -y_1 h_1(x_1) ... -y_1 h_T(x_1) ┃ ≤ ┃ 0 ┃
///       ┃ 1 ┃  0 -1  ...  . ┃ -y_2 h_1(x_2) ... -y_2 h_T(x_2) ┃ ≤ ┃ 0 ┃
///   m   ┃ . ┃  .  0  .    . ┃      .        ...      .        ┃ . ┃ . ┃
///       ┃ . ┃  .  .   .   . ┃      .        ...      .        ┃ . ┃ . ┃
///       ┃ . ┃  .  .    .  . ┃      .        ...      .        ┃ . ┃ . ┃
///       ┃ 1 ┃  0  0  ... -1 ┃ -y_m h_1(x_m) ... -y_m h_T(x_m) ┃ ≤ ┃ 0 ┃
///      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
///   1   ┃ 0 ┃  0  0  ...  0 ┃      1        ...      1        ┃ = ┃ 1 ┃
///      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
///       ┃ 0 ┃ -1  0  ...  0 ┃                                 ┃ ≤ ┃ 0 ┃
///       ┃ 0 ┃  0 -1  ...  . ┃                                 ┃ ≤ ┃ 0 ┃
///       ┃ . ┃  .  0  .    . ┃                                 ┃ . ┃ . ┃
///   m   ┃ . ┃  .  .   .   . ┃                O                ┃ . ┃ . ┃
///       ┃ . ┃  .  .    .  . ┃                                 ┃ . ┃ . ┃
///       ┃ 0 ┃  0  0  ... -1 ┃                                 ┃ ≤ ┃ 0 ┃
///      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
///       ┃ 0 ┃               ┃     -1    0   ...     0         ┃ ≤ ┃ 0 ┃
///       ┃ 0 ┃               ┃      0   -1   ...     0         ┃ ≤ ┃ 0 ┃
///       ┃ . ┃               ┃      .        ...     .         ┃ . ┃ . ┃
///   H   ┃ . ┃       O       ┃      .        ...     .         ┃ . ┃ . ┃
///       ┃ . ┃               ┃      .        ...     .         ┃ . ┃ . ┃
///       ┃ 0 ┃               ┃      0    0   ...    -1         ┃ ≤ ┃ 0 ┃
///       ┗   ┃               ┃                                 ┛   ┗   ┛
///
/// # of
/// cols    1 ┃       m       ┃               H
/// ```
/// where the first `m` rows are the margin constraints, the next row is the
/// equality `Σ_j w_j = 1`, and the remaining `m + T` rows encode `ξ ≥ 0` and
/// `w ≥ 0`. The `T` hypothesis columns are shown as `H` above.
///
/// Since the `clarabel` crate solves minimization problems, the objective is
/// negated: Clarabel minimizes `-ρ + (1/ν) Σ_i ξ_i`.
///
/// # Examples
/// Three hypotheses on four examples, each wrong on a different example.
/// With weights `w`, the margins of examples 2, 3, 4 sum to `Σ_j w_j = 1`,
/// so the smallest margin is at most `1/3`, attained only by
/// `w = (1/3, 1/3, 1/3)`. With `ν = 1` the objective is the smallest margin.
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
/// let (value, weights) = miniboosts::soft_margin_optimization(1.0, &sample, &hypotheses);
///
/// // The interior-point solution is checked to 1e-7 residuals and gap.
/// assert!((value - 1.0 / 3.0).abs() < 1e-6);
/// assert_eq!(weights.len(), hypotheses.len());
/// assert!(weights.iter().all(|w| (w - 1.0 / 3.0).abs() < 1e-6));
/// // The returned weights are renormalized, so they sum to one up to rounding.
/// assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
/// ```
pub fn soft_margin_optimization<T, H>(
    // capping parameter
    nu: f64,
    // training examples
    sample: &Sample,
    // hypotheses
    hypotheses: T,
) -> (f64, Vec<f64>)
where
    T: AsRef<[H]>,
    H: Classifier,
{
    let hypotheses = hypotheses.as_ref();
    let n_hypotheses = hypotheses.len();
    let n_examples = sample.shape().0;

    checkers::capping_parameter(nu, n_examples);

    let obj = build_objective(nu, n_examples, n_hypotheses);

    let mat = build_constraint_matrix(sample, hypotheses);
    let sns = build_sns(n_examples, n_hypotheses);
    let rhs = build_rhs(n_examples, n_hypotheses);

    let mut solver = build_solver(obj, mat, sns, rhs);
    solver.solve();

    // AlmostSolved is a numerical status, not an optimality certificate.
    // Apply the same explicit residual/gap checks to both successful statuses.
    let solution = &solver.solution;
    let tolerance = 1e-7;
    assert!(
        matches!(
            solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ) && solution.x.iter().chain(&solution.z).all(|v| v.is_finite())
            && solution.obj_val.is_finite()
            && solution.obj_val_dual.is_finite()
            && solution.r_prim.is_finite()
            && solution.r_dual.is_finite()
            && solution.r_prim <= tolerance
            && solution.r_dual <= tolerance
            && (solution.obj_val - solution.obj_val_dual).abs() <= tolerance,
        "soft-margin LP failed numerical checks: status={}, primal residual={}, dual residual={}, gap={}",
        solution.status,
        solution.r_prim,
        solution.r_dual,
        (solution.obj_val - solution.obj_val_dual).abs(),
    );

    let objval = -solution.obj_val;
    let start = 1 + n_examples;
    let raw_weights = &solution.x[start..];
    assert!(raw_weights.iter().all(|w| *w >= -tolerance));
    assert!((raw_weights.iter().sum::<f64>() - 1.0).abs() <= tolerance);
    let mut weight: Vec<_> = raw_weights.iter().map(|w| w.max(0.0)).collect();
    let sum: f64 = weight.iter().sum();
    assert!(sum.is_finite() && sum > 0.0);
    weight.iter_mut().for_each(|w| *w /= sum);

    (objval, weight)
}

fn build_objective(nu: f64, n_examples: usize, n_hypotheses: usize) -> Vec<f64> {
    iter::once(-1f64)
        .chain(iter::repeat_n(1f64 / nu, n_examples))
        .chain(iter::repeat_n(0f64, n_hypotheses))
        .collect()
}

fn build_constraint_matrix<H>(sample: &Sample, hypotheses: &[H]) -> CscMatrix<f64>
where
    H: Classifier,
{
    let n_hypotheses = hypotheses.len();
    let n_examples = sample.shape().0;

    let mut col_ptr = vec![0];
    let mut row_idx = (0..n_examples).collect::<Vec<_>>();
    let mut nonzero = vec![1f64; n_examples];

    // columns for `ξ.`
    for i in 0..n_examples {
        // insert a new column index
        col_ptr.push(row_idx.len());
        // insert a coefficient `-1` for `ξ[i]` of constraint
        // `ρ - ξ[i] ≤ y[i] Σ_{h} w[h] h(x[i])`
        row_idx.push(i);
        nonzero.push(-1f64);
        // insert a coefficient `-1` for `ξ[i]` of constraint
        // `-ξ[i] ≤ 0` (i.e., `ξ[i] ≥ 0.`)
        row_idx.push(n_examples + 1 + i);
        nonzero.push(-1f64);
    }

    // columns for `w.`
    for (j, h) in hypotheses.iter().enumerate() {
        // insert a new column index
        col_ptr.push(row_idx.len());

        // insert a coefficient for `w[h]` of constraint
        // `ρ - ξ[i] - y[i] Σ_{h} w[h] h(x[i]) ≤ 0`
        // (i.e., `ρ - ξ[i] ≤ y[i] Σ_{h} w[h] h(x[i])`)
        let iter = helpers::margins(sample, h).enumerate();
        for (i, yhx) in iter {
            row_idx.push(i);
            nonzero.push(-yhx);
        }

        // insert a coefficient `1` for `w[h]` of constraint
        // `Σ_h w[h] = 1.`
        row_idx.push(n_examples);
        nonzero.push(1f64);

        // insert a coefficient `-1` for `w[h]` of constraint
        // `-w[h] ≤ 0` (i.e., `w[h] ≥ 0.`)
        let offset = 2 * n_examples + 1;
        row_idx.push(offset + j);
        nonzero.push(-1f64);
    }
    col_ptr.push(row_idx.len());

    let n_rows = 2 * n_examples + 1 + n_hypotheses;
    let n_cols = 1 + n_examples + n_hypotheses;
    CscMatrix::new(n_rows, n_cols, col_ptr, row_idx, nonzero)
}

fn build_sns(n_examples: usize, n_hypotheses: usize) -> Vec<SupportedConeT<f64>> {
    vec![
        NonnegativeConeT(n_examples),
        ZeroConeT(1),
        NonnegativeConeT(n_examples),
        NonnegativeConeT(n_hypotheses),
    ]
}

fn build_rhs(n_examples: usize, n_hypotheses: usize) -> Vec<f64> {
    let mut rhs = vec![0f64; 2 * n_examples + 1 + n_hypotheses];
    rhs[n_examples] = 1f64;
    rhs
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

/// Column generation for the soft-margin LP (the LPBoost subproblem).
///
/// Each call to [`solve`](Self::solve) appends one hypothesis column to the
/// restricted problem
/// ```txt
/// max_{ρ, w, ξ}  ρ - (1/ν) Σ_i ξ_i
/// s.t.           y_i Σ_j w_j h_j(x_i) ≥ ρ - ξ_i,   Σ_j w_j = 1,  w, ξ ≥ 0,
/// ```
/// over all hypotheses added since the last [`initialize`](Self::initialize),
/// and re-solves it from scratch with Clarabel (no warm start). After a
/// solve:
/// - [`optimal_value`](Self::optimal_value) is `ρ - (1/ν) Σ_i ξ_i`, which by
///   LP duality equals the minimum over `d ∈ Δ_{m,ν}` of the maximum edge;
/// - [`weights_on_hypotheses`](Self::weights_on_hypotheses) is the primal `w`;
/// - [`distribution_on_examples`](Self::distribution_on_examples) is the dual
///   multiplier vector `d` of the `m` margin constraints, a distribution in
///   `Δ_{m,ν}`.
///
/// Usage: [`new`](Self::new), then [`initialize`](Self::initialize), then
/// [`solve`](Self::solve) once per new hypothesis. Hypotheses are not
/// deduplicated; adding the same hypothesis twice adds a duplicate column.
///
/// # Solver failure
/// [`solve`](Self::solve) accepts Clarabel status `Solved` or `AlmostSolved`.
/// Unlike [`soft_margin_optimization`], it does **not** check residuals or the
/// duality gap explicitly; it only checks (by panicking) that the clipped
/// primal weights and the dual distribution lie in their capped simplices up
/// to an absolute sum tolerance of `1e-5` and an upper-bound tolerance of
/// `1e-5`. Any other status panics. No error value is returned.
///
/// # Examples
/// Add the hypotheses one by one with `ν = 1` and check each restricted
/// optimum by hand:
/// - `{h_1}`: the smallest margin is `-1` (example 4), and the optimal
///   distribution puts all mass on example 4;
/// - `{h_1, h_2}`: margins of examples 3 and 4 are `±(w_1 - w_2)`, so the
///   value is `0` at `w = (1/2, 1/2)` with `d = (0, 0, 1/2, 1/2)`;
/// - `{h_1, h_2, h_3}`: the value is `1/3` at `w = (1/3, 1/3, 1/3)` (see
///   [`soft_margin_optimization`]) with `d = (0, 1/3, 1/3, 1/3)`.
///
/// Each optimum is unique, so the solver's output is compared directly. The
/// observed errors are about `1e-8`; `1e-6` leaves room for the
/// interior-point method's accuracy.
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
/// use miniboosts::ColumnGeneration;
///
/// let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6);
///
/// let mut solver = ColumnGeneration::new(&sample);
/// solver.initialize(4, 1.0);
///
/// solver.solve(&hypotheses[0]);
/// assert!((solver.optimal_value() - (-1.0)).abs() < 1e-6);
/// assert!(close(&solver.weights_on_hypotheses(), &[1.0]));
/// assert!(close(&solver.distribution_on_examples(), &[0.0, 0.0, 0.0, 1.0]));
///
/// solver.solve(&hypotheses[1]);
/// assert!(solver.optimal_value().abs() < 1e-6);
/// assert!(close(&solver.weights_on_hypotheses(), &[0.5, 0.5]));
/// assert!(close(&solver.distribution_on_examples(), &[0.0, 0.0, 0.5, 0.5]));
///
/// solver.solve(&hypotheses[2]);
/// let third = 1.0 / 3.0;
/// assert!((solver.optimal_value() - third).abs() < 1e-6);
/// assert!(close(&solver.weights_on_hypotheses(), &[third; 3]));
/// assert!(close(&solver.distribution_on_examples(), &[0.0, third, third, third]));
/// ```
pub struct ColumnGeneration<'a> {
    nu: f64,
    sample: &'a Sample,

    n_hypotheses: usize,

    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    nonzero: Vec<f64>,

    initialized: bool,

    optval: f64,
    primal: Vec<f64>,
    dual: Vec<f64>,
}

impl<'a> ColumnGeneration<'a> {
    /// Creates an uninitialized solver for `sample`.
    /// Call [`initialize`](Self::initialize) before [`solve`](Self::solve).
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{ColumnGeneration, Sample};
    ///
    /// let sample = Sample::dummy(4);
    /// let solver = ColumnGeneration::new(&sample);
    /// // No solution yet.
    /// assert_eq!(solver.optimal_value(), f64::MAX);
    /// assert!(solver.weights_on_hypotheses().is_empty());
    /// ```
    pub fn new(sample: &'a Sample) -> Self {
        Self {
            nu: 0f64,
            sample,
            n_hypotheses: 0,

            col_ptr: Vec::new(),
            row_idx: Vec::new(),
            nonzero: Vec::new(),

            initialized: false,

            optval: f64::MAX,
            primal: Vec::new(),
            dual: Vec::new(),
        }
    }

    /// Sets the capping parameter `ν` and discards all columns and
    /// solutions. Afterwards [`optimal_value`](Self::optimal_value) is
    /// `f64::MAX` and both solution vectors are empty.
    ///
    /// # Panics
    /// Panics if `n_examples` differs from the number of rows of the sample,
    /// or if `ν ∉ [1, n_examples]`.
    ///
    /// # Examples
    /// Re-initializing discards the columns, so the next solve sees only the
    /// new hypothesis:
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
    /// use miniboosts::ColumnGeneration;
    ///
    /// let mut solver = ColumnGeneration::new(&sample);
    /// solver.initialize(sample.shape().0, 1.0);
    /// solver.solve(&hypotheses[0]);
    /// solver.solve(&hypotheses[1]);
    /// assert_eq!(solver.weights_on_hypotheses().len(), 2);
    ///
    /// // Start over with ν = 2.
    /// solver.initialize(sample.shape().0, 2.0);
    /// assert_eq!(solver.optimal_value(), f64::MAX);
    /// assert!(solver.distribution_on_examples().is_empty());
    /// solver.solve(&hypotheses[0]);
    /// // One column: the value is the mean of the 2 smallest margins of h_1,
    /// // (1 + (-1)) / 2 = 0.
    /// assert_eq!(solver.weights_on_hypotheses().len(), 1);
    /// assert!(solver.optimal_value().abs() < 1e-6);
    /// // Every weight of d is capped at 1/ν = 1/2 (up to the 1e-5 check).
    /// let d = solver.distribution_on_examples();
    /// assert!(d.iter().all(|&di| di <= 0.5 + 1e-5));
    /// ```
    pub fn initialize(&mut self, n_examples: usize, nu: f64) {
        assert_eq!(n_examples, self.sample.shape().0);
        checkers::capping_parameter(nu, n_examples);
        self.nu = nu;
        self.n_hypotheses = 0;
        self.optval = f64::MAX;
        self.primal.clear();
        self.dual.clear();

        self.col_ptr = vec![0];
        self.row_idx = (0..n_examples).collect();
        self.nonzero = vec![1f64; n_examples];

        for i in 0..n_examples {
            self.col_ptr.push(self.row_idx.len());

            self.row_idx.push(i);
            self.nonzero.push(-1f64);

            self.row_idx.push(n_examples + 1 + i);
            self.nonzero.push(-1f64);
        }

        self.initialized = true;
    }

    fn n_hypotheses(&self) -> usize {
        self.n_hypotheses
    }

    /// Adds hypothesis `h` as a new column and re-solves the restricted LP.
    ///
    /// Time complexity: `O(m)` confidence evaluations for the new column,
    /// plus an interior-point solve with `1 + m + T` variables, where `T` is
    /// the number of columns added so far.
    ///
    /// # Panics
    /// Panics if [`initialize`](Self::initialize) has not been called, if
    /// Clarabel returns a status other than `Solved` or `AlmostSolved`, or if
    /// the solution fails the capped-simplex checks described on
    /// [`ColumnGeneration`]. After a panic the solver state is unspecified;
    /// call [`initialize`](Self::initialize) before reuse.
    ///
    /// # Examples
    /// Weights are reported in insertion order, one per call; see
    /// [`ColumnGeneration`] for the values.
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
    /// use miniboosts::ColumnGeneration;
    ///
    /// let mut solver = ColumnGeneration::new(&sample);
    /// solver.initialize(4, 1.0);
    /// for (t, h) in hypotheses.iter().enumerate() {
    ///     solver.solve(h);
    ///     assert_eq!(solver.weights_on_hypotheses().len(), t + 1);
    ///     assert_eq!(solver.distribution_on_examples().len(), 4);
    /// }
    /// ```
    pub fn solve<H>(&mut self, h: &H)
    where
        H: Classifier,
    {
        if !self.initialized {
            panic!(
                "solver is uninitialized.\
                call `.initialize()` method before `.solve().`"
            );
        }

        self.n_hypotheses += 1;
        let n_examples = self.sample.shape().0;

        let obj = build_objective(self.nu, n_examples, self.n_hypotheses);

        let mat = self.append_matrix(h);
        let sns = build_sns(n_examples, self.n_hypotheses);
        let rhs = build_rhs(n_examples, self.n_hypotheses);

        let mut solver = build_solver(obj, mat, sns, rhs);
        solver.solve();

        assert!(
            matches! {
                solver.solution.status,
                SolverStatus::Solved | SolverStatus::AlmostSolved
            },
            "unexped solver status. got {}",
            solver.solution.status,
        );

        self.optval = -solver.solution.obj_val;

        let start = 1 + n_examples;
        self.primal = solver.solution.x[start..]
            .iter()
            .map(|w| w.max(0f64))
            .collect::<Vec<f64>>();
        checkers::capped_simplex_condition(&self.primal[..], 1f64);

        self.dual = solver.solution.z[..n_examples]
            .iter()
            .map(|d| d.abs())
            .collect::<Vec<f64>>();
        checkers::capped_simplex_condition(&self.dual[..], self.nu);
    }

    fn append_matrix<H>(&mut self, h: &H) -> CscMatrix<f64>
    where
        H: Classifier,
    {
        let n_examples = self.sample.shape().0;
        let n_hypotheses = self.n_hypotheses();

        self.col_ptr.push(self.row_idx.len());
        for (i, yh) in helpers::margins(self.sample, h).enumerate() {
            self.row_idx.push(i);
            self.nonzero.push(-yh);
        }

        self.row_idx.push(n_examples);
        self.nonzero.push(1f64);

        self.row_idx.push(2 * n_examples + n_hypotheses);
        self.nonzero.push(-1f64);

        let n_rows = 1 + 2 * n_examples + n_hypotheses;
        let n_cols = 1 + n_examples + n_hypotheses;

        let col_ptr = {
            let mut col_ptr = self.col_ptr.clone();
            col_ptr.push(self.row_idx.len());
            col_ptr
        };
        let row_idx = self.row_idx.clone();
        let nonzero = self.nonzero.clone();

        CscMatrix::new(n_rows, n_cols, col_ptr, row_idx, nonzero)
    }

    /// Returns the optimal value `ρ - (1/ν) Σ_i ξ_i` of the last solve, or
    /// `f64::MAX` before the first solve.
    pub fn optimal_value(&self) -> f64 {
        self.optval
    }

    /// Returns the weights `w` on the hypotheses, in insertion order, from the
    /// last solve. Negative solver roundoff is clipped to zero, but the
    /// vector is not renormalized. Empty before the first solve.
    pub fn weights_on_hypotheses(&self) -> Vec<f64> {
        self.primal.clone()
    }

    /// Returns the distribution `d ∈ Δ_{m,ν}` on examples (absolute values of
    /// the margin-constraint dual multipliers) from the last solve. Empty
    /// before the first solve.
    pub fn distribution_on_examples(&self) -> Vec<f64> {
        self.dual.clone()
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    struct Margins([f64; 3]);
    impl Classifier for Margins {
        fn confidence(&self, sample: &Sample, row: usize) -> f64 {
            self.0[row] * sample.target()[row]
        }
    }

    #[test]
    fn final_soft_margin_lp_returns_the_known_simplex_optimum() {
        let sample = Sample::dummy(3);
        let (value, weights) = soft_margin_optimization(
            1.0,
            &sample,
            [Margins([1.0, 0.0, -1.0]), Margins([-1.0, 0.0, 1.0])],
        );
        assert!(value.abs() < 1e-7);
        assert!(
            weights
                .iter()
                .all(|w| w.is_finite() && (*w - 0.5).abs() < 1e-7)
        );
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn initialize_discards_previous_columns_and_solutions() {
        let sample = Sample::dummy(3);
        let mut solver = ColumnGeneration::new(&sample);
        for _ in 0..2 {
            solver.initialize(3, 1.0);
            assert_eq!(solver.n_hypotheses, 0);
            assert!(solver.weights_on_hypotheses().is_empty());
            assert!(solver.distribution_on_examples().is_empty());
            assert_eq!(solver.optimal_value(), f64::MAX);
            solver.solve(&Margins([1.0, 0.0, -1.0]));
            solver.solve(&Margins([-1.0, 0.0, 1.0]));
            assert!(solver.optimal_value().abs() < 1e-7);
            assert_eq!(solver.weights_on_hypotheses().len(), 2);
            assert!(
                solver
                    .weights_on_hypotheses()
                    .iter()
                    .all(|w| (*w - 0.5).abs() < 1e-7)
            );
        }
    }
}
