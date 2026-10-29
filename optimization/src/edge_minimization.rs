use miniboosts_core::{
    Classifier, Sample,
    constants::{DEFAULT_CAPPING, PERTURBATION, SQP_TOLERANCE},
    tools::checkers,
    tools::helpers,
};

use clarabel::{algebra::*, solver::*};

use std::iter;

/// A linear programming model for edge minimization.
/// `LPModel` solves the entropy regularized edge minimization problem:
///
/// ```txt
/// min γ
/// γ,d
/// s.t. Σ_i d_i y_i h_j (x_i) ≤ γ,   ∀j = 1, 2, ..., t
///      Σ_i d_i = 1,
///      d_1, d_2, ..., d_m ≤ 1/ν
///      d_1, d_2, ..., d_m ≥ 0.
/// ```
/// `LPModel` solves this approximated problem until convergent.
///
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

/// `RowGeneration` is a struct for row-generation algorithm
/// over a capped simplex `Δ_{m,ν}` for a specified objective function.
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

pub trait RowGenerationObjective {
    /// Opt into the exact Shannon-entropy cone formulation. Return eta only
    /// for max-edge + KL(d || uniform)/eta; other objectives retain SQP.
    fn entropy_parameter(&self) -> Option<f64> {
        None
    }

    fn objective_value<H: Classifier>(
        &self,
        sample: &Sample,
        hypotheses: &[H],
        dist: &[f64],
    ) -> f64;
    fn gradient(&self, dist: &[f64]) -> Vec<f64>;
    fn hessian(&self, dist: &[f64]) -> CscMatrix<f64>;
}

impl<T> RowGeneration<'_, T>
where
    T: RowGenerationObjective,
{
    /// Solves the restricted Shannon-entropy subproblem with exponential cones.
    /// Other objectives retain the safeguarded SQP implementation.
    ///
    /// Panics on non-finite inputs, uncertified QP solver failure, or exhausted
    /// iteration/line-search limits. Such failures leave no published solution.
    /// The entropy path checks the original primal/dual gap and exposes its
    /// lower bound. Generic SQP only checks its quadratic-model diagnostics.
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
            let result =
                crate::entropy_solver::solve(eta, self.nu, mat, rhs, sns, n_examples, n_hypotheses);
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

    pub fn distribution_on_examples(&self) -> Vec<f64> {
        self.primal.clone()
    }

    pub fn weights_on_hypotheses(&self) -> Vec<f64> {
        self.dual.clone()
    }

    /// Lower bound from the entropy dual, when the objective supports it.
    /// Unlike the primal objective, this can be used conservatively in an
    /// outer optimality-gap stopping condition (up to floating-point error).
    pub fn objective_lower_bound(&self) -> Option<f64> {
        self.lower_bound
    }

    pub fn optval(&self) -> f64 {
        self.optval
    }

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

/// The objective function for ErlpBoost.
pub struct EntropyRegularizedMaxEdge(f64);

impl EntropyRegularizedMaxEdge {
    /// Takes the regularization parameter `η > 0` and returns `Self.`
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

/// The objective function for ErlpBoost.
pub struct DeformedEntropyRegularizedMaxEdge {
    t: f64,
    eta: f64,
}

impl DeformedEntropyRegularizedMaxEdge {
    /// Constructs `Self` from the deformation parameter `t ∈ [0, 1]`
    /// and regularization parameter `η > 0.`
    pub fn new(t: f64, eta: f64) -> Self {
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
        let entropy = helpers::deformed_entropy(self.t, &dist);
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
