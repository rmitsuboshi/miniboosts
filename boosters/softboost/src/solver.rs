use miniboosts_core::{
    Sample,
    Classifier,
    tools::checkers,
    tools::helpers,
    constants::{
        SQP_TOLERANCE,
        PERTURBATION,
        DEFAULT_CAPPING,
    },
};

use clarabel::{
    algebra::*,
    solver::*,
};

use std::iter;

/// Build the constraint matrix:
/// ```txt
/// # of   
/// rows       d1        ...     dm
///       ┏                                ┓
///   1   ┃     1        ...      1        ┃
///      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
///       ┃                                ┃
///       ┃                                ┃
///       ┃    (-1) * Identity matrix      ┃
///   m   ┃             m x m              ┃
///       ┃                                ┃
///       ┃                                ┃
///      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
///       ┃                                ┃
///       ┃                                ┃
///   m   ┃        Identity matrix         ┃
///       ┃             m x m              ┃
///       ┃                                ┃
///       ┃                                ┃
///      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
///       ┃ y_1 h_1(x_1) ...  y_m h_1(x_m) ┃
///       ┃ y_1 h_2(x_1) ...  y_m h_2(x_m) ┃
///       ┃     .        ...      .        ┃
///   H   ┃     .        ...      .        ┃
///       ┃     .        ...      .        ┃
///       ┃ y_1 h_T(x_1) ...  y_m h_T(x_m) ┃
///       ┗                                ┛
///
/// # of
/// cols                 m
/// ```
fn build_constraint_matrix<H>(sample: &Sample, hypotheses: &[H])
    -> CscMatrix::<f64>
    where H: Classifier,
{
    let n_hypotheses = hypotheses.len();
    let n_examples   = sample.shape().0;

    let mut col_ptr = Vec::new();
    let mut row_idx = Vec::new();
    let mut nonzero = Vec::new();

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

    assert_eq!(col_ptr.len(), n_examples + 1);

    let n_cols = n_examples;
    let n_rows = 1 + 2 * n_examples + n_hypotheses;
    CscMatrix::new(n_rows, n_cols, col_ptr, row_idx, nonzero)
}

fn build_sns(n_examples: usize, n_hypotheses: usize)
    -> Vec<SupportedConeT<f64>>
{
    vec![ZeroConeT(1), NonnegativeConeT(2 * n_examples + n_hypotheses)]
}

fn build_rhs(
    nu: f64,
    gamma: f64,
    delta: f64,
    n_examples: usize,
    n_hypotheses: usize,
) -> Vec<f64>
{
    iter::once(1f64)
        .chain(iter::repeat_n(0f64, n_examples))
        .chain(iter::repeat_n(1f64/nu, n_examples))
        .chain(iter::repeat_n(gamma - delta, n_hypotheses))
        .collect()
}

/// A solver for SoftBoost.
/// This solver solves optimization problems of the form:
/// ```txt
/// min Σ_i d_i ln(d_i)
/// γ,d
/// s.t. Σ_i d_i y_i h_j (x_i) ≤ γ_t - δ,   ∀j = 1, 2, ..., t
///      Σ_i d_i = 1,
///      d_1, d_2, ..., d_m ≤ 1/ν
///      d_1, d_2, ..., d_m ≥ 0.
/// ```
/// where `γ_t` is the current estimation of weak learnability.
///
/// To solve the problem we build the constraint matrix
/// ```txt
/// # of   
/// rows       d1        ...     dm
///       ┏                                ┓   ┏         ┓
///   1   ┃     1        ...      1        ┃ = ┃    1    ┃
///      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
///       ┃                                ┃ ≤ ┃    0    ┃
///       ┃                                ┃ ≤ ┃    0    ┃
///       ┃    (-1) * Identity matrix      ┃ . ┃    .    ┃
///   m   ┃             m x m              ┃ . ┃    .    ┃
///       ┃                                ┃ . ┃    .    ┃
///       ┃                                ┃ ≤ ┃    0    ┃
///      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
///       ┃                                ┃ ≤ ┃   1/ν   ┃
///       ┃                                ┃ ≤ ┃   1/ν   ┃
///   m   ┃        Identity matrix         ┃ . ┃    .    ┃
///       ┃             m x m              ┃ . ┃    .    ┃
///       ┃                                ┃ . ┃    .    ┃
///       ┃                                ┃ ≤ ┃   1/ν   ┃
///      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
///       ┃ y_1 h_1(x_1) ...  y_m h_1(x_m) ┃ ≤ ┃ γ_t - δ ┃
///       ┃ y_1 h_2(x_1) ...  y_m h_2(x_m) ┃ ≤ ┃ γ_t - δ ┃
///       ┃     .        ...      .        ┃ . ┃    .    ┃
///   H   ┃     .        ...      .        ┃ . ┃    .    ┃
///       ┃     .        ...      .        ┃ . ┃    .    ┃
///       ┃ y_1 h_T(x_1) ...  y_m h_T(x_m) ┃ ≤ ┃ γ_t - δ ┃
///       ┗                                ┛   ┗         ┛
///
/// # of
/// cols                 m
/// ```
pub struct SoftBoostSolver<'a> {
    nu: f64,
    sample: &'a Sample,

    primal: Vec<f64>,
}

impl<'a> SoftBoostSolver<'a> {
    pub fn new(sample: &'a Sample) -> Self {
        Self {
            sample,
            nu:     DEFAULT_CAPPING,
            primal: Vec::new(),
        }
    }

    pub fn initialize(&mut self, nu: f64) {
        let n_examples = self.sample.shape().0;
        checkers::capping_parameter(nu, n_examples);
        self.nu = nu;
    }
}

impl SoftBoostSolver<'_>
{
    pub fn solve<H>(&mut self, gamma: f64, delta: f64, hypotheses: &[H])
        -> Option<()>
        where H: Classifier,
    {
        let n_hypotheses = hypotheses.len();
        let n_examples   = self.sample.shape().0;

        let mat = build_constraint_matrix(self.sample, hypotheses);
        let sns = build_sns(n_examples, n_hypotheses);
        let rhs = build_rhs(self.nu, gamma, delta, n_examples, n_hypotheses);

        let mut current = vec![1f64 / n_examples as f64; n_examples];
        let mut objval = self.objval(&current[..]);
        loop {
            let set = DefaultSettingsBuilder::default()
                .equilibrate_enable(true)
                .verbose(false)
                .build()
                .unwrap();
            let g = self.gradient(&current[..]);
            let h = self.hessian(&current[..]);
            let mut solver = DefaultSolver::new(&h, &g, &mat, &rhs, &sns, set);

            solver.solve();
            let status = solver.solution.status;

            if matches!{ status, SolverStatus::PrimalInfeasible } {
                return None;
            }

            assert!(
                matches!{
                    solver.solution.status,
                    SolverStatus::Solved
                        | SolverStatus::AlmostSolved
                        | SolverStatus::InsufficientProgress
                },
                "unexped solver status. got {}",
                solver.solution.status,
            );

            if solver.solution.status == SolverStatus::InsufficientProgress {
                println!("warning! solver status is InsufficientProgress");
            }

            let solution = solver.solution.x[..]
                .iter()
                .map(|d| d.clamp(0f64, 1f64/self.nu))
                .collect::<Vec<_>>();
            checkers::capped_simplex_condition(&solution[..], self.nu);
            let optval = self.objval(&solution[..]);

            if objval - optval < SQP_TOLERANCE {
                self.primal = solution;

                break;
            }

            objval  = optval;
            current = solution;
        }
        checkers::capped_simplex_condition(&self.primal[..], self.nu);
        Some(())
    }

    pub fn distribution_on_examples(&self) -> Vec<f64> {
        self.primal.clone()
    }

    pub fn objval(&self, dist: &[f64]) -> f64 {
        helpers::entropy_from_uni_distribution(dist)
    }

    /// outputs the gradient vector at a given point `d`:
    /// ```txt
    /// ┏                            ┓
    /// ┃ ln(d₁)  ln(d₂)  ... ln(dm) ┃
    /// ┗                            ┛
    /// ```
    pub fn gradient(&self, dist: &[f64]) -> Vec<f64> {
        dist.iter().map(|&d| (d + PERTURBATION).ln()).collect()
    }

    /// outputs the hessian at a given point `d`:
    /// ```txt
    ///       d₁   d₂  ...  dm
    ///    ┏                    ┓
    /// d₁ ┃ 1/d₁  0   ...  0   ┃
    /// d₂ ┃  0   1/d₂  0       ┃
    ///  . ┃  .    0   1/d₃     ┃
    ///  . ┃  .                 ┃
    ///  . ┃  .                 ┃
    /// dm ┃  0   ...  ... 1/dm ┃
    ///    ┗                    ┛
    /// ```
    fn hessian(&self, dist: &[f64]) -> CscMatrix::<f64> {
        let n_rows = dist.len();
        let n_cols = n_rows;

        let m = dist.len();

        let col_ptr = (0..=m).collect();
        let row_idx = (0..m).collect();
        let nonzero = dist.iter()
            .map(|&d| 1f64 / (d + PERTURBATION))
            .collect();
        CscMatrix::new(n_rows, n_cols, col_ptr, row_idx, nonzero)
    }
}

