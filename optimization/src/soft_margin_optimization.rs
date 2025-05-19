use miniboosts_core::{
    Sample,
    Classifier,
    tools::{
        helpers,
        checkers,
    },
};

use clarabel::{
    algebra::*,
    solver::*,
};

use std::iter;

/// A linear programming model for edge minimization. 
/// `LPModel` solves the soft margin optimization:
///
/// ```txt
/// max ρ - (1/ν) Σ_i ξ_i
/// s.t. y_i Σ_j w_j h_j (x_i) ≥ ρ - ξ_i,   ∀i = 1, 2, ..., m
///      Σ_j w_j = 1,
///      w_1, w_2, ..., w_T ≥ 0,
///      ξ_1, ξ_2, ..., ξ_m ≥ 0.
/// ```
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
/// where, the first `m` rows correspond to the inequality constraints,
/// while the last row corresponds to the simplex constraint.
///
/// Since the `clarabel` crate solves the minimization problems,
/// we need to negate the objective function.
pub fn soft_margin_optimization<T, H>(
    // capping parameter
    nu: f64,
    // training examples
    sample: &Sample,
    // hypotheses
    hypotheses: T,
) -> (f64, Vec<f64>)
    where T: AsRef<[H]>,
          H: Classifier,
{
    let hypotheses   = hypotheses.as_ref();
    let n_hypotheses = hypotheses.len();
    let n_examples   = sample.shape().0;

    checkers::capping_parameter(nu, n_examples);

    let obj = build_objective(nu, n_examples, n_hypotheses);

    let mat = build_constraint_matrix(sample, hypotheses);
    let sns = build_sns(n_examples, n_hypotheses);
    let rhs = build_rhs(n_examples, n_hypotheses);

    let mut solver = build_solver(obj, mat, sns, rhs);
    solver.solve();

    assert!(
        matches!{ solver.solution.status, SolverStatus::Solved },
        "unexped solver status. got {}",
        solver.solution.status,
    );

    let objval = - solver.solution.obj_val;
    let start  = 1 + n_examples;
    let weight = solver.solution.x[start..]
        .iter()
        .map(|w| w.max(0f64))
        .collect::<Vec<f64>>();

    (objval, weight)
}

fn build_objective(nu: f64, n_examples: usize, n_hypotheses: usize)
    -> Vec<f64>
{
    iter::once(-1f64)
        .chain(iter::repeat_n(1f64 / nu, n_examples))
        .chain(iter::repeat_n(0f64, n_hypotheses))
        .collect()
}

fn build_constraint_matrix<H>(sample: &Sample, hypotheses: &[H])
    -> CscMatrix::<f64>
    where H: Classifier,
{
    let n_hypotheses = hypotheses.len();
    let n_examples   = sample.shape().0;

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
        let iter = helpers::margins(sample, h)
            .enumerate();
        for (i, yhx) in iter {
            row_idx.push(i);
            nonzero.push(- yhx);
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

fn build_sns(n_examples: usize, n_hypotheses: usize)
    -> Vec<SupportedConeT<f64>>
{
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
    mat: CscMatrix::<f64>,
    sns: Vec<SupportedConeT<f64>>,
    rhs: Vec<f64>,
) -> DefaultSolver<f64>
{
    let settings = DefaultSettingsBuilder::default()
        .equilibrate_enable(true)
        .verbose(false)
        .build()
        .unwrap();

    let n_variables = obj.len();
    let zmx = CscMatrix::<f64>::zeros((n_variables, n_variables));

    DefaultSolver::new(&zmx, &obj, &mat, &rhs, &sns, settings)
}

/// The column generation algorithm for soft margin optimization.
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

    pub fn initialize(&mut self, n_examples: usize, nu: f64) {
        self.nu = nu;

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

    pub fn solve<H>(&mut self, h: &H)
        where H: Classifier,
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
            matches!{
                solver.solution.status,
                SolverStatus::Solved | SolverStatus::AlmostSolved
            },
            "unexped solver status. got {}",
            solver.solution.status,
        );

        self.optval = - solver.solution.obj_val;

        let start  = 1 + n_examples;
        self.primal = solver.solution.x[start..]
            .iter()
            .map(|w| w.max(0f64))
            .collect::<Vec<f64>>();
        checkers::capped_simplex_condition(&self.primal[..], 1f64);

        self.dual = solver.solution.z[..n_examples].iter()
            .map(|d| d.abs())
            .collect::<Vec<f64>>();
        checkers::capped_simplex_condition(&self.dual[..], self.nu);
    }

    fn append_matrix<H>(&mut self, h: &H) -> CscMatrix::<f64>
        where H: Classifier
    {
        let n_examples   = self.sample.shape().0;
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

    pub fn optimal_value(&self) -> f64 {
        self.optval
    }

    pub fn weights_on_hypotheses(&self) -> Vec<f64> {
        self.primal.clone()
    }

    pub fn distribution_on_examples(&self) -> Vec<f64> {
        self.dual.clone()
    }
}

