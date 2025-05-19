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
pub fn edge_minimization<T, H>(
    nu: f64,
    sample: &Sample,
    hypotheses: T,
) -> (f64, Vec<f64>)
    where T: AsRef<[H]>,
          H: Classifier,
{
    let hypotheses   = hypotheses.as_ref();
    let n_hypotheses = hypotheses.len();
    let n_examples   = sample.shape().0;

    checkers::capping_parameter(nu, n_examples);

    let obj = build_objective(n_examples + 1);
    let mat = build_constraint_matrix(sample, hypotheses);
    let sns = build_sns(n_examples, n_hypotheses);
    let rhs = build_rhs(nu, n_examples, n_hypotheses);

    let mut solver = build_solver(obj, mat, sns, rhs);
    solver.solve();

    assert!(
        matches!{ solver.solution.status, SolverStatus::Solved },
        "unexped solver status. got {}",
        solver.solution.status,
    );

    let objval = solver.solution.obj_val;
    let dist = solver.solution.x[1..].iter()
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
fn build_constraint_matrix<H>(sample: &Sample, hypotheses: &[H])
    -> CscMatrix::<f64>
    where H: Classifier,
{
    let n_hypotheses = hypotheses.len();
    let n_examples   = sample.shape().0;

    let start = 1 + 2 * n_examples;
    let end   = 1 + 2 * n_examples + n_hypotheses;

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

fn build_sns(n_examples: usize, n_hypotheses: usize)
    -> Vec<SupportedConeT<f64>>
{
    vec![ZeroConeT(1), NonnegativeConeT(2 * n_examples + n_hypotheses)]
}

fn build_rhs(nu: f64, n_examples: usize, n_hypotheses: usize) -> Vec<f64> {
    iter::once(1f64)
        .chain(iter::repeat_n(0f64, n_examples))
        .chain(iter::repeat_n(1f64/nu, n_examples))
        .chain(iter::repeat_n(0f64, n_hypotheses))
        .collect()
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

/// `RowGeneration` is a struct for row-generation algorithm
/// over a capped simplex `Δ_{m,ν}` for a specified objective function.
pub struct RowGeneration<'a, T> {
    nu: f64,
    sample: &'a Sample,

    objective: T,

    optval: f64,
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
            primal: Vec::new(),
            dual: Vec::new(),
        }
    }

    pub fn initialize(&mut self, nu: f64, objective: T) {
        let n_examples = self.sample.shape().0;
        checkers::capping_parameter(nu, n_examples);
        self.nu = nu;
        self.objective = objective;
    }
}

pub trait RowGenerationObjective {
    fn objective_value<H: Classifier>(
        &self,
        sample: &Sample,
        hypotheses: &[H],
        dist: &[f64],
    ) -> f64;
    fn gradient(&self, dist: &[f64]) -> Vec<f64>;
    fn hessian(&self, dist: &[f64]) -> CscMatrix::<f64>;
}

impl<T> RowGeneration<'_, T>
    where T: RowGenerationObjective,
{
    pub fn solve<H>(&mut self, hypotheses: &[H])
        where H: Classifier,
    {
        let n_hypotheses = hypotheses.len();
        let n_examples   = self.sample.shape().0;

        let mat = build_constraint_matrix(self.sample, hypotheses);
        let sns = build_sns(n_examples, n_hypotheses);
        let rhs = build_rhs(self.nu, n_examples, n_hypotheses);

        let mut current = vec![1f64 / n_examples as f64; n_examples];
        let mut objval = self.objval(hypotheses, &current[..]);
        loop {
            let set = DefaultSettingsBuilder::default()
                .equilibrate_enable(true)
                .verbose(false)
                .build()
                .unwrap();
            let g = self.objective.gradient(&current[..]);
            let h = self.objective.hessian(&current[..]);
            let mut solver = DefaultSolver::new(&h, &g, &mat, &rhs, &sns, set);

            solver.solve();

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

            let solution = solver.solution.x[1..]
                .iter()
                .map(|d| d.clamp(0f64, 1f64/self.nu))
                .collect::<Vec<_>>();
            checkers::capped_simplex_condition(&solution[..], self.nu);
            let optval = self.objval(hypotheses, &solution[..]);

            if objval - optval < SQP_TOLERANCE {
                self.optval = optval;

                self.primal = solution;

                let start = 1 + 2 * n_examples;
                self.dual = solver.solution.z[start..].iter()
                    .map(|w| w.abs())
                    .collect::<Vec<_>>();

                break;
            }

            objval  = optval;
            current = solution;
        }
        checkers::capped_simplex_condition(&self.primal[..], self.nu);
        checkers::capped_simplex_condition(&self.dual[..], 1f64);
    }

    pub fn distribution_on_examples(&self) -> Vec<f64> {
        self.primal.clone()
    }

    pub fn weights_on_hypotheses(&self) -> Vec<f64> {
        self.dual.clone()
    }

    pub fn optval(&self) -> f64 {
        self.optval
    }

    pub fn objval<H>(&self, hypotheses: &[H], dist: &[f64]) -> f64
        where H: Classifier
    {
        self.objective.objective_value(self.sample, hypotheses, dist)
    }
}

/// The objective function for ErlpBoost.
pub struct EntropyRegularizedMaxEdge(f64);

impl EntropyRegularizedMaxEdge {
    /// Takes the regularization parameter `η > 0` and returns `Self.`
    pub fn new(eta: f64) -> Self { Self(eta) }
}

impl RowGenerationObjective for EntropyRegularizedMaxEdge {
    fn objective_value<H: Classifier>(
        &self,
        sample: &Sample,
        hypotheses: &[H],
        dist: &[f64],
    ) -> f64
    {
        assert_eq!(dist.len(), sample.shape().0);

        let eta = self.0;
        let max_edge = hypotheses.iter()
            .map(|h| helpers::edge(sample, dist, h))
            .reduce(f64::max)
            .expect("failed to compute the max-edge.");
        let entropy = helpers::entropy_from_uni_distribution(dist);
        assert!(
            entropy.is_finite(),
            "not a finite entropy. dist = {dist:?}",
        );
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
    fn hessian(&self, dist: &[f64]) -> CscMatrix::<f64> {
        let eta = self.0;
        let n_rows = dist.len() + 1;
        let n_cols = n_rows;

        let n_examples = dist.len();

        let col_ptr = iter::once(0).chain(0..=n_examples).collect();
        let row_idx = (1..=n_examples).collect();
        let nonzero = dist.iter()
            .map(|&d| 1f64 / ((d + PERTURBATION) * eta))
            .collect();
        CscMatrix::new(n_rows, n_cols, col_ptr, row_idx, nonzero)
    }
}

/// The objective function for ErlpBoost.
pub struct DeformedEntropyRegularizedMaxEdge {
    t:   f64,
    eta: f64,
}

impl DeformedEntropyRegularizedMaxEdge {
    /// Constructs `Self` from the deformation parameter `t ∈ [0, 1]`
    /// and regularization parameter `η > 0.`
    pub fn new(t: f64, eta: f64) -> Self {
        Self { t, eta, }
    }
}

impl RowGenerationObjective for DeformedEntropyRegularizedMaxEdge {
    fn objective_value<H: Classifier>(
        &self,
        sample: &Sample,
        hypotheses: &[H],
        dist: &[f64],
    ) -> f64
    {
        assert_eq!(dist.len(), sample.shape().0);
        let max_edge = hypotheses.iter()
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
            .chain(dist.iter().map(|&d| {
                constant
                    * (d + PERTURBATION).powf(1f64 - self.t)
                    / self.eta
            }))
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
    fn hessian(&self, dist: &[f64]) -> CscMatrix::<f64> {
        let n_rows = dist.len() + 1;
        let n_cols = n_rows;

        let n_examples = dist.len();

        let col_ptr = iter::once(0).chain(0..=n_examples).collect();
        let row_idx = (1..=n_examples).collect();

        let constant = (2f64 - self.t) / self.eta;
        let nonzero = dist.iter()
            .map(|&d| constant / (d + PERTURBATION).powf(self.t))
            .collect();
        CscMatrix::new(n_rows, n_cols, col_ptr, row_idx, nonzero)
    }
}

