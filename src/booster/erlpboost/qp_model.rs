use clarabel::{
    algebra::*,
    solver::*,
};

use crate::{
    Sample,
    common::utils,
};

use crate::hypothesis::Classifier;

use std::iter;

const EPS: f64 = 1e-100;
const QP_TOLERANCE: f64 = 1e-9;

/// A quadratic programming model for edge minimization. 
/// `QPModel` solves the entropy regularized edge minimization problem:
///
/// ```txt
/// min γ + (1/η) Σ_i d_i ln( d_i )
/// γ,d
/// s.t. Σ_i d_i y_i h_j (x_i) ≤ γ,   ∀j = 1, 2, ..., t
///      Σ_i d_i = 1,
///      d_1, d_2, ..., d_m ≤ 1/ν
///      d_1, d_2, ..., d_m ≥ 0.
/// ```
/// Since no solver can solve entropy minimization problem,
/// we use the sequential quadratic programming technique:
///
/// Let `q` be the current solution 
/// in the **interior** of `m`-dimensional probability simplex.
/// The following is the second-order approximation of the above problem:
/// ```txt
/// min γ + (1/η) Σ_i [ (1/(2q_i)) d_i^2 + ln( q_i ) d_i ]
/// γ,d
/// s.t. Σ_i d_i y_i h_j (x_i) ≤ γ,   ∀j = 1, 2, ..., t
///      Σ_i d_i = 1,
///      d_1, d_2, ..., d_m ≤ 1/ν
///      d_1, d_2, ..., d_m ≥ 0.
/// ```
/// `QPModel` solves this approximated problem untile convergent.
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
/// cols    1 ┃               m
/// ```
pub(super) struct QPModel {
    pub(self) n_examples: usize,        // number of columns
    pub(self) n_hypotheses: usize,      // number of rows
    pub(self) margins: Vec<Vec<f64>>,   // margin vectors
    pub(self) weights: Vec<f64>,        // weight on hypothesis
    pub(self) dist: Vec<f64>,           // distribution over examples
    pub(self) cap_inv: f64,             // the capping parameter, `1/ν.`
    pub(self) eta: f64,                 // regularization parameter
}


impl QPModel {
    /// Initialize the QP model.
    /// arguments.
    /// - `size`: Number of variables (Number of examples).
    /// - `upper_bound`: Capping parameter. `[1, size]`.
    pub(super) fn init(eta: f64, size: usize, upper_bound: f64) -> Self {
        let dist = {
            let mut dist = vec![1f64 / size as f64; size];
            dist.shrink_to_fit();
            dist
        };
        let margins = vec![vec![]; size];
        Self {
            n_examples:   size,
            n_hypotheses: 0usize,
            margins,
            weights:      Vec::with_capacity(0usize),
            dist,
            cap_inv:      upper_bound,
            eta,
        }
    }


    /// Solve the edge minimization problem 
    /// over the hypotheses `h1, ..., ht` 
    /// and outputs the optimal value.
    pub(super) fn update<F>(
        &mut self,
        sample: &Sample,
        clf: &F
    )
        where F: Classifier
    {
        self.n_hypotheses += 1;
        let margins = utils::margins_of_hypothesis(sample, clf);
        self.margins.iter_mut()
            .zip(margins)
            .for_each(|(mvec, yh)| { mvec.push(yh); });
        let constraint_matrix = self.build_constraint_matrix();
        let sense = self.build_sense();
        let rhs = self.build_rhs();


        let mut old_objval = 1e3;

        // Initialize `dist` as the uniform distribution.
        self.dist.iter_mut()
            .for_each(|di| {
                *di = 1f64 / self.n_examples as f64;
            });
        loop {
            let settings = DefaultSettingsBuilder::default()
                .equilibrate_enable(true)
                .verbose(false)
                .build()
                .unwrap();
            let linear = self.build_linear_part_objective(&self.dist[..]);
            let quad   = self.build_quadratic_part_objective(&self.dist[..]);
            let mut solver = DefaultSolver::new(
                &quad,
                &linear,
                &constraint_matrix,
                &rhs,
                &sense[..],
                settings
            );

            solver.solve();
            assert!(
                solver.solution.status == SolverStatus::Solved
                || solver.solution.status == SolverStatus::AlmostSolved
                || solver.solution.status == SolverStatus::InsufficientProgress,
                "unexpected solver status. got {}",
                solver.solution.status,
            );

            let objval = solver.solution.obj_val;
            if old_objval - objval < QP_TOLERANCE
                || solver.solution.status == SolverStatus::InsufficientProgress
            {
                break;
            }
            old_objval = objval;
            self.dist = solver.solution.x[1..].into_iter()
                .map(|di| di.max(0f64))
                .collect();
            let start = 1 + 2 * self.n_examples;
            self.weights = solver.solution.z[start..].into_iter()
                .map(|wi| wi.max(0f64))
                .collect();
        }
    }


    pub(self) fn build_linear_part_objective(&self, dist: &[f64]) -> Vec<f64> {
        let mut linear = Vec::with_capacity(1 + self.n_examples);
        linear.push(1f64);
        let iter = dist.iter()
            .copied()
            .map(|di| {
                let coef = (di + EPS).ln();
                assert!(coef.is_finite());
                coef / self.eta
            });
        linear.extend(iter);
        linear
    }


    pub(self) fn build_quadratic_part_objective(&self, dist: &[f64])
        -> CscMatrix::<f64>
    {
        let eta = self.eta;
        let n_rows = dist.len() + 1;
        let n_cols = n_rows;

        let n_examples = dist.len();

        let col_ptr = iter::once(0).chain(0..=n_examples).collect();
        let row_idx = (1..=n_examples).collect();
        let nonzero = dist.iter()
            .map(|&d| 1f64 / ((d + EPS) * eta))
            .collect();
        CscMatrix::new(n_rows, n_cols, col_ptr, row_idx, nonzero)
    }


    /// Build the constraint matrix in the 0-indexed CSC form.
    pub(self) fn build_constraint_matrix(&self) -> CscMatrix::<f64> {
        let n_rows = 1 + 2*self.n_examples + self.n_hypotheses;
        let n_cols = 1 + self.n_examples;

        let mut col_ptr = Vec::new();
        let mut row_idx = Vec::new();
        let mut nonzero = Vec::new();

        // the first index of margin constraints
        let gam = 1 + 2 * self.n_examples;
        col_ptr.push(0);
        row_idx.extend(gam..n_rows);
        nonzero.extend(iter::repeat(-1f64).take(n_rows - gam));

        for (j, margins) in (1..).zip(&self.margins) {
            col_ptr.push(row_idx.len());
            // the sum constraint: `Σ_i d_i = 1`
            row_idx.push(0);
            nonzero.push(1f64);

            // non-negative constraint: `d_i ≥ 0`
            row_idx.push(j);
            nonzero.push(-1f64);

            // capping constraint: `d_i ≤ 1/ν`
            row_idx.push(self.n_examples + j);
            nonzero.push(1f64);

            // margin constraints of `i`-th column
            for (i, &yh) in (0..).zip(margins) {
                row_idx.push(gam + i);
                nonzero.push(yh);
            }
        }
        col_ptr.push(row_idx.len());

        CscMatrix::new(
            n_rows,
            n_cols,
            col_ptr,
            row_idx,
            nonzero,
        )
    }


    /// Build the vector of constraint sense: `[=, ≤, ≥, ...].`
    pub(self) fn build_sense(&self) -> Vec<SupportedConeT<f64>> {
        let n_ineq = 2*self.n_examples + self.n_hypotheses;
        vec![
            ZeroConeT(1),
            NonnegativeConeT(n_ineq),
        ]
    }


    /// Build the right-hand-side of the constraints.
    pub(self) fn build_rhs(&self) -> Vec<f64> {
        let n_constraints = 1 + 2*self.n_examples + self.n_hypotheses;
        let mut rhs = Vec::with_capacity(n_constraints);
        rhs.push(1f64);
        rhs.extend(iter::repeat(0f64).take(self.n_examples));
        rhs.extend(iter::repeat(self.cap_inv).take(self.n_examples));
        rhs.extend(iter::repeat(0f64).take(self.n_hypotheses));
        rhs
    }

    /// Returns the distribution over examples.
    pub(super) fn distribution(&self)
        -> Vec<f64>
    {
        self.dist.clone()
    }


    /// Returns the weights over the hypotheses.
    pub(super) fn weight(&self) -> impl Iterator<Item=f64> + '_
    {
        self.weights.iter().copied()
    }
}


