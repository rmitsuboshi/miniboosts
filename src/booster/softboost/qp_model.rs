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

const QP_TOLERANCE: f64 = 1e-9;

/// A quadratic programming model for edge minimization. 
/// `QPModel` solves the entropy regularized edge minimization problem:
///
/// ```txt
/// min Σ_i d_i ln( d_i )
///  d
/// s.t. Σ_i d_i y_i h_j (x_i) ≤ γ,   ∀j = 1, 2, ..., t
///      Σ_i d_i = 1,
///      d_1, d_2, ..., d_m ≤ 1/ν
///      d_1, d_2, ..., d_m ≥ 0.
/// ```
/// where `γ` is the estimation of the weak-learnability.
///
/// Since no solver can solve entropy minimization problem,
/// we use the sequential quadratic programming technique:
///
/// Let `q` be the current solution 
/// in the **interior** of `m`-dimensional probability simplex.
/// The following is the second-order approximation of the above problem:
/// ```txt
/// min Σ_i [ (1/(2q_i)) d_i^2 + ln( q_i ) d_i ]
///  d
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
/// rows        d1        ...     dm
///       ┏                                 ┓   ┏     ┓
///   1   ┃      1        ...      1        ┃ = ┃  1  ┃
///      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
///       ┃                                 ┃ ≤ ┃  0  ┃
///       ┃                                 ┃ ≤ ┃  0  ┃
///       ┃     (-1) * Identity matrix      ┃ . ┃  .  ┃
///   m   ┃              m x m              ┃ . ┃  .  ┃
///       ┃                                 ┃ . ┃  .  ┃
///       ┃                                 ┃ ≤ ┃  0  ┃
///      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
///       ┃                                 ┃ ≤ ┃ 1/ν ┃
///       ┃                                 ┃ ≤ ┃ 1/ν ┃
///   m   ┃         Identity matrix         ┃ . ┃  .  ┃
///       ┃              m x m              ┃ . ┃  .  ┃
///       ┃                                 ┃ . ┃  .  ┃
///       ┃                                 ┃ ≤ ┃ 1/ν ┃
///      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
///       ┃  y_1 h_1(x_1) ...  y_m h_1(x_m) ┃ ≤ ┃  0  ┃
///       ┃  y_1 h_2(x_1) ...  y_m h_2(x_m) ┃ ≤ ┃  0  ┃
///       ┃      .        ...      .        ┃ . ┃  .  ┃
///   H   ┃      .        ...      .        ┃ . ┃  .  ┃
///       ┃      .        ...      .        ┃ . ┃  .  ┃
///       ┃  y_1 h_T(x_1) ...  y_m h_T(x_m) ┃ ≤ ┃  0  ┃
///       ┗                                 ┛   ┗     ┛
///
/// # of
/// cols                   m
/// ```
pub(super) struct QPModel {
    pub(self) n_examples: usize,        // number of columns
    pub(self) n_hypotheses: usize,      // number of rows
    pub(self) margins: Vec<Vec<f64>>,   // margin vectors
    pub(self) cap_inv: f64,             // the capping parameter, `1/ν.`
}


impl QPModel {
    /// Initialize the QP model.
    /// arguments.
    /// - `size`: Number of variables (Number of examples).
    /// - `upper_bound`: Capping parameter. `[1, size]`.
    pub(super) fn init(size: usize, upper_bound: f64) -> Self {
        let margins = vec![vec![]; size];
        Self {
            n_examples:   size,
            n_hypotheses: 0usize,
            margins,
            cap_inv:      upper_bound,
        }
    }


    /// Solve the edge minimization problem 
    /// over the hypotheses `h1, ..., ht` 
    /// and outputs the optimal value.
    pub(super) fn update<F>(
        &mut self,
        sample: &Sample,
        dist: &mut [f64],
        ghat: f64,
        clf: &F
    ) -> Option<()>
        where F: Classifier
    {
        self.n_hypotheses += 1;
        let margins = utils::margins_of_hypothesis(sample, clf);
        self.margins.iter_mut()
            .zip(margins)
            .for_each(|(mvec, yh)| { mvec.push(yh); });
        let constraint_matrix = self.build_constraint_matrix();
        let sense = self.build_sense();
        let rhs = self.build_rhs(ghat);


        let mut old_objval = 1e9;
        loop {
            let settings = DefaultSettingsBuilder::default()
                .equilibrate_enable(true)
                .verbose(false)
                .build()
                .unwrap();
            let linear = self.build_linear_part_objective(dist);
            let quad   = self.build_quadratic_part_objective(dist);
            let mut solver = DefaultSolver::new(
                &quad,
                &linear,
                &constraint_matrix,
                &rhs,
                &sense[..],
                settings
            );

            solver.solve();
            let solution = &solver.solution.x[..];

            // breaks if there is a zero-valued coordinate in the solution.
            if !self.all_positive(solution) { return None; }
            dist.iter_mut()
                .zip(solution)
                .for_each(|(di, s)| { *di = *s; });

            let objval = solver.solution.obj_val;
            if old_objval - objval < QP_TOLERANCE {
                break;
            }
            old_objval = objval;
        }
        Some(())
    }


    /// Returns `true` if `dist[i] > 0` holds for all `i = 1, 2, ..., m.` 
    pub(self) fn all_positive(&self, dist: &[f64]) -> bool {
        dist.iter()
            .copied()
            .all(|d| d > 0f64)
    }


    pub(self) fn build_linear_part_objective(&self, dist: &[f64]) -> Vec<f64> {
        let mut linear = Vec::with_capacity(self.n_examples);
        let iter = dist.iter()
            .copied()
            .map(|di| di.ln());
        linear.extend(iter);
        linear
    }


    pub(self) fn build_linear_part_objective_lp(&self)
        -> Vec<f64>
    {
        let mut linear = vec![0f64; 1 + self.n_examples];
        linear[0] = 1f64;
        linear
    }


    pub(self) fn build_quadratic_part_objective(&self, dist: &[f64])
        -> CscMatrix::<f64>
    {
        let n_rows = self.n_examples;
        let n_cols = n_rows;

        let mut col_ptr = Vec::with_capacity(n_cols);
        let mut row_val = Vec::with_capacity(n_cols);
        let mut nonzero = Vec::with_capacity(n_cols);

        for (i, &di) in (0..).zip(dist) {
            col_ptr.push(i);
            row_val.push(i);
            nonzero.push(1f64 / di);
        }
        col_ptr.push(row_val.len());

        CscMatrix::new(
            n_rows,
            n_cols,
            col_ptr,
            row_val,
            nonzero,
        )
    }


    /// Build the constraint matrix in the 0-indexed CSC form.
    pub(self) fn build_constraint_matrix(&self) -> CscMatrix::<f64> {
        let n_rows = 1 + 2*self.n_examples + self.n_hypotheses;
        let n_cols = self.n_examples;

        let mut col_ptr = Vec::new();
        let mut row_val = Vec::new();
        let mut nonzero = Vec::new();

        // the first index of margin constraints
        let gam = 1 + 2 * self.n_examples;

        for (j, margins) in self.margins.iter().enumerate() {
            col_ptr.push(row_val.len());
            // the sum constraint: `Σ_i d_i = 1`
            row_val.push(0);
            nonzero.push(1f64);

            // non-negative constraint: `d_i ≥ 0`
            row_val.push(j + 1);
            nonzero.push(-1f64);

            // capping constraint: `d_i ≤ 1/ν`
            row_val.push(self.n_examples + j + 1);
            nonzero.push(self.cap_inv);

            // margin constraints of `i`-th column
            for (i, &yh) in (0..).zip(margins) {
                row_val.push(gam + i);
                nonzero.push(yh);
            }
        }
        col_ptr.push(row_val.len());

        CscMatrix::new(
            n_rows,
            n_cols,
            col_ptr,
            row_val,
            nonzero,
        )
    }


    /// Build the constraint matrix in the 0-indexed CSC form.
    pub(self) fn build_constraint_matrix_lp(&self) -> CscMatrix::<f64> {
        let n_rows = 1 + 2*self.n_examples + self.n_hypotheses;
        let n_cols = 1 + self.n_examples;

        let mut col_ptr = Vec::new();
        let mut row_val = Vec::new();
        let mut nonzero = Vec::new();

        // the first index of margin constraints
        let gam = 1 + 2 * self.n_examples;
        col_ptr.push(0);
        row_val.extend(gam..n_rows);
        nonzero.extend(iter::repeat(-1f64).take(n_rows - gam));

        for (j, margins) in (1..).zip(&self.margins) {
            col_ptr.push(row_val.len());
            // the sum constraint: `Σ_i d_i = 1`
            row_val.push(0);
            nonzero.push(1f64);

            // non-negative constraint: `d_i ≥ 0`
            row_val.push(j);
            nonzero.push(-1f64);

            // capping constraint: `d_i ≤ 1/ν`
            row_val.push(self.n_examples + j);
            nonzero.push(1f64);

            // margin constraints of `i`-th column
            for (i, &yh) in (0..).zip(margins) {
                row_val.push(gam + i);
                nonzero.push(yh);
            }
        }
        col_ptr.push(row_val.len());

        CscMatrix::new(
            n_rows,
            n_cols,
            col_ptr,
            row_val,
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


    /// Build the vector of constraint sense: `[=, ≤, ≥, ...].`
    pub(self) fn build_sense_lp(&self) -> Vec<SupportedConeT<f64>> {
        self.build_sense()
    }


    /// Build the right-hand-side of the constraints.
    pub(self) fn build_rhs(&self, ghat: f64) -> Vec<f64> {
        let n_constraints = 1 + 2*self.n_examples + self.n_hypotheses;
        let mut rhs = Vec::with_capacity(n_constraints);
        rhs.push(1f64);
        rhs.extend(iter::repeat(0f64).take(self.n_examples));
        rhs.extend(iter::repeat(self.cap_inv).take(self.n_examples));
        rhs.extend(iter::repeat(ghat).take(self.n_hypotheses));
        rhs
    }


    /// Build the right-hand-side of the constraints.
    pub(self) fn build_rhs_lp(&self) -> Vec<f64> {
        let n_constraints = 1 + 2*self.n_examples + self.n_hypotheses;
        let mut rhs = Vec::with_capacity(n_constraints);
        rhs.push(1f64);
        rhs.extend(iter::repeat(0f64).take(self.n_examples));
        rhs.extend(iter::repeat(self.cap_inv).take(self.n_examples));
        rhs.extend(iter::repeat(0f64).take(self.n_hypotheses));
        rhs
    }


    /// Returns the weights over the hypotheses.
    pub(super) fn weights<F>(
        &self,
        _sample: &Sample,
        _hypotheses: &[F]
    ) -> impl Iterator<Item=f64> + '_
        where F: Classifier
    {
        let n_variables = self.n_examples + 1;
        let constraint_matrix = self.build_constraint_matrix_lp();
        let sense = self.build_sense_lp();
        let rhs = self.build_rhs_lp();


        let settings = DefaultSettingsBuilder::default()
            .equilibrate_enable(true)
            .verbose(false)
            .build()
            .unwrap();
        let linear = self.build_linear_part_objective_lp();
        let mut solver = DefaultSolver::new(
            &CscMatrix::zeros((n_variables, n_variables)),
            &linear,
            &constraint_matrix,
            &rhs,
            &sense[..],
            settings
        );

        solver.solve();
        let start = 1 + 2*self.n_examples;
        let solution = solver.solution.z[start..].to_vec();
        solution.into_iter()
    }
}


