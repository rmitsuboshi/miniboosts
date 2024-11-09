use clarabel::{
    algebra::*,
    solver::*,
};

use crate::{
    Sample,
    common::utils,
};
use crate::hypothesis::Classifier;

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
pub(super) struct LPModel {
    // -----
    // clarabel settings
    pub(self) lin_obj: Vec<f64>,        // LP objective
    pub(self) nonzero: Vec<f64>,        // non-zero values in constraint matrix
    pub(self) col_ptr: Vec<usize>,      // column pointer
    pub(self) row_val: Vec<usize>,      // row value
    // End of clarabel setting
    // -----
    pub(self) n_examples: usize,        // number of columns
    pub(self) n_hypotheses: usize,      // number of rows
    pub(self) weights: Vec<f64>,        // weight on hypothesis
    pub(self) dist: Vec<f64>,           // distribution over examples
}


impl LPModel {
    /// Initialize the LP model.
    /// arguments.
    /// - `size`: Number of variables (Number of examples).
    /// - `upper_bound`: Capping parameter. `[1, size]`.
    pub(super) fn init(size: usize, upper_bound: f64) -> Self {
        let n_examples = size;
        // Set the linear part of the objective function 
        // as the minimization form
        // - ρ + (1/ν) Σ_i ξ_i
        let mut lin_obj = vec![1f64/upper_bound; n_examples+1];
        lin_obj[0] = -1f64;

        let mut col_ptr = vec![0usize];
        let mut row_val = (0usize..n_examples).collect::<Vec<usize>>();


        let mut nonzero = vec![1f64; n_examples];
        // Adding the constraint column vectors for ξ.
        for r in 0..n_examples {
            col_ptr.push(row_val.len());
            row_val.push(r);
            nonzero.push(-1f64);
            row_val.push(n_examples + 1 + r);
            nonzero.push(-1f64);
        }

        Self {
            lin_obj,
            nonzero,
            col_ptr,
            row_val,
            n_examples,
            n_hypotheses: 0usize,
            weights:      Vec::with_capacity(0usize),
            dist:         Vec::with_capacity(0usize),
        }
    }


    /// Solve the edge minimization problem 
    /// over the hypotheses `h1, ..., ht` 
    /// and outputs the optimal value.
    pub(super) fn update<F>(
        &mut self,
        sample: &Sample,
        opt_h: Option<&F>
    ) -> Vec<f64>
        where F: Classifier
    {
        if let Some(clf) = opt_h {
            self.n_hypotheses += 1;
            let margins = utils::margins_of_hypothesis(sample, clf);
            self.col_ptr.push(self.row_val.len());
            for (i, yh) in margins.into_iter().enumerate() {
                self.row_val.push(i);
                self.nonzero.push(-yh);
            }
            // append 1 for equality constraint.
            self.row_val.push(self.n_examples);
            self.nonzero.push(1f64);
            // append 1 for non-negative constraint of weight on `clf.`
            self.row_val.push(2*self.n_examples + self.n_hypotheses);
            self.nonzero.push(-1f64);

            // In the CSC format, the following is equired:
            let n_rows = 2 * self.n_examples + self.n_hypotheses + 1;
            let n_cols = self.n_examples + self.n_hypotheses + 1;
            let mut col_ptr = self.col_ptr.clone();
            col_ptr.push(self.row_val.len());
            let row_val = self.row_val.clone();
            let nonzero = self.nonzero.clone();
            let constraint_matrix = CscMatrix::new(
                n_rows,  // # of rows
                n_cols,  // # of cols
                col_ptr, // col ptr
                row_val, // row val
                nonzero, // non-zero values
            );

            let mut rhs = vec![0f64; 2*self.n_examples + self.n_hypotheses + 1];
            rhs[self.n_examples] = 1f64;
            let cones = [
                NonnegativeConeT(self.n_examples),
                ZeroConeT(1),
                NonnegativeConeT(self.n_examples),
                NonnegativeConeT(self.n_hypotheses),
            ];

            let settings = DefaultSettingsBuilder::default()
                .equilibrate_enable(true)
                .verbose(false)
                .max_iter(u32::MAX)
                .build()
                .unwrap();

            let n_variables = 1 + self.n_examples + self.n_hypotheses;
            let zero_mat = CscMatrix::<f64>::zeros((n_variables, n_variables));
            self.lin_obj.push(0f64);
            let mut solver = DefaultSolver::new(
                &zero_mat,
                &self.lin_obj,
                &constraint_matrix,
                &rhs[..],
                &cones,
                settings
            );

            solver.solve();
            // `size` is the first index of weights on hypotheses.
            //         here
            //          ↓
            // [ ρ, ξ, w[0], w[1], ..., w[T] ]
            let size = 1 + self.n_examples;
            self.weights = solver.solution.x[size..].to_vec();
            self.dist = solver.solution.z[..self.n_examples].to_vec();

            let wsum = self.weights.iter().sum::<f64>();
            if (wsum - 1f64).abs() > 1e-6 {
                eprintln!(
                    "[WRN] weight sum on hypotheses far from 1. sum is: {wsum}"
                );
            }
            let dsum = self.dist.iter().sum::<f64>();
            if (dsum - 1f64).abs() > 1e-6 {
                eprintln!(
                    "[WRN] dist sum on examples far from 1. sum is: {dsum}"
                );
            }
        }

        self.dist.clone()
    }
}


