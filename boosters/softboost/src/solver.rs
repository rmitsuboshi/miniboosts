use miniboosts_core::{
    Classifier, Sample, constants::DEFAULT_CAPPING, tools::checkers, tools::helpers,
};

use clarabel::{algebra::*, solver::*};

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
fn build_constraint_matrix<H>(sample: &Sample, hypotheses: &[H]) -> CscMatrix<f64>
where
    H: Classifier,
{
    let n_hypotheses = hypotheses.len();
    let n_examples = sample.shape().0;

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

fn build_sns(n_examples: usize, n_hypotheses: usize) -> Vec<SupportedConeT<f64>> {
    vec![
        ZeroConeT(1),
        NonnegativeConeT(2 * n_examples + n_hypotheses),
    ]
}

fn build_rhs(nu: f64, gamma: f64, delta: f64, n_examples: usize, n_hypotheses: usize) -> Vec<f64> {
    iter::once(1f64)
        .chain(iter::repeat_n(0f64, n_examples))
        .chain(iter::repeat_n(1f64 / nu, n_examples))
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
            nu: DEFAULT_CAPPING,
            primal: Vec::new(),
        }
    }

    pub fn initialize(&mut self, nu: f64) {
        let n_examples = self.sample.shape().0;
        checkers::capping_parameter(nu, n_examples);
        self.nu = nu;
        self.primal.clear();
    }
}

impl SoftBoostSolver<'_> {
    pub fn solve<H>(&mut self, gamma: f64, delta: f64, hypotheses: &[H]) -> Option<()>
    where
        H: Classifier,
    {
        let n_hypotheses = hypotheses.len();
        let n_examples = self.sample.shape().0;

        assert!(gamma.is_finite() && delta.is_finite() && delta > 0.0);
        self.primal.clear();
        let linear = build_constraint_matrix(self.sample, hypotheses);
        assert!(linear.nzval.iter().all(|v| v.is_finite()));
        let offset = linear.m;
        let mut colptr = vec![0];
        let mut rowval = Vec::new();
        let mut nzval = Vec::new();
        // Variables are (d, t). The cone slack (-t_i, d_i, 1) gives
        // d_i * exp(-t_i / d_i) <= 1, or t_i >= d_i ln(d_i).
        // The closed cone also represents d_i = 0 without log perturbations.
        for i in 0..n_examples {
            for k in linear.colptr[i]..linear.colptr[i + 1] {
                rowval.push(linear.rowval[k]);
                nzval.push(linear.nzval[k]);
            }
            rowval.push(offset + 3 * i + 1);
            nzval.push(-1.0);
            colptr.push(rowval.len());
        }
        for i in 0..n_examples {
            rowval.push(offset + 3 * i);
            nzval.push(1.0);
            colptr.push(rowval.len());
        }
        let mat = CscMatrix::new(
            offset + 3 * n_examples,
            2 * n_examples,
            colptr,
            rowval,
            nzval,
        );
        let mut rhs = build_rhs(self.nu, gamma, delta, n_examples, n_hypotheses);
        let mut cones = build_sns(n_examples, n_hypotheses);
        for _ in 0..n_examples {
            rhs.extend([0.0, 0.0, 1.0]);
            cones.push(ExponentialConeT());
        }
        let p = CscMatrix::zeros((2 * n_examples, 2 * n_examples));
        let q: Vec<_> = iter::repeat_n(0.0, n_examples)
            .chain(iter::repeat_n(1.0, n_examples))
            .collect();
        // Algorithm 2, step 3(b), Warmuth, Glocer and Raetsch (2007):
        // https://proceedings.neurips.cc/paper/2007/file/cfbce4c1d7c425baf21d6b6f2babe6be-Paper.pdf
        // Minimize entropy relative to uniform (the omitted ln(m) is constant).
        // An exponential-cone solve replaces SQP; convergence uses primal/dual
        // residuals and a duality gap, not objective changes at infeasible points.
        let settings = DefaultSettingsBuilder::default()
            .verbose(false)
            .max_iter(200)
            .tol_feas(1e-9)
            .tol_gap_abs(1e-9)
            .tol_gap_rel(1e-9)
            .build()
            .unwrap();
        let mut solver = DefaultSolver::new(&p, &q, &mat, &rhs, &cones, settings)
            .expect("failed to construct the SoftBoost entropy solver");
        solver.solve();
        if solver.solution.status == SolverStatus::PrimalInfeasible {
            return None;
        }
        assert_eq!(
            solver.solution.status,
            SolverStatus::Solved,
            "SoftBoost entropy solver did not converge"
        );
        assert!(
            solver.solution.obj_val.is_finite()
                && solver.solution.obj_val_dual.is_finite()
                && solver.solution.r_prim.is_finite()
                && solver.solution.r_dual.is_finite()
        );
        let solution = &solver.solution.x[..n_examples];
        const FEASIBILITY_TOLERANCE: f64 = 1e-7;
        assert!(solution.iter().all(|d| d.is_finite()
            && *d >= -FEASIBILITY_TOLERANCE
            && *d <= 1.0 / self.nu + FEASIBILITY_TOLERANCE));
        // Remove only negative roundoff; do not clamp positive near-zero mass
        // or renormalize. Recheck feasibility after this numerical correction.
        let solution: Vec<_> = solution.iter().map(|&d| d.max(0.0)).collect();
        assert!((solution.iter().sum::<f64>() - 1.0).abs() <= FEASIBILITY_TOLERANCE);
        for h in hypotheses {
            let edge = helpers::edge(self.sample, &solution, h);
            assert!(
                edge.is_finite() && edge <= gamma - delta + FEASIBILITY_TOLERANCE,
                "SoftBoost entropy solution violates an edge constraint"
            );
        }
        self.primal = solution;
        Some(())
    }

    pub fn distribution_on_examples(&self) -> Vec<f64> {
        self.primal.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};

    struct Margins([f64; 3]);
    impl Classifier for Margins {
        fn confidence(&self, sample: &Sample, row: usize) -> f64 {
            self.0[row] * sample.target()[row]
        }
    }

    fn sample() -> Sample {
        Sample::from_reader(
            BufReader::new(Cursor::new(b"x,class\n0,1\n1,-1\n2,1\n")),
            true,
        )
        .unwrap()
        .set_target("class")
    }

    #[test]
    fn entropy_projection_matches_three_point_solution() {
        let sample = sample();
        let mut solver = SoftBoostSolver::new(&sample);
        solver.initialize(1.0);
        assert!(
            solver
                .solve(0.0, 0.2, &[Margins([1.0, 0.0, -1.0])])
                .is_some()
        );
        let d = solver.distribution_on_examples();
        // KKT: d0*d2=d1^2, d2=d0+0.2, sum(d)=1.
        let a = (17.0 - 97.0_f64.sqrt()) / 30.0;
        let expected = [a, 0.8 - 2.0 * a, a + 0.2];
        for (actual, expected) in d.iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-5, "{d:?}");
        }
        assert!((d[0] * d[2] - d[1] * d[1]).abs() < 2e-5);
    }

    #[test]
    fn capped_projection_and_infeasibility() {
        let sample = sample();
        let mut solver = SoftBoostSolver::new(&sample);
        solver.initialize(2.0);
        let h = [Margins([1.0, 0.0, -1.0])];
        assert!(solver.solve(0.0, 0.4, &h).is_some());
        let d = solver.distribution_on_examples();
        for (actual, expected) in d.iter().zip([0.1, 0.4, 0.5]) {
            assert!((actual - expected).abs() < 2e-5, "{d:?}");
        }
        assert!(solver.solve(0.0, 0.6, &h).is_none());
        assert!(solver.distribution_on_examples().is_empty());
    }

    #[test]
    fn boundary_projection_has_finite_distribution() {
        let sample = sample();
        let mut solver = SoftBoostSolver::new(&sample);
        solver.initialize(1.0);
        assert!(
            solver
                .solve(0.0, 1.0, &[Margins([1.0, 0.0, -1.0])])
                .is_some()
        );
        let d = solver.distribution_on_examples();
        assert!(d[0].abs() < 1e-7 && d[1].abs() < 1e-7);
        assert!((d[2] - 1.0).abs() < 1e-7);
    }
}
