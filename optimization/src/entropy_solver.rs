//! The restricted ERLPBoost problem, solved without a quadratic entropy surrogate.
use clarabel::{algebra::*, solver::*};
use miniboosts_core::helpers;

pub(crate) struct EntropySolution {
    pub value: f64,
    pub lower_bound: f64,
    pub distribution: Vec<f64>,
    pub weights: Vec<f64>,
}

/// Warmuth, Glocer & Vishwanathan (2008), Algorithm 1 / restricted problem:
/// <https://mwarmuth.bitbucket.io/pubs/C83.pdf>.
/// Minimize gamma + KL(d || uniform)/eta under the existing linear constraints.
/// The slack (-t_i, d_i, 1/m) in K_exp means t_i >= d_i ln(m d_i).
/// This is the original entropy objective, including d_i=0 by cone closure;
/// no log perturbation, SQP model or descent line search is required.
pub(crate) fn solve(
    eta: f64,
    nu: f64,
    linear: CscMatrix<f64>,
    mut rhs: Vec<f64>,
    mut cones: Vec<SupportedConeT<f64>>,
    m: usize,
    n: usize,
) -> EntropySolution {
    let edge_start = 1 + 2 * m;
    // Retain the margin matrix for validation and a separate entropy dual bound.
    let mut margins = vec![vec![0.0; m]; n];
    for i in 0..m {
        for k in linear.colptr[i + 1]..linear.colptr[i + 2] {
            let row = linear.rowval[k];
            if row >= edge_start {
                margins[row - edge_start][i] = linear.nzval[k];
            }
        }
    }
    if nu == m as f64 {
        let distribution = vec![1.0 / m as f64; m];
        let (index, value) = margins
            .iter()
            .enumerate()
            .map(|(j, a)| (j, dot(a, &distribution)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        let mut weights = vec![0.0; n];
        weights[index] = 1.0;
        return EntropySolution {
            value,
            lower_bound: crate::entropy_certificate::lower_bound(eta, nu, &margins[index]),
            distribution,
            weights,
        };
    }
    let offset = linear.m;
    let variables = 1 + 2 * m;
    let mut colptr = vec![0];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    for col in 0..=m {
        for k in linear.colptr[col]..linear.colptr[col + 1] {
            rowval.push(linear.rowval[k]);
            nzval.push(linear.nzval[k]);
        }
        if col > 0 {
            rowval.push(offset + 3 * (col - 1) + 1);
            nzval.push(-1.0);
        }
        colptr.push(rowval.len());
    }
    for i in 0..m {
        rowval.push(offset + 3 * i);
        nzval.push(1.0);
        colptr.push(rowval.len());
        rhs.extend([0.0, 0.0, 1.0 / m as f64]);
        cones.push(ExponentialConeT());
    }
    let matrix = CscMatrix::new(offset + 3 * m, variables, colptr, rowval, nzval);
    let p = CscMatrix::zeros((variables, variables));
    let mut q = vec![0.0; variables];
    q[0] = 1.0;
    q[m + 1..].fill(1.0 / eta);
    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .max_iter(200)
        .tol_feas(1e-10)
        .tol_gap_abs(1e-10)
        .tol_gap_rel(1e-10)
        .build()
        .unwrap();
    let mut solver = DefaultSolver::new(&p, &q, &matrix, &rhs, &cones, settings)
        .expect("failed to construct ERLPBoost entropy cone problem");
    solver.solve();
    let sol = &solver.solution;
    assert!(
        matches!(
            sol.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ),
        "ERLPBoost entropy solver did not solve: {:?}",
        sol.status
    );
    const FEAS: f64 = 1e-7;
    assert!(
        sol.x.iter().chain(&sol.z).all(|v| v.is_finite())
            && sol.obj_val.is_finite()
            && sol.obj_val_dual.is_finite()
            && sol.r_prim.is_finite()
            && sol.r_dual.is_finite()
            && sol.r_prim <= FEAS
            && sol.r_dual <= FEAS,
        "invalid ERLPBoost entropy solution or residuals"
    );
    let raw = &sol.x[1..=m];
    assert!(
        raw.iter().all(|d| *d >= -FEAS && *d <= 1.0 / nu + FEAS)
            && (raw.iter().sum::<f64>() - 1.0).abs() <= FEAS,
        "ERLPBoost entropy distribution violates constraints"
    );
    // Repair only numerical simplex error, then check the ORIGINAL objective
    // against an independent dual bound. Solver objective changes are not a
    // convergence certificate for a clipped/normalized solution.
    let distribution = repair_distribution(raw, nu);
    let mut weights: Vec<_> = sol.z[edge_start..edge_start + n]
        .iter()
        .map(|w| {
            assert!(*w >= -FEAS, "negative ERLPBoost hypothesis weight");
            w.max(0.0)
        })
        .collect();
    let total: f64 = weights.iter().sum();
    assert!(
        (total - 1.0).abs() <= FEAS,
        "ERLPBoost weights do not sum to one"
    );
    weights.iter_mut().for_each(|w| *w /= total);
    let combined: Vec<_> = (0..m)
        .map(|i| weights.iter().zip(&margins).map(|(w, a)| w * a[i]).sum())
        .collect();
    let lower_bound = crate::entropy_certificate::lower_bound(eta, nu, &combined);
    let value = margins
        .iter()
        .map(|a| dot(a, &distribution))
        .fold(f64::NEG_INFINITY, f64::max)
        + helpers::entropy_from_uni_distribution(&distribution) / eta;
    let gap = value - lower_bound;
    assert!(
        value.is_finite() && lower_bound.is_finite() && gap >= -1e-10 && gap <= 1e-7,
        "ERLPBoost entropy primal/dual gap is not acceptable: {gap}"
    );
    EntropySolution {
        value,
        lower_bound,
        distribution,
        weights,
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

// A bounded simplex correction must be able to revive clipped negative
// roundoff: keeping the old zero support can make a fractional cap infeasible.
fn repair_distribution(raw: &[f64], nu: f64) -> Vec<f64> {
    let cap = 1.0 / nu;
    let mut d: Vec<_> = raw.iter().map(|x| x.clamp(0.0, cap)).collect();
    let sum: f64 = d.iter().sum();
    if sum > 1.0 {
        d.iter_mut().for_each(|x| *x /= sum);
    } else if sum < 1.0 {
        let slack: f64 = d.iter().map(|x| cap - x).sum();
        if slack > 0.0 {
            let fraction = ((1.0 - sum) / slack).min(1.0);
            d.iter_mut().for_each(|x| *x += fraction * (cap - *x));
        }
    }
    assert!(
        d.iter()
            .all(|x| x.is_finite() && *x >= 0.0 && *x <= cap + 1e-15)
    );
    assert!((d.iter().sum::<f64>() - 1.0).abs() <= 1e-12);
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repair_revives_mass_for_fractional_caps() {
        let nu = 2.000000001;
        let d = repair_distribution(&[0.5, 0.5, -1e-10], nu);
        assert!(d[2] > 0.0);
        assert!(d.iter().all(|x| *x <= 1.0 / nu));
        assert!((d.iter().sum::<f64>() - 1.0).abs() <= 1e-15);
    }
}
