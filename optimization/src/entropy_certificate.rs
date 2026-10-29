//! A lower bound for the entropy-regularized restricted boosting objective.
//!
//! For a convex combination of hypothesis margin columns `g = A w`, weak
//! duality gives `min_d <g,d> + KL(d || uniform)/eta` as a lower bound on
//! `min_d max_j <A_j,d> + KL(d || uniform)/eta`. Both minima have
//! `sum(d)=1, 0 <= d_i <= 1/nu`. See Warmuth, Glocer and Vishwanathan (2008),
//! Sections 3–4: <https://mwarmuth.bitbucket.io/pubs/C83.pdf>.
//!
//! Evaluating a nearly normalized Gibbs distribution's *primal* objective is
//! not a lower-bound certificate. Instead, this module evaluates the scalar
//! equality-constraint dual. Its bound remains valid for any scalar, even if
//! the search stops before the best scalar is reached.

/// Returns a numerically padded dual lower bound for combined margins `A w`.
/// The caller must ensure that `w` is a nonnegative vector summing to one.
///
/// In exact arithmetic, for every real `r`,
/// ```text
/// L(r) = r + sum_i min_{0 <= d <= 1/nu}
///                    [d (g_i-r) + d ln(m d)/eta]
/// ```
/// is a lower bound. The scalar minimizers are
/// `d_i = min(1/nu, exp(eta*(r-g_i)-1)/m)`.
///
/// Bisection improves the bound by solving `sum_i d_i=1`; its accuracy is
/// unnecessary for weak duality. A conservative f64 evaluation allowance is
/// subtracted before return. This is a numerical certificate, not a formal
/// directed-rounding interval proof. Callers should also allow their explicit
/// numerical tolerance when comparing primal and dual values.
pub(crate) fn lower_bound(eta: f64, nu: f64, margins: &[f64]) -> f64 {
    assert!(
        eta.is_finite() && eta > 0.0,
        "eta must be finite and positive"
    );
    assert!(!margins.is_empty(), "entropy certificate needs margin rows");
    assert!(margins.iter().all(|g| g.is_finite()), "non-finite margin");
    let m = margins.len() as f64;
    assert!(
        nu.is_finite() && nu >= 1.0 && nu <= m,
        "invalid capping parameter"
    );
    let min_margin = margins.iter().copied().fold(f64::INFINITY, f64::min);
    let max_margin = margins.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // At full capping the feasible set contains only the uniform distribution.
    if nu == m {
        let mean = compensated_sum(margins.iter().map(|g| g / m));
        let allowance = evaluation_allowance(m, max_margin.abs().max(min_margin.abs()));
        return if mean.is_finite() && allowance.is_finite() {
            (mean - allowance).max(min_margin)
        } else {
            min_margin
        };
    }

    let cap = 1.0 / nu;
    let log_ratio = (m / nu).ln();
    let cap_threshold = (1.0 + log_ratio) / eta;
    let mut lower = min_margin;
    let mut upper = max_margin + cap_threshold;
    // An extreme eta can put the best multiplier outside f64 range. The
    // trivial min(g) bound remains valid since KL is nonnegative.
    if !upper.is_finite() || !cap_threshold.is_finite() {
        return min_margin;
    }
    let log_m = m.ln();
    let mass = |r: f64, g: f64| {
        let delta = r - g;
        if delta >= cap_threshold {
            cap
        } else {
            // This branch is below the cap, so exponentiation cannot overflow.
            (eta * delta - 1.0 - log_m).exp()
        }
    };
    let mut r = 0.5 * lower + 0.5 * upper;
    for _ in 0..100 {
        r = 0.5 * lower + 0.5 * upper;
        if r == lower || r == upper {
            break;
        }
        let total = compensated_sum(margins.iter().map(|&g| mass(r, g)));
        if total < 1.0 {
            lower = r;
        } else if total > 1.0 {
            upper = r;
        } else {
            break;
        }
    }
    let mut magnitude = r.abs();
    let terms = margins.iter().map(|&g| {
        let term = if r - g >= cap_threshold {
            cap * (g - r + log_ratio / eta)
        } else {
            -mass(r, g) / eta
        };
        magnitude += term.abs();
        term
    });
    let value = r + compensated_sum(terms);
    let allowance = evaluation_allowance(m, magnitude);
    if value.is_finite() && allowance.is_finite() {
        (value - allowance).max(min_margin)
    } else {
        min_margin
    }
}

fn evaluation_allowance(m: f64, magnitude: f64) -> f64 {
    64.0 * f64::EPSILON * (m + 1.0) * (1.0 + magnitude)
}

fn compensated_sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for value in values {
        let adjusted = value - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn objective(eta: f64, margins: &[f64], d: &[f64]) -> f64 {
        let m = margins.len() as f64;
        margins
            .iter()
            .zip(d)
            .map(|(g, d)| {
                g * d
                    + if *d == 0.0 {
                        0.0
                    } else {
                        d * (m * d).ln() / eta
                    }
            })
            .sum()
    }

    #[test]
    fn uncapped_bound_matches_closed_form_including_tiny_tails() {
        for eta in [0.5_f64, 2.0, 10.0, 100.0, 1000.0] {
            let exact = -1.0 - ((1.0 + (-eta).exp() + (-2.0 * eta).exp()) / 3.0_f64).ln() / eta;
            let actual = lower_bound(eta, 1.0, &[-1.0, 0.0, 1.0]);
            assert!(actual <= exact + 1e-14, "{eta}: {actual} > {exact}");
            assert!(exact - actual < 1e-11, "{eta}: {actual} vs {exact}");
        }
    }

    #[test]
    fn active_cap_matches_kkt_solution() {
        let eta = 2.0_f64;
        let middle = 0.5 / (1.0 + (-eta).exp());
        let exact = objective(eta, &[-1.0, 0.0, 1.0], &[0.5, middle, 0.5 - middle]);
        let actual = lower_bound(eta, 2.0, &[-1.0, 0.0, 1.0]);
        assert!(actual <= exact + 1e-14);
        assert!(exact - actual < 1e-11);
    }

    #[test]
    fn full_cap_and_common_shift() {
        let margins = [-0.9, 0.1, 0.5];
        let exact = margins.iter().sum::<f64>() / 3.0;
        let actual = lower_bound(10.0, 3.0, &margins);
        assert!(actual <= exact && exact - actual < 1e-11);
        let shifted = margins.map(|g| g + 0.25);
        for nu in [1.0, 2.0, 3.0] {
            assert!(
                (lower_bound(10.0, nu, &shifted) - lower_bound(10.0, nu, &margins) - 0.25).abs()
                    < 1e-11
            );
        }
    }

    #[test]
    fn lower_than_every_feasible_two_point_primal() {
        let margins = [-0.7, 0.4];
        for eta in [0.5, 2.0, 100.0] {
            for nu in [1.0, 1.5, 2.0] {
                let bound = lower_bound(eta, nu, &margins);
                let cap = 1.0 / nu;
                for k in 0..=1000 {
                    let a = 1.0 - cap + (2.0 * cap - 1.0) * k as f64 / 1000.0;
                    assert!(bound <= objective(eta, &margins, &[a, 1.0 - a]) + 1e-14);
                }
            }
        }
    }

    #[test]
    fn extreme_eta_has_a_finite_conservative_fallback() {
        for eta in [f64::from_bits(1), 1e300] {
            let bound = lower_bound(eta, 2.0, &[-1.0, 0.0, 1.0]);
            assert!(bound.is_finite() && (-1.0..=0.0).contains(&bound));
        }
    }

    #[test]
    fn reported_455_row_margin_histogram_has_a_tight_bound() {
        let mut margins = Vec::new();
        for (margin, count) in [
            (0.7853107344632764, 158),
            (0.8992805755395681, 264),
            (-0.7853107344632764, 19),
            (-0.8992805755395681, 14),
        ] {
            margins.extend(std::iter::repeat_n(margin, count));
        }
        let eta = 100.0_f64.ln() / 0.0005;
        // The uncapped Gibbs mass is 1/14 on the lowest-margin group;
        // other groups contribute less than f64's exponential underflow.
        // Since 1/14 < 1/4.55, the cap is inactive.
        let exact = -0.8992805755395681 + (455.0_f64 / 14.0).ln() / eta;
        let actual = lower_bound(eta, 4.55, &margins);
        assert!(actual <= exact + 1e-14);
        assert!(exact - actual < 1e-10, "{actual} vs {exact}");
    }
}
