//! The potential, weight, and per-round problem of RobustBoost (Freund, 2009).
//!
//! With time `t ∈ [0, 1]` and parameters `θ` (goal margin), `σ_f` (final
//! width), and `ρ`, Equations (5) and (6) of the paper are
//! ```txt
//! σ²(t) = (σ_f² + 1) e^{2(1-t)} - 1,     μ(t) = (θ - 2ρ) e^{1-t} + 2ρ.
//! ```
//! The paper defines the potential (4) as `Φ(m, t) = 1 - erf((m - μ)/σ)`
//! with `erf(a) = (1/√π) ∫_{-∞}^a e^{-x²} dx`, and the weight (7) as
//! `w(m, t) = exp(-(m - μ)² / (2σ²))`. These are not consistent: the weight
//! should be proportional to `-∂Φ/∂m`, which for that `erf` is
//! `exp(-(m - μ)²/σ²)`. This module reads `erf` as the standard normal
//! distribution function `N`, which makes the two consistent:
//! ```txt
//! Φ(m, t) = 1 - N((m - μ(t)) / σ(t)) = (1/2) erfc((m - μ(t)) / (σ(t) √2)),
//! w(m, t) = exp(-(m - μ(t))² / (2σ(t)²)) ∝ -∂Φ/∂m.
//! ```
use crate::core::tools::special::{erf_diff, erfc_inverse};
use std::f64::consts::{E, SQRT_2};

/// The parameters `θ`, `σ_f`, and `ρ` of the potential.
#[derive(Clone, Copy, Debug)]
pub(super) struct Potential {
    pub(super) theta: f64,
    pub(super) sigma_f: f64,
    pub(super) rho: f64,
}

impl Potential {
    /// Sets `ρ` from the error goal `ε` by Equation (8),
    /// `ε = Φ(0, 0)`, i.e. `-μ(0) / σ(0) = N⁻¹(1 - ε)`.
    pub(super) fn new(epsilon: f64, theta: f64, sigma_f: f64) -> Self {
        let sigma0 = ((sigma_f * sigma_f + 1.0) * E * E - 1.0).sqrt();
        let z = normal_quantile(1.0 - epsilon);
        let rho = (theta * E + z * sigma0) / (2.0 * (E - 1.0));
        Self {
            theta,
            sigma_f,
            rho,
        }
    }

    pub(super) fn sigma(&self, t: f64) -> f64 {
        ((self.sigma_f * self.sigma_f + 1.0) * (2.0 * (1.0 - t)).exp() - 1.0).sqrt()
    }

    pub(super) fn mu(&self, t: f64) -> f64 {
        (self.theta - 2.0 * self.rho) * (1.0 - t).exp() + 2.0 * self.rho
    }

    /// `z = (m - μ(t)) / σ(t)`.
    fn z(&self, m: f64, t: f64) -> f64 {
        (m - self.mu(t)) / self.sigma(t)
    }

    /// `Φ(m, t) = (1/2) erfc(z / √2)`.
    #[cfg(test)]
    pub(super) fn value(&self, m: f64, t: f64) -> f64 {
        0.5 * libm::erfc(self.z(m, t) / SQRT_2)
    }

    /// `Φ(m', t') - Φ(m, t)`, accurate when both are in the same tail.
    fn difference(&self, m_new: f64, t_new: f64, m: f64, t: f64) -> f64 {
        -0.5 * erf_diff(self.z(m_new, t_new) / SQRT_2, self.z(m, t) / SQRT_2)
    }

    /// `log w(m, t) = -z² / 2`.
    pub(super) fn log_weight(&self, m: f64, t: f64) -> f64 {
        -0.5 * self.z(m, t).powi(2)
    }
}

/// `N⁻¹(p)` for `p ∈ (0, 1)`, from `N(z) = (1/2) erfc(-z/√2)`.
fn normal_quantile(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0);
    if p >= 0.5 {
        // erfc(z/√2) = 2(1 - p) ≤ 1.
        if p == 0.5 {
            0.0
        } else {
            SQRT_2 * erfc_inverse(2.0 * (1.0 - p))
        }
    } else {
        -SQRT_2 * erfc_inverse(2.0 * p)
    }
}

/// Which condition ended the round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Event {
    /// The weighted correlation `Σ_j y_j h(x_j) w(m'_j, t + Δt)` reached `0`.
    EdgeReached,
    /// `Δt = 1 - t`: the round used up the remaining time.
    ClockExpired,
}

/// The solution `(Δm, Δt)` of one round.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Step {
    pub(super) delta_m: f64,
    pub(super) delta_t: f64,
    pub(super) event: Event,
}

/// The data of one round: margins `m_j`, steps `b_j = y_j h(x_j)`, and the
/// current time `t`.
pub(super) struct Round<'a> {
    potential: Potential,
    margins: &'a [f64],
    steps: &'a [f64],
    t: f64,
}

impl<'a> Round<'a> {
    pub(super) fn new(potential: Potential, margins: &'a [f64], steps: &'a [f64], t: f64) -> Self {
        assert_eq!(margins.len(), steps.len());
        assert!((0.0..1.0).contains(&t));
        Self {
            potential,
            margins,
            steps,
            t,
        }
    }

    /// `m'_j = m_j e^{-Δt} + b_j Δm`.
    fn new_margin(&self, j: usize, delta_m: f64, delta_t: f64) -> f64 {
        self.margins[j] * (-delta_t).exp() + self.steps[j] * delta_m
    }

    /// `P(Δm, Δt) = Σ_j Φ(m'_j, t + Δt) - Σ_j Φ(m_j, t)`.
    pub(super) fn potential_change(&self, delta_m: f64, delta_t: f64) -> f64 {
        (0..self.margins.len())
            .map(|j| {
                self.potential.difference(
                    self.new_margin(j, delta_m, delta_t),
                    self.t + delta_t,
                    self.margins[j],
                    self.t,
                )
            })
            .sum()
    }

    /// The normalized correlation `Σ_j w_j b_j / Σ_j w_j` at
    /// `(m'_j, t + Δt)`, computed with a log-space shift.
    pub(super) fn edge(&self, delta_m: f64, delta_t: f64) -> f64 {
        let logs: Vec<f64> = (0..self.margins.len())
            .map(|j| {
                self.potential
                    .log_weight(self.new_margin(j, delta_m, delta_t), self.t + delta_t)
            })
            .collect();
        let max = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let (num, den) = logs
            .iter()
            .zip(self.steps)
            .fold((0.0, 0.0), |(num, den), (l, b)| {
                let w = (l - max).exp();
                (num + w * b, den + w)
            });
        num / den
    }

    /// For a given `Δm`, returns `Some(Δt)` with `P(Δm, Δt) = 0` and
    /// `Δt ∈ [lo, 1 - t]`, or `None` if `P(Δm, 1 - t) < 0` (the root lies
    /// beyond the remaining time). Requires `P(Δm, lo) ≤ 0`. Bisection with
    /// the bracket kept by sign; `P` is assumed to increase in `Δt` on the
    /// bracket, so the root found is the one the solution curve reaches.
    fn time(&self, delta_m: f64, lo: f64) -> Option<f64> {
        let end = 1.0 - self.t;
        if self.potential_change(delta_m, end) < 0.0 {
            return None;
        }
        let (mut lo, mut hi) = (lo, end);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if mid <= lo || mid >= hi {
                break;
            }
            if self.potential_change(delta_m, mid) < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(0.5 * (lo + hi))
    }

    /// Solves the round: the first `(Δm, Δt)` along the potential-preserving
    /// curve from the origin where the correlation reaches `0` or
    /// `Δt = 1 - t`. Grows `Δm` geometrically from `2^-10`, then bisects.
    pub(super) fn solve(&self) -> Result<Step, &'static str> {
        let edge = self.edge(0.0, 0.0);
        if !edge.is_finite() {
            return Err("the weighted correlation of the weak hypothesis is not finite");
        }
        if edge <= 0.0 {
            return Err("the weak hypothesis has nonpositive weighted correlation");
        }
        // The predicate "the round has ended by Δm": Some(Δt) with the edge
        // still positive means it has not.
        let state = |delta_m: f64, lo: f64| -> Result<Option<f64>, &'static str> {
            match self.time(delta_m, lo) {
                None => Ok(None),
                Some(dt) => {
                    let gamma = self.edge(delta_m, dt);
                    if !gamma.is_finite() {
                        return Err("solving the round: the weighted correlation is not finite");
                    }
                    Ok(if gamma > 0.0 { Some(dt) } else { None })
                }
            }
        };

        let mut below = (0.0, 0.0);
        let mut above = None;
        let mut delta_m = 1.0 / 1024.0;
        for _ in 0..1100 {
            match state(delta_m, below.1)? {
                Some(dt) => below = (delta_m, dt),
                None => {
                    above = Some(delta_m);
                    break;
                }
            }
            delta_m *= 2.0;
            if !delta_m.is_finite() {
                break;
            }
        }
        let mut above =
            above.ok_or("solving the round: the stopping condition is never reached")?;
        for _ in 0..200 {
            let mid = 0.5 * (below.0 + above);
            if mid <= below.0 || mid >= above {
                break;
            }
            match state(mid, below.1)? {
                Some(dt) => below = (mid, dt),
                None => above = mid,
            }
        }

        // Classify the event at the crossing.
        let end = 1.0 - self.t;
        let delta_m = above;
        Ok(match self.time(delta_m, below.1) {
            None => Step {
                delta_m,
                delta_t: end,
                event: Event::ClockExpired,
            },
            Some(dt) if dt >= end => Step {
                delta_m,
                delta_t: end,
                event: Event::ClockExpired,
            },
            Some(dt) => Step {
                delta_m,
                delta_t: dt,
                event: Event::EdgeReached,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rho_satisfies_equation_eight() {
        for (epsilon, theta) in [(0.1, 0.0), (0.2, 0.5), (0.05, 1.0)] {
            let p = Potential::new(epsilon, theta, 0.1);
            assert!(
                (p.value(0.0, 0.0) - epsilon).abs() < 1e-12,
                "{epsilon} {theta}"
            );
        }
        // At t = 1: μ = θ and σ = σ_f, so Φ(θ, 1) = 1/2.
        let p = Potential::new(0.1, 0.3, 0.1);
        assert!((p.mu(1.0) - 0.3).abs() < 1e-15 && (p.sigma(1.0) - 0.1).abs() < 1e-15);
        assert!((p.value(0.3, 1.0) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn weight_is_proportional_to_minus_the_potential_slope() {
        // -∂Φ/∂m = φ(z)/σ = w / (σ √(2π)).
        let p = Potential::new(0.1, 0.2, 0.1);
        let (m, t, h) = (0.4, 0.3, 1e-6);
        let slope = (p.value(m + h, t) - p.value(m - h, t)) / (2.0 * h);
        let w = p.log_weight(m, t).exp();
        let expected = w / (p.sigma(t) * (2.0 * std::f64::consts::PI).sqrt());
        assert!((-slope - expected).abs() < 1e-8, "{slope} vs {expected}");
    }

    #[test]
    fn normal_quantile_inverts_the_distribution_function() {
        for q in [0.01, 0.3, 0.5, 0.8, 0.999] {
            let z = normal_quantile(q);
            assert!((1.0 - 0.5 * libm::erfc(z / SQRT_2) - q).abs() < 1e-12);
        }
    }

    #[test]
    fn solve_keeps_the_potential_and_meets_a_boundary_condition() {
        let p = Potential::new(0.1, 0.1, 0.1);
        let margins = [0.0; 6];
        let steps = [1.0, 1.0, -1.0, 1.0, 0.5, -0.5];
        let round = Round::new(p, &margins, &steps, 0.0);
        let step = round.solve().unwrap();
        assert!(step.delta_m > 0.0 && step.delta_t > 0.0);
        assert!(round.potential_change(step.delta_m, step.delta_t).abs() < 1e-10);
        match step.event {
            Event::EdgeReached => assert!(round.edge(step.delta_m, step.delta_t).abs() < 1e-6),
            Event::ClockExpired => assert_eq!(step.delta_t, 1.0),
        }
    }
}
