//! The per-round problem of BrownBoost: step 3 of Figure 2 in Freund (2001).
//!
//! Notation of Section 4 of the paper, for one round with remaining time `s`:
//! `a_j = r_j + s` (old margin), `b_j = y_j h(x_j)` (step), and
//! `d_j = a_j + α b_j - t` (new margin). Equation (5) is the differential
//! equation `dt/dα = γ(α, t)` with `γ(α, t) = Σ_j w_j b_j`,
//! `w_j ∝ exp(-d_j² / c)`, started at `α = t = 0` and stopped where
//! `γ ≤ ν` or `t = s`.
//!
//! The proof of Theorem 2 shows that the solution keeps the potential
//! `Φ(α, t) = Σ_j erf(d_j / √c) - Σ_j erf(a_j / √c)` at zero, and
//! `∂Φ/∂t = -(2/√(πc)) Σ_j exp(-d_j² / c) < 0`, so for each `α` there is a
//! unique `t(α)` with `Φ(α, t(α)) = 0`. The solution curve of Equation (5)
//! is therefore `α ↦ (α, t(α))`. Instead of integrating the equation or
//! running the two-dimensional Newton–Raphson iteration of Section 7.2, this
//! module computes `t(α)` by safeguarded Newton steps in `t` and searches
//! `α` for the first point where `γ(α, t(α)) ≤ ν` or `t(α) ≥ s`.
use crate::core::tools::special::erf_diff;
#[cfg(test)]
use libm::{erf, erfc};

/// Which boundary condition of step 3 ended the round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Event {
    /// `γ(α, t) ≤ ν`: the weighted correlation dropped to the threshold.
    EdgeReached,
    /// `t = s`: the round used up the remaining time.
    ClockExpired,
}

/// The solution `(α_i, t_i)` of step 3.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Step {
    pub(super) alpha: f64,
    pub(super) time: f64,
    pub(super) event: Event,
}

/// Errors are reported as messages, as for SoftBoost's numerical stops.
pub(super) type SolveResult = Result<Step, &'static str>;

/// The data of one round of Equation (5).
pub(super) struct Round<'a> {
    // a_j = r_j + s
    a: &'a [f64],
    // b_j = y_j h(x_j)
    b: &'a [f64],
    c: f64,
    sqrt_c: f64,
}

impl<'a> Round<'a> {
    pub(super) fn new(a: &'a [f64], b: &'a [f64], c: f64) -> Self {
        assert_eq!(a.len(), b.len());
        assert!(c > 0.0 && c.is_finite());
        Self {
            a,
            b,
            c,
            sqrt_c: c.sqrt(),
        }
    }

    /// `Φ(α, t) = Σ_j [erf(d_j/√c) - erf(a_j/√c)]`.
    pub(super) fn potential_change(&self, alpha: f64, t: f64) -> f64 {
        self.a
            .iter()
            .zip(self.b)
            .map(|(a, b)| erf_diff((a + alpha * b - t) / self.sqrt_c, a / self.sqrt_c))
            .sum()
    }

    /// `∂Φ/∂t = -(2/√(πc)) Σ_j exp(-d_j²/c)`; it can underflow to `0`.
    fn potential_slope(&self, alpha: f64, t: f64) -> f64 {
        let total: f64 = self
            .a
            .iter()
            .zip(self.b)
            .map(|(a, b)| (-(a + alpha * b - t).powi(2) / self.c).exp())
            .sum();
        -2.0 / (std::f64::consts::PI * self.c).sqrt() * total
    }

    /// `γ(α, t) = Σ_j w_j b_j` with `w_j ∝ exp(-d_j²/c)`, computed with a
    /// shift by the largest exponent so that it is defined even when every
    /// unshifted weight underflows.
    pub(super) fn edge(&self, alpha: f64, t: f64) -> f64 {
        let logs: Vec<f64> = self
            .a
            .iter()
            .zip(self.b)
            .map(|(a, b)| -(a + alpha * b - t).powi(2) / self.c)
            .collect();
        let max = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let (num, den) = logs
            .iter()
            .zip(self.b)
            .fold((0.0, 0.0), |(num, den), (l, b)| {
                let w = (l - max).exp();
                (num + w * b, den + w)
            });
        num / den
    }

    /// Returns `t(α)`, the root of `t ↦ Φ(α, t)`, starting from `guess`.
    /// `Φ` decreases in `t` from `Σ_j (1 - erf(a_j/√c)) > 0` at `-∞` to
    /// `Σ_j (-1 - erf(a_j/√c)) < 0` at `+∞`.
    pub(super) fn time(&self, alpha: f64, guess: f64) -> Option<f64> {
        let f = |t: f64| self.potential_change(alpha, t);
        let value = f(guess);
        if value == 0.0 {
            return Some(guess);
        }
        // Bracket the root: f(lo) > 0 ≥ f(hi).
        let (mut lo, mut hi);
        let mut step = self.sqrt_c.max(1.0);
        if value > 0.0 {
            lo = guess;
            hi = guess + step;
            while f(hi) > 0.0 {
                lo = hi;
                step *= 2.0;
                hi = guess + step;
                if !hi.is_finite() {
                    return None;
                }
            }
        } else {
            hi = guess;
            lo = guess - step;
            while f(lo) <= 0.0 {
                hi = lo;
                step *= 2.0;
                lo = guess - step;
                if !lo.is_finite() {
                    return None;
                }
            }
        }

        // Newton steps, replaced by bisection when they leave the bracket.
        let mut t = 0.5 * (lo + hi);
        for _ in 0..200 {
            let value = f(t);
            if value == 0.0 {
                return Some(t);
            }
            if value > 0.0 {
                lo = t;
            } else {
                hi = t;
            }
            let slope = self.potential_slope(alpha, t);
            let newton = t - value / slope;
            let next = if slope < 0.0 && newton > lo && newton < hi {
                newton
            } else {
                0.5 * (lo + hi)
            };
            let scale = t.abs().max(1.0);
            if (next - t).abs() <= 4.0 * f64::EPSILON * scale
                || hi - lo <= 4.0 * f64::EPSILON * scale
            {
                return Some(next);
            }
            t = next;
        }
        None
    }

    /// Solves step 3 for remaining time `s > 0` and threshold `ν ≥ 0`.
    ///
    /// Requires `γ(0, 0) > ν` (otherwise the solution is `α = t = 0` and the
    /// round makes no progress). Grows `α` geometrically from `2^-10` until
    /// the stopping condition holds, then bisects between the last `α`
    /// where it fails and the first where it holds. If the condition holds,
    /// fails, and holds again within one growth step, the bisection may
    /// return the later crossing instead of the first one.
    pub(super) fn solve(&self, nu: f64, s: f64) -> SolveResult {
        assert!(s > 0.0, "BrownBoost needs positive remaining time");
        let edge = self.edge(0.0, 0.0);
        if !edge.is_finite() {
            return Err("the weighted correlation of the weak hypothesis is not finite");
        }
        if edge <= nu {
            return Err("the weak hypothesis has weighted correlation at most the edge threshold");
        }
        const NO_ROOT: &str = "solving Equation (5): no time t keeps the potential unchanged";
        let stops = |t: f64, gamma: f64| t >= s || gamma <= nu;

        // (α, t(α)) with the stopping condition false; starts at the origin.
        let mut below = (0.0, 0.0);
        let mut above = None;
        let mut alpha = 1.0 / 1024.0;
        for _ in 0..1100 {
            let t = self.time(alpha, below.1).ok_or(NO_ROOT)?;
            let gamma = self.edge(alpha, t);
            if !gamma.is_finite() {
                return Err("solving Equation (5): the weighted correlation is not finite");
            }
            if stops(t, gamma) {
                above = Some((alpha, t));
                break;
            }
            below = (alpha, t);
            alpha *= 2.0;
            if !alpha.is_finite() {
                break;
            }
        }
        let mut above =
            above.ok_or("solving Equation (5): the stopping condition is never reached")?;

        for _ in 0..200 {
            let mid = 0.5 * (below.0 + above.0);
            if mid <= below.0 || mid >= above.0 {
                break;
            }
            let t = self.time(mid, below.1).ok_or(NO_ROOT)?;
            let gamma = self.edge(mid, t);
            if stops(t, gamma) {
                above = (mid, t);
            } else {
                below = (mid, t);
            }
        }

        let (alpha, t) = above;
        Ok(if t >= s {
            Step {
                alpha,
                time: s,
                event: Event::ClockExpired,
            }
        } else {
            Step {
                alpha,
                time: t,
                event: Event::EdgeReached,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tools::special::erfc_inverse;

    #[test]
    fn erf_diff_is_accurate_in_the_tails() {
        // erf(6) and erf(7) both round to 1, so a plain difference is 0.
        // Reference: CPython's math.erfc(6) - math.erfc(7), an independent
        // implementation, is 2.1519694874242835e-17.
        assert_eq!(erf(7.0) - erf(6.0), 0.0);
        let d = erf_diff(7.0, 6.0);
        assert!((d / 2.151_969_487_424_283_5e-17 - 1.0).abs() < 1e-14, "{d}");
        assert_eq!(erf_diff(-7.0, -6.0), -d);
        // Mixed signs: plain difference.
        assert_eq!(erf_diff(0.5, -0.5), erf(0.5) - erf(-0.5));
    }

    #[test]
    fn erfc_inverse_inverts_erfc() {
        for target in [0.9, 0.5, 0.1, 1e-3, 1e-10, 1e-200] {
            let x = erfc_inverse(target);
            assert!((erfc(x) / target - 1.0).abs() < 1e-12, "{target}");
        }
        // erfc(1.1630871536766743) = 0.1 (to the printed digits).
        assert!((erfc_inverse(0.1) - 1.163_087_153_676_674_3).abs() < 1e-14);
    }

    #[test]
    fn time_keeps_the_potential_and_is_increasing_while_the_edge_is_positive() {
        let a = [1.0, 0.5, -0.2, 2.0, 1.5];
        let b = [1.0, -1.0, 1.0, 1.0, -0.5];
        let round = Round::new(&a, &b, 2.0);
        let mut previous = 0.0;
        for alpha in [0.0, 0.05, 0.1, 0.2] {
            let t = round.time(alpha, previous).unwrap();
            assert!(round.potential_change(alpha, t).abs() < 1e-14);
            if alpha > 0.0 {
                // dt/dα = γ > 0 on this range.
                assert!(round.edge(alpha, t) > 0.0 && t > previous);
            }
            previous = t;
        }
        assert_eq!(round.time(0.0, 0.0), Some(0.0));
    }

    #[test]
    fn solve_meets_one_boundary_condition_and_keeps_the_potential() {
        let a = [1.0, 0.5, -0.2, 2.0, 1.5, 0.3];
        let b = [1.0, -1.0, 1.0, 1.0, -0.5, 0.8];
        let round = Round::new(&a, &b, 2.0);
        // Plenty of remaining time: the round ends at γ = ν.
        let step = round.solve(0.01, 10.0).unwrap();
        assert_eq!(step.event, Event::EdgeReached);
        assert!(step.alpha > 0.0 && step.time > 0.0 && step.time < 10.0);
        let gamma = round.edge(step.alpha, step.time);
        assert!((gamma - 0.01).abs() < 1e-9, "{gamma}");
        assert!(round.potential_change(step.alpha, step.time).abs() < 1e-12);

        // Little remaining time: the round ends at t = s, with γ > ν there.
        let s = 0.5 * step.time;
        let short = round.solve(0.01, s).unwrap();
        assert_eq!(short.event, Event::ClockExpired);
        assert_eq!(short.time, s);
        assert!(short.alpha < step.alpha);
        assert!(round.edge(short.alpha, s) > 0.01);
        assert!(round.potential_change(short.alpha, s).abs() < 1e-12);
    }

    #[test]
    fn a_perfect_hypothesis_shifts_every_margin_until_the_clock_expires() {
        // b_j = 1: Φ(α, t) = 0 forces t = α, so γ stays 1 and t reaches s.
        let a = [0.5, 1.0, 1.5];
        let b = [1.0; 3];
        let step = Round::new(&a, &b, 1.0).solve(0.01, 0.7).unwrap();
        assert_eq!(step.event, Event::ClockExpired);
        assert!((step.alpha - 0.7).abs() < 1e-12);
    }

    #[test]
    fn a_weak_or_negative_edge_is_rejected() {
        let a = [1.0, 1.0];
        let b = [1.0, -1.0];
        assert!(Round::new(&a, &b, 1.0).solve(0.0, 1.0).is_err());
        let b = [-1.0, -1.0];
        assert!(Round::new(&a, &b, 1.0).solve(0.0, 1.0).is_err());
    }
}
