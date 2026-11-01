//! Objective functions evaluated by [`FrankWolfe`](super::FrankWolfe) and
//! by the soft-margin loggers.
//!
//! The soft-margin objectives take a point `θ ∈ R^m` in *negated-margin*
//! coordinates, `θ_i = -y_i f(x_i)`, where `f = Σ_j w_j h_j` is a combined
//! hypothesis. Throughout, `Δ_{m,ν}` is the capped simplex
//! `{ d : Σ_i d_i = 1, 0 ≤ d_i ≤ 1/ν }` and
//! `KL(d ‖ u) = Σ_i d_i ln(m d_i)` is the relative entropy from the uniform
//! distribution.
use crate::core::tools::helpers;

/// A differentiable objective used by [`FrankWolfe`](super::FrankWolfe).
///
/// Implementations define the value, the gradient, and a smoothness
/// constant. [`FrankWolfe`](super::FrankWolfe) only calls
/// [`gradient`](Self::gradient) and [`smooth`](Self::smooth); none of the
/// declared properties are verified at runtime.
///
/// # Examples
/// A custom objective `f(x) = ‖x - c‖² / 2`. Its gradient `x - c` is
/// 1-Lipschitz in the ℓ2-norm, so `smooth` returns `(1, 2)`.
/// ```
/// use miniboosts::ObjectiveFunction;
///
/// struct SquaredDistance(Vec<f64>);
///
/// impl ObjectiveFunction for SquaredDistance {
///     fn name(&self) -> &str {
///         "Squared distance"
///     }
///
///     fn smooth(&self) -> (f64, f64) {
///         (1.0, 2.0)
///     }
///
///     fn objective_value(&self, x: &[f64]) -> f64 {
///         x.iter().zip(&self.0).map(|(a, c)| (a - c).powi(2)).sum::<f64>() / 2.0
///     }
///
///     fn gradient(&self, x: &[f64]) -> Vec<f64> {
///         x.iter().zip(&self.0).map(|(a, c)| a - c).collect()
///     }
/// }
///
/// let f = SquaredDistance(vec![0.2, 0.3, 0.5]);
/// let x = [1.0, 0.0, 0.0];
/// // f(x) = (0.8² + 0.3² + 0.5²) / 2 = 0.49 and ∇f(x) = x - c.
/// assert!((f.objective_value(&x) - 0.49).abs() < 1e-12);
/// let grad = f.gradient(&x);
/// for (g, expected) in grad.iter().zip([0.8, -0.3, -0.5]) {
///     assert!((g - expected).abs() < 1e-12);
/// }
/// ```
pub trait ObjectiveFunction {
    /// A human-readable name of the objective.
    fn name(&self) -> &str;

    /// Returns the smoothness constant and norm.
    /// We say that a function `f` is L-smooth w.r.t. l_p-norm if
    /// ```txt
    /// f(y) ≤ f(x) + ∇f(x)∙(y - x) + (L/2) ‖y - x‖_p^2
    /// ```
    /// This method returns the pair `(L, p)`. `p == f64::MAX` denotes the
    /// ℓ∞-norm. Only [`FwUpdateRule::ShortStep`](super::FwUpdateRule::ShortStep)
    /// uses this value.
    fn smooth(&self) -> (f64, f64);

    /// Objective value at `point`.
    fn objective_value(&self, point: &[f64]) -> f64;

    /// Gradient vector at `point`; it must have the same length as `point`.
    fn gradient(&self, point: &[f64]) -> Vec<f64>;
}

/// The (non-smooth) soft-margin objective in negated-margin coordinates.
///
/// For `θ_i = -y_i f(x_i)`,
/// ```txt
/// objective_value(θ) = min_{d ∈ Δ_{m,ν}} Σ_i d_i y_i f(x_i) = -max_{d ∈ Δ_{m,ν}} ⟨θ, d⟩,
/// gradient(θ)        = argmax_{d ∈ Δ_{m,ν}} ⟨θ, d⟩,
/// ```
/// i.e., the value is the average of the `ν` smallest margins (with a
/// fractional weight when `ν` is not an integer). By LP duality this equals
/// `max_{ρ, ξ ≥ 0} ρ - (1/ν) Σ_i ξ_i` subject to `y_i f(x_i) ≥ ρ - ξ_i`, the
/// soft-margin objective for a fixed `f`.
///
/// The maximizer is computed greedily by sorting `θ` in decreasing order and
/// assigning mass `1/ν` until the total reaches one (`O(m log m)` time).
/// Ties are broken by the sort order, so `gradient` returns one element of the
/// subdifferential.
///
/// Note that [`objective_value`](ObjectiveFunction::objective_value) returns
/// `-max ⟨θ, d⟩` while [`gradient`](ObjectiveFunction::gradient) returns the
/// maximizer `d`, which is the gradient of `+max ⟨θ, d⟩`; the signs of the two
/// methods are not consistent with each other.
///
/// The function is not smooth; [`smooth`](ObjectiveFunction::smooth) returns
/// the placeholder `(0, 0)` and must not be used with
/// [`FwUpdateRule::ShortStep`](super::FwUpdateRule::ShortStep).
///
/// # Panics
/// `gradient` and `objective_value` panic if `θ` contains NaN. The capping
/// parameter is not validated; the result lies in `Δ_{m,ν}` only when
/// `1 ≤ ν ≤ m`.
///
/// # Examples
/// Margins `y_i f(x_i) = (1, 0.2, -0.4, 2)` and `ν = 1.5`, so each weight is
/// capped at `1/ν = 2/3`. The minimizing distribution puts `2/3` on the
/// smallest margin `-0.4` and the remaining `1/3` on the next one, `0.2`:
/// ```
/// use miniboosts::{ObjectiveFunction, SoftMarginObjective};
///
/// let margins = [1.0, 0.2, -0.4, 2.0];
/// // The objective takes negated margins θ_i = -y_i f(x_i).
/// let theta: Vec<f64> = margins.iter().map(|m| -m).collect();
///
/// let objective = SoftMarginObjective::new(1.5);
/// // (2/3)(-0.4) + (1/3)(0.2) = -0.2
/// let value = objective.objective_value(&theta);
/// assert!((value - (-0.2)).abs() < 1e-12);
///
/// // `gradient` returns the minimizing distribution d (not ∇ of `value`).
/// let d = objective.gradient(&theta);
/// for (di, expected) in d.iter().zip([0.0, 1.0 / 3.0, 2.0 / 3.0, 0.0]) {
///     assert!((di - expected).abs() < 1e-12);
/// }
/// ```
pub struct SoftMarginObjective(f64);

impl SoftMarginObjective {
    /// Creates the objective with capping parameter `capping = ν`.
    /// The intended range is `1 ≤ ν ≤ m`; it is not checked here.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{ObjectiveFunction, SoftMarginObjective};
    ///
    /// // With ν = 1 the value is the smallest margin (hard margin).
    /// let objective = SoftMarginObjective::new(1.0);
    /// assert_eq!(objective.objective_value(&[-0.5, 0.25, -1.0]), -0.25);
    /// // The objective is not smooth; `(0, 0)` is a placeholder.
    /// assert_eq!(objective.smooth(), (0.0, 0.0));
    /// ```
    pub fn new(capping: f64) -> Self {
        Self(capping)
    }
}

impl ObjectiveFunction for SoftMarginObjective {
    fn name(&self) -> &str {
        "Soft Margin Objective"
    }

    fn smooth(&self) -> (f64, f64) {
        (0f64, 0f64)
    }

    /// Returns `min_{d ∈ Δ_{m,ν}} Σ_i d_i y_i f(x_i) = -f*(θ)` at
    /// `θ = -Aw`, where `f*(θ) = max_{d ∈ Δ_{m,ν}} ⟨θ, d⟩`.
    fn objective_value(&self, point: &[f64]) -> f64 {
        let dist = self.gradient(point);
        -helpers::inner_product(point, &dist[..])
    }

    /// Returns a maximizer `d ∈ Δ_{m,ν}` of `⟨θ, d⟩` at `θ = -Aw`, i.e., a
    /// subgradient of `f*(θ)`.
    fn gradient(&self, point: &[f64]) -> Vec<f64> {
        let m = point.len();
        let indices = {
            let mut indices = (0..m).collect::<Vec<_>>();
            indices.sort_by(|&i, &j| point[j].partial_cmp(&point[i]).unwrap());
            indices
        };
        let mut grad = vec![0f64; m];

        let cap = 1f64 / self.0;
        let mut rest = 1f64;
        for i in indices {
            if rest > cap {
                grad[i] = cap;
                rest -= cap;
            } else {
                grad[i] = rest;
                break;
            }
        }
        grad
    }
}

/// The entropy-regularized soft-margin objective of ERLPBoost, in
/// negated-margin coordinates.
///
/// For `θ_i = -y_i f(x_i)`, let
/// ```txt
/// d*(θ) = argmin_{d ∈ Δ_{m,ν}} Σ_i d_i y_i f(x_i) + KL(d ‖ u)/η
///       = KL projection onto Δ_{m,ν} of  d_i ∝ exp(-η y_i f(x_i)).
/// ```
/// Then
/// ```txt
/// objective_value(θ) = Σ_i d*_i y_i f(x_i) + KL(d* ‖ u)/η   (= -f*(θ)),
/// gradient(θ)        = d*(θ)                                (= ∇f*(θ)),
/// ```
/// where `f*(θ) = max_{d ∈ Δ_{m,ν}} ⟨θ, d⟩ - KL(d ‖ u)/η`. The value is the
/// smoothed soft margin that ERLPBoost maximizes. As with
/// [`SoftMarginObjective`], the value and the gradient have opposite sign
/// conventions; the boosters wrap this type in a minimization objective that
/// returns `f*(θ)`.
///
/// [`smooth`](ObjectiveFunction::smooth) returns `(η, ∞)`: `f*` is
/// `η`-smooth w.r.t. the ℓ∞-norm because `KL(· ‖ u)/η` is `(1/η)`-strongly
/// convex w.r.t. the ℓ1-norm.
///
/// The projection uses the sorted active-set method (`O(m log m)` time) in
/// log space.
///
/// # Panics
/// Panics if `θ` yields invalid log weights (NaN or `+∞`) or if `ν` is
/// non-finite or outside `[1, m]` (checked when the projection runs, not in
/// [`new`](Self::new)).
///
/// # Examples
/// With `ν = 1` no cap is active, `d*` is the softmax of `-η y_i f(x_i)`,
/// and the value has the closed form
/// `-(1/η) ln((1/m) Σ_i exp(-η y_i f(x_i)))`:
/// ```
/// use miniboosts::{ErlpSoftMarginObjective, ObjectiveFunction};
///
/// let margins = [1.0, 0.2, -0.4, 2.0];
/// let theta: Vec<f64> = margins.iter().map(|m| -m).collect();
/// let eta = 2.0;
///
/// let objective = ErlpSoftMarginObjective::new(1.0, eta);
/// let exps: Vec<f64> = margins.iter().map(|m| (-eta * m).exp()).collect();
/// let total: f64 = exps.iter().sum();
/// let closed_form = -(total / margins.len() as f64).ln() / eta;
/// // Both sides are computed in closed form; only rounding error remains.
/// assert!((objective.objective_value(&theta) - closed_form).abs() < 1e-12);
///
/// let d = objective.gradient(&theta);
/// for (di, e) in d.iter().zip(&exps) {
///     assert!((di - e / total).abs() < 1e-12);
/// }
///
/// // With ν = 2 the weight of the smallest margin is capped at 1/2; the
/// // others keep their Gibbs ratios and share the remaining 1/2.
/// let capped = ErlpSoftMarginObjective::new(2.0, eta).gradient(&theta);
/// assert!((capped[2] - 0.5).abs() < 1e-12);
/// assert!((capped.iter().sum::<f64>() - 1.0).abs() < 1e-12);
/// assert!((capped[1] / capped[0] - exps[1] / exps[0]).abs() < 1e-9);
/// ```
#[derive(Clone)]
pub struct ErlpSoftMarginObjective {
    nu: f64,
    eta: f64,
}

impl ErlpSoftMarginObjective {
    /// Creates the objective with capping parameter `ν` (`1 ≤ ν ≤ m`) and
    /// regularization parameter `η > 0`. Neither value is validated here.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{ErlpSoftMarginObjective, ObjectiveFunction};
    ///
    /// let objective = ErlpSoftMarginObjective::new(2.0, 10.0);
    /// // η-smooth w.r.t. the ℓ∞-norm (`f64::MAX` encodes p = ∞).
    /// assert_eq!(objective.smooth(), (10.0, f64::MAX));
    /// ```
    pub fn new(nu: f64, eta: f64) -> Self {
        Self { nu, eta }
    }
}

impl ObjectiveFunction for ErlpSoftMarginObjective {
    fn name(&self) -> &str {
        "The dual objective for ErlpBoost"
    }

    fn smooth(&self) -> (f64, f64) {
        (self.eta, f64::MAX)
    }

    /// Returns the regularized soft margin `-f*(θ)` at `θ = -Aw`.
    fn objective_value(&self, point: &[f64]) -> f64 {
        let dist = self.gradient(point);
        let edge = -helpers::inner_product(point, &dist[..]);
        let entropy = helpers::entropy_from_uni_distribution(&dist[..]);

        edge + (entropy / self.eta)
    }

    /// Returns `∇f*(θ) = d*(θ)` at `θ = -Aw`, a distribution in `Δ_{m,ν}`.
    fn gradient(&self, point: &[f64]) -> Vec<f64> {
        let point = point.iter().map(|&pi| -pi);
        helpers::exp_distribution_from_margins(self.eta, self.nu, point)
    }
}

/// The negative Shannon entropy `f(p) = Σ_i p_i ln p_i` (with `0 ln 0 = 0`).
///
/// Its gradient is `∇f(p)_i = 1 + ln p_i`; at `p_i = 0` the implementation
/// returns `f64::MIN` in place of `-∞`.
///
/// This function is **not** smooth on the simplex (its gradient is unbounded
/// near the boundary). [`smooth`](ObjectiveFunction::smooth) nevertheless
/// returns `(1, 1)`, which is not a valid smoothness constant, so
/// [`FwUpdateRule::ShortStep`](super::FwUpdateRule::ShortStep) has no
/// guarantee for this objective. It is used in tests of the Frank-Wolfe step
/// rules.
///
/// # Examples
/// ```
/// use miniboosts::{Entropy, ObjectiveFunction};
///
/// let f = Entropy::new();
/// let p = [0.5, 0.5, 0.0];
/// // 2 · (1/2) ln(1/2) + 0 = -ln 2
/// assert!((f.objective_value(&p) + 2f64.ln()).abs() < 1e-12);
///
/// let grad = f.gradient(&p);
/// assert!((grad[0] - (1.0 - 2f64.ln())).abs() < 1e-12);
/// // `f64::MIN` stands in for -∞ at a zero coordinate.
/// assert_eq!(grad[2], f64::MIN);
/// ```
pub struct Entropy;

impl Entropy {
    /// Creates the objective.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{Entropy, ObjectiveFunction};
    ///
    /// let f = Entropy::new();
    /// assert_eq!(f.name(), "Entropy");
    /// // A placeholder, not a valid smoothness constant (see the type docs).
    /// assert_eq!(f.smooth(), (1.0, 1.0));
    /// ```
    pub fn new() -> Self {
        Self {}
    }
}

impl ObjectiveFunction for Entropy {
    fn name(&self) -> &str {
        "Entropy"
    }

    fn smooth(&self) -> (f64, f64) {
        (1f64, 1f64)
    }

    fn objective_value(&self, point: &[f64]) -> f64 {
        point
            .iter()
            .map(|&p| if p == 0f64 { 0f64 } else { p * p.ln() })
            .sum::<f64>()
    }

    fn gradient(&self, point: &[f64]) -> Vec<f64> {
        point
            .iter()
            .map(|&p| if p != 0f64 { 1f64 + p.ln() } else { f64::MIN })
            .collect()
    }
}
