//! Adaptive first-order methods: [`Adam`] and [`Adopt`].
//!
//! Both minimize a differentiable function `f : R^n → R` from (possibly
//! stochastic) gradients and know nothing about where the parameters come
//! from. Choose one with [`AdaptiveOptimizer`], then call
//! [`AdaptiveState::step`] once per gradient:
//! ```
//! use miniboosts::optimization::adaptive::{AdaptiveOptimizer, AdaptiveState, Adopt};
//!
//! // f(x) = (x_0 - 3)² + 10 (x_1 + 1)², minimized at (3, -1).
//! let gradient = |x: &[f64]| vec![2.0 * (x[0] - 3.0), 20.0 * (x[1] + 1.0)];
//! let mut x = vec![0.0, 0.0];
//! let mut state = AdaptiveState::new(AdaptiveOptimizer::Adopt(Adopt::new(0.05)), 2);
//! for _ in 0..2_000 {
//!     let g = gradient(&x);
//!     state.step(&mut x, &g);
//! }
//! // ADOPT's first call only initializes the second moment.
//! assert_eq!(state.updates(), 1_999);
//! assert!((x[0] - 3.0).abs() < 1e-2 && (x[1] + 1.0).abs() < 1e-2, "{x:?}");
//! ```

use serde::{Deserialize, Serialize};

/// Adam (Kingma and Ba, 2015, Algorithm 1), with bias correction.
///
/// For the `t`-th gradient `g_t` (`t = 1, 2, ...`), starting from
/// `m_0 = v_0 = 0`:
/// ```txt
/// m_t = β₁ m_{t-1} + (1 - β₁) g_t
/// v_t = β₂ v_{t-1} + (1 - β₂) g_t ⊙ g_t
/// θ_t = θ_{t-1} - α (m_t / (1 - β₁^t)) / (sqrt(v_t / (1 - β₂^t)) + ε)
/// ```
///
/// # Guarantee
/// The paper's convergence analysis is for online convex optimization, with
/// a decaying step size and `β₁² < sqrt(β₂)`. Reddi, Kale, and Kumar (2018)
/// showed that this analysis has a gap and that Adam can diverge on a convex
/// problem for some `(β₁, β₂)`. With the constant step size used here,
/// there is no guarantee for non-convex objectives in general.
///
/// # Paper
/// Diederik P. Kingma and Jimmy Ba, [Adam: A Method for Stochastic
/// Optimization](https://arxiv.org/abs/1412.6980), ICLR 2015.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
}

impl Adam {
    /// Adam with step size `α = learning_rate` and the paper's defaults
    /// `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`.
    ///
    /// # Panics
    /// Panics unless `learning_rate` is finite and positive.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::optimization::adaptive::Adam;
    ///
    /// let adam = Adam::new(1e-3).betas(0.8, 0.99).epsilon(1e-10);
    /// assert_eq!(adam.learning_rate(), 1e-3);
    /// ```
    pub fn new(learning_rate: f64) -> Self {
        check_learning_rate(learning_rate);
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    /// Sets the decay rates `β₁` and `β₂` of the moment estimates.
    ///
    /// # Panics
    /// Panics unless `0 ≤ β₁ < 1` and `0 ≤ β₂ < 1`.
    ///
    /// # Examples
    /// See [`Adam::new`].
    #[must_use]
    pub fn betas(mut self, beta1: f64, beta2: f64) -> Self {
        check_beta("beta1", beta1);
        check_beta("beta2", beta2);
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Sets `ε`, added to the square root of the second moment.
    ///
    /// # Panics
    /// Panics unless `epsilon` is finite and positive.
    ///
    /// # Examples
    /// See [`Adam::new`].
    #[must_use]
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        check_epsilon(epsilon);
        self.epsilon = epsilon;
        self
    }

    /// Returns the step size `α`.
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

/// ADOPT (Taniguchi et al., 2024): Adam with the gradient normalized by the
/// *previous* second-moment estimate before the momentum update.
///
/// The first gradient `g_0` only initializes `v_0 = g_0 ⊙ g_0`; the
/// parameters do not move. For `t = 1, 2, ...`, with `m_0 = 0`
/// (Algorithm 2 of the paper):
/// ```txt
/// m_t = β₁ m_{t-1} + (1 - β₁) Clip(g_t / max(sqrt(v_{t-1}), ε), c_t)
/// θ_t = θ_{t-1} - α m_t
/// v_t = β₂ v_{t-1} + (1 - β₂) g_t ⊙ g_t
/// ```
/// where `Clip(a, c)_i = min(max(a_i, -c), c)` (Eq. 16) and `c_t = t^{1/4}`.
/// [`Adopt::without_clipping`] sets `c_t = ∞`, which is Algorithm 3 (the
/// paper's equivalent form of Algorithm 1). `ε` enters as a floor
/// `max(sqrt(v), ε)` as in the paper's algorithms and the official code; the
/// analysis writes `sqrt(v + ε²)` instead.
///
/// # Guarantee
/// For a lower-bounded, `L`-smooth `f` and unbiased stochastic gradients
/// with `E‖g_t‖² ≤ G²` (Assumptions 2.1, 2.2, 2.3, 2.5), any `β₂ ∈ [0, 1)`,
/// and the constant step size `α = Θ(1/sqrt(T))`, the unclipped method
/// satisfies `min_t E[‖∇f(θ_{t-1})‖^{4/3}]^{3/2} = O(1/sqrt(T))` after `T`
/// updates (Theorem 4.1); the clipped one has the same rate with
/// `c_t = Θ(t^{1/4})` (Theorems E.2 and E.3). This is convergence to a
/// stationary point, not to a minimizer. Full-batch gradients satisfy the
/// assumptions with zero variance. Gradients of minibatches drawn without
/// replacement are not unbiased, and the paper (Appendix B) shows that ADOPT
/// can then fail.
///
/// # Limitations
/// The step size here is the constant `α` chosen by the caller; the rate
/// above needs `α` proportional to `1/sqrt(T)`.
///
/// # Paper
/// Shohei Taniguchi, Keno Harada, Gouki Minegishi, Yuta Oshima, Seong Cheol
/// Jeong, Go Nagahara, Tomoshi Iiyama, Masahiro Suzuki, Yusuke Iwasawa, and
/// Yutaka Matsuo, [ADOPT: Modified Adam Can Converge with Any β₂ with the
/// Optimal Rate](https://arxiv.org/abs/2411.02853), NeurIPS 2024. The
/// clipping by `t^{1/4}` is the default of the authors' implementation
/// (<https://github.com/iShohei220/adopt>); the paper's experiments used the
/// unclipped Algorithm 1.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Adopt {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    clip: bool,
}

impl Adopt {
    /// ADOPT with step size `α = learning_rate`, clipping by `t^{1/4}`, and
    /// the paper's defaults (Table 5) `β₁ = 0.9`, `β₂ = 0.9999`, `ε = 1e-6`.
    ///
    /// # Panics
    /// Panics unless `learning_rate` is finite and positive.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::optimization::adaptive::Adopt;
    ///
    /// let adopt = Adopt::new(1e-3).betas(0.9, 0.999).without_clipping();
    /// assert!(!adopt.is_clipped());
    /// ```
    pub fn new(learning_rate: f64) -> Self {
        check_learning_rate(learning_rate);
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.9999,
            epsilon: 1e-6,
            clip: true,
        }
    }

    /// Sets the decay rates `β₁` and `β₂`.
    ///
    /// # Panics
    /// Panics unless `0 ≤ β₁ < 1` and `0 ≤ β₂ < 1`.
    ///
    /// # Examples
    /// See [`Adopt::new`].
    #[must_use]
    pub fn betas(mut self, beta1: f64, beta2: f64) -> Self {
        check_beta("beta1", beta1);
        check_beta("beta2", beta2);
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Sets the floor `ε` of `sqrt(v)`.
    ///
    /// # Panics
    /// Panics unless `epsilon` is finite and positive.
    ///
    /// # Examples
    /// See [`Adopt::new`].
    #[must_use]
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        check_epsilon(epsilon);
        self.epsilon = epsilon;
        self
    }

    /// Turns the clipping off: Algorithm 1 (in the form of Algorithm 3) of
    /// the paper.
    ///
    /// # Examples
    /// See [`Adopt::new`].
    #[must_use]
    pub fn without_clipping(mut self) -> Self {
        self.clip = false;
        self
    }

    /// Returns the step size `α`.
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Returns `true` if the normalized gradient is clipped by `t^{1/4}`.
    pub fn is_clipped(&self) -> bool {
        self.clip
    }
}

/// The method used by an [`AdaptiveState`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AdaptiveOptimizer {
    /// [`Adam`].
    Adam(Adam),
    /// [`Adopt`].
    Adopt(Adopt),
}

impl AdaptiveOptimizer {
    /// Returns the step size `α`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::optimization::adaptive::{AdaptiveOptimizer, Adam};
    ///
    /// assert_eq!(AdaptiveOptimizer::Adam(Adam::new(0.01)).learning_rate(), 0.01);
    /// ```
    pub fn learning_rate(&self) -> f64 {
        match self {
            Self::Adam(adam) => adam.learning_rate,
            Self::Adopt(adopt) => adopt.learning_rate,
        }
    }

    /// Returns `"Adam"` or `"ADOPT"`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Adam(_) => "Adam",
            Self::Adopt(_) => "ADOPT",
        }
    }
}

/// The moment estimates of one run of an [`AdaptiveOptimizer`].
///
/// # Examples
/// See the [module documentation](self).
#[derive(Debug, Clone)]
pub struct AdaptiveState {
    optimizer: AdaptiveOptimizer,
    first: Vec<f64>,
    second: Vec<f64>,
    // Number of gradients received.
    gradients: u64,
}

impl AdaptiveState {
    /// Starts a run of `optimizer` on `dimension` parameters.
    ///
    /// Time complexity: `O(dimension)`.
    pub fn new(optimizer: AdaptiveOptimizer, dimension: usize) -> Self {
        Self {
            optimizer,
            first: vec![0.0; dimension],
            second: vec![0.0; dimension],
            gradients: 0,
        }
    }

    /// Takes the gradient `grad` at `params` and updates `params` in place.
    /// Returns `false` if the parameters did not move, which happens only on
    /// ADOPT's first call (it initializes `v_0`).
    ///
    /// Time complexity: `O(n)` for `n` parameters.
    ///
    /// # Panics
    /// Panics if `params` or `grad` does not have the dimension given to
    /// [`AdaptiveState::new`], or if `grad` has a non-finite entry.
    pub fn step(&mut self, params: &mut [f64], grad: &[f64]) -> bool {
        let n = self.first.len();
        assert_eq!(params.len(), n, "the parameters have the wrong dimension");
        assert_eq!(grad.len(), n, "the gradient has the wrong dimension");
        assert!(
            grad.iter().all(|g| g.is_finite()),
            "the gradient has a non-finite entry"
        );
        self.gradients += 1;
        match self.optimizer {
            AdaptiveOptimizer::Adam(Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
            }) => {
                let t = self.gradients as f64;
                let bias1 = 1.0 - beta1.powf(t);
                let bias2 = 1.0 - beta2.powf(t);
                for i in 0..n {
                    let g = grad[i];
                    self.first[i] = beta1 * self.first[i] + (1.0 - beta1) * g;
                    self.second[i] = beta2 * self.second[i] + (1.0 - beta2) * g * g;
                    let m_hat = self.first[i] / bias1;
                    let v_hat = self.second[i] / bias2;
                    params[i] -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
                }
                true
            }
            AdaptiveOptimizer::Adopt(Adopt {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                clip,
            }) => {
                if self.gradients == 1 {
                    for (v, g) in self.second.iter_mut().zip(grad) {
                        *v = g * g;
                    }
                    return false;
                }
                // The update index t = 1, 2, ... (the first gradient is g_0).
                let t = (self.gradients - 1) as f64;
                let bound = if clip { t.powf(0.25) } else { f64::INFINITY };
                for i in 0..n {
                    let g = grad[i];
                    let normalized = (g / self.second[i].sqrt().max(epsilon)).clamp(-bound, bound);
                    self.first[i] = beta1 * self.first[i] + (1.0 - beta1) * normalized;
                    params[i] -= learning_rate * self.first[i];
                    self.second[i] = beta2 * self.second[i] + (1.0 - beta2) * g * g;
                }
                true
            }
        }
    }

    /// Returns the number of parameter updates so far: the number of calls
    /// to [`AdaptiveState::step`], minus one for ADOPT.
    pub fn updates(&self) -> u64 {
        match self.optimizer {
            AdaptiveOptimizer::Adam(_) => self.gradients,
            AdaptiveOptimizer::Adopt(_) => self.gradients.saturating_sub(1),
        }
    }
}

fn check_learning_rate(learning_rate: f64) {
    assert!(
        learning_rate.is_finite() && learning_rate > 0.0,
        "the learning rate must be finite and positive, got {learning_rate}"
    );
}

fn check_beta(name: &str, beta: f64) {
    assert!(
        (0.0..1.0).contains(&beta),
        "{name} must lie in [0, 1), got {beta}"
    );
}

fn check_epsilon(epsilon: f64) {
    assert!(
        epsilon.is_finite() && epsilon > 0.0,
        "epsilon must be finite and positive, got {epsilon}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adam_follows_algorithm_one() {
        // Two steps by hand with α = 0.1, β₁ = 0.5, β₂ = 0.75, ε = 1e-8.
        let adam = Adam::new(0.1).betas(0.5, 0.75);
        let mut state = AdaptiveState::new(AdaptiveOptimizer::Adam(adam), 1);
        let mut x = [1.0];
        assert!(state.step(&mut x, &[2.0]));
        // m = 1, v = 1; m̂ = 2, v̂ = 4: the step is 0.1 · 2 / (2 + ε).
        let expected = 1.0 - 0.1 * 2.0 / (2.0 + 1e-8);
        assert!((x[0] - expected).abs() < 1e-15);
        state.step(&mut x, &[-1.0]);
        // m = 0.5·1 + 0.5·(-1) = 0, so the second step is zero.
        assert!((x[0] - expected).abs() < 1e-15);
        assert_eq!(state.updates(), 2);
    }

    #[test]
    fn adopt_follows_algorithm_two() {
        // α = 0.1, β₁ = 0.5, β₂ = 0.75, ε = 1e-6.
        let adopt = Adopt::new(0.1).betas(0.5, 0.75);
        let mut state = AdaptiveState::new(AdaptiveOptimizer::Adopt(adopt), 2);
        let mut x = [1.0, 1.0];
        // g_0 initializes v_0 = (4, 1e-14) and moves nothing.
        assert!(!state.step(&mut x, &[2.0, 1e-7]));
        assert_eq!(x, [1.0, 1.0]);
        assert_eq!(state.updates(), 0);
        // t = 1, c_1 = 1: g_1 / max(sqrt(v_0), ε) = (3/2, 3) is clipped to
        // (1, 1), so m_1 = (0.5, 0.5) and x moves by -0.05 in each entry.
        assert!(state.step(&mut x, &[3.0, 3e-6]));
        assert!((x[0] - 0.95).abs() < 1e-15 && (x[1] - 0.95).abs() < 1e-15);
        // v_1 = 0.75 v_0 + 0.25 g_1²: (5.25, …); t = 2 uses v_1.
        state.step(&mut x, &[0.0, 0.0]);
        // m_2 = 0.5 m_1, so x moves by another -0.025.
        assert!((x[0] - 0.925).abs() < 1e-15);
        assert_eq!(state.updates(), 2);

        // Without clipping, the first update uses the full ratio 3/2.
        let mut state = AdaptiveState::new(AdaptiveOptimizer::Adopt(adopt.without_clipping()), 1);
        let mut x = [1.0];
        state.step(&mut x, &[2.0]);
        state.step(&mut x, &[3.0]);
        assert!((x[0] - (1.0 - 0.1 * 0.5 * 1.5)).abs() < 1e-15);
    }

    #[test]
    fn both_minimize_a_quadratic() {
        // f(x) = Σ_i i (x_i - 1)², a badly scaled but convex problem.
        for optimizer in [
            AdaptiveOptimizer::Adam(Adam::new(0.05)),
            AdaptiveOptimizer::Adopt(Adopt::new(0.05)),
            AdaptiveOptimizer::Adopt(Adopt::new(0.05).without_clipping()),
        ] {
            let mut x = vec![0.0; 5];
            let mut state = AdaptiveState::new(optimizer, 5);
            for _ in 0..3_000 {
                let g: Vec<f64> = (0..5)
                    .map(|i| 2.0 * (i + 1) as f64 * (x[i] - 1.0))
                    .collect();
                state.step(&mut x, &g);
            }
            assert!(
                x.iter().all(|v| (v - 1.0).abs() < 1e-2),
                "{optimizer:?}: {x:?}"
            );
        }
    }

    #[test]
    fn invalid_inputs_panic() {
        assert!(std::panic::catch_unwind(|| Adam::new(0.0)).is_err());
        assert!(std::panic::catch_unwind(|| Adopt::new(f64::NAN)).is_err());
        assert!(std::panic::catch_unwind(|| Adam::new(0.1).betas(1.0, 0.5)).is_err());
        assert!(std::panic::catch_unwind(|| Adopt::new(0.1).betas(0.5, -0.1)).is_err());
        assert!(std::panic::catch_unwind(|| Adam::new(0.1).epsilon(0.0)).is_err());
        let mut state = AdaptiveState::new(AdaptiveOptimizer::Adam(Adam::new(0.1)), 2);
        let mut x = [0.0, 0.0];
        for grad in [vec![1.0], vec![1.0, f64::INFINITY]] {
            let mut copy = state.clone();
            assert!(std::panic::catch_unwind(move || copy.step(&mut x, &grad)).is_err());
        }
        assert!(state.step(&mut x, &[1.0, 1.0]));
    }
}
