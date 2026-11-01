//! Step-size rules for the Frank-Wolfe (conditional gradient) method.
//!
//! [`FrankWolfe`] does not run a Frank-Wolfe loop itself: the caller supplies
//! the current point `x` and the linear-minimization (Frank-Wolfe) vertex `v`,
//! and [`FrankWolfe`] returns a step size for
//! ```txt
//! x_{t+1} = x_t + λ_t (v_t - x_t),   λ_t ∈ [0, 1],
//! ```
//! according to a [`FwUpdateRule`]. [`MlpBoost`](crate::MlpBoost) and
//! [`CorrectiveErlpBoost`](crate::CorrectiveErlpBoost) call it with points in
//! negated-margin coordinates (`x = -Aw`, see
//! [`objective_function`](super::objective_function)) and apply the returned
//! step to their hypothesis weights.
//!
//! References: Frank and Wolfe (1956), "An algorithm for quadratic
//! programming"; Jaggi (2013), "Revisiting Frank-Wolfe: Projection-free sparse
//! convex optimization"; Tsuji, Tanaka and Pokutta (2022), "Pairwise
//! conditional gradients without swap steps and sparser kernel herding"
//! (blended pairwise variant).
use crate::core::{
    constants::BINARY_SEARCH_TOLERANCE,
    tools::{checkers, helpers},
};
use crate::optimization::objective_function::ObjectiveFunction;
use std::fmt;

/// Step-size rule used by [`FrankWolfe`].
///
/// In the formulas below `x` is the current point, `v` the Frank-Wolfe
/// vertex, `f` the [`ObjectiveFunction`], and `t` the internal iteration
/// counter of [`FrankWolfe`] (see [`FrankWolfe::iteration`]).
///
/// # Examples
/// Minimize `f(x) = ‖x - c‖² / 2` over the probability simplex in `R^3`
/// (optimum `x = c`, `f = 0`); the linear-minimization oracle returns the
/// vertex `e_k` with the smallest gradient entry. For the classic, short-step,
/// and line-search rules, Jaggi (2013, Theorem 1) bounds the error after `t`
/// steps by `2 C_f / (t + 2)` with `C_f ≤ L diam(Δ)² = 1 · 2`.
/// ```
/// # /// `f(x) = ‖x - c‖² / 2`, 1-smooth w.r.t. the ℓ2-norm.
/// # struct SquaredDistance(Vec<f64>);
/// # impl miniboosts::ObjectiveFunction for SquaredDistance {
/// #     fn name(&self) -> &str { "Squared distance" }
/// #     fn smooth(&self) -> (f64, f64) { (1.0, 2.0) }
/// #     fn objective_value(&self, x: &[f64]) -> f64 {
/// #         x.iter().zip(&self.0).map(|(a, c)| (a - c).powi(2)).sum::<f64>() / 2.0
/// #     }
/// #     fn gradient(&self, x: &[f64]) -> Vec<f64> {
/// #         x.iter().zip(&self.0).map(|(a, c)| a - c).collect()
/// #     }
/// # }
/// use miniboosts::{FrankWolfe, FwUpdateRule, ObjectiveFunction};
///
/// let c = vec![0.2, 0.3, 0.5];
/// let f = SquaredDistance(c.clone());
/// let n_steps = 50;
/// for rule in [FwUpdateRule::Classic, FwUpdateRule::ShortStep, FwUpdateRule::LineSearch] {
///     let mut fw = FrankWolfe::new(SquaredDistance(c.clone()), rule);
///     let mut x = vec![1.0, 0.0, 0.0];
///     for _ in 0..n_steps {
///         let grad = f.gradient(&x);
///         let k = (0..3).min_by(|&i, &j| grad[i].total_cmp(&grad[j])).unwrap();
///         let mut vertex = vec![0.0; 3];
///         vertex[k] = 1.0;
///         x = fw.next(x, vertex);
///     }
///     // Iterates stay in the simplex.
///     assert!((x.iter().sum::<f64>() - 1.0).abs() < 1e-12);
///     assert!(f.objective_value(&x) <= 4.0 / (n_steps as f64 + 2.0), "{rule}");
/// }
/// assert_eq!(FwUpdateRule::LineSearch.to_string(), "Line-search update");
/// ```
#[derive(Clone, Copy)]
pub enum FwUpdateRule {
    /// Classic open-loop step size
    /// ```txt
    /// λ = 2 / (t + 2).
    /// ```
    /// Does not evaluate the objective.
    Classic,

    /// Short step, the minimizer over `[0, 1]` of the smoothness upper bound:
    /// ```txt
    /// λ = clamp( ⟨∇f(x), x - v⟩ / (L ‖v - x‖_p^2), 0, 1 ),
    /// ```
    /// where `(L, p) = f.smooth()` and `p == f64::MAX` means the ℓ∞-norm.
    /// If `x` has length one, `λ = 1` is returned without evaluating `f`.
    ///
    /// Requires a valid smoothness constant `L > 0`. If `v == x` or `L == 0`,
    /// the quotient is NaN or infinite; a NaN step is not caught here, but
    /// [`FrankWolfe::next`] panics on it.
    ShortStep,

    /// Line search on the segment `[x, v]`:
    /// ```txt
    /// λ ≈ argmin_{λ ∈ [0, 1]} f(x + λ (v - x)),
    /// ```
    /// found by bisection on the sign of the directional derivative
    /// `⟨∇f(x + λ (v - x)), v - x⟩` until the bracket is at most `1e-10`
    /// wide; the bracket midpoint is returned. This is exact only up to that
    /// tolerance and assumes `f` is convex along the segment.
    ///
    /// Panics if a directional derivative is not finite.
    LineSearch,

    /// Blended pairwise Frank-Wolfe step
    /// ([Tsuji, Tanaka and Pokutta, 2022](https://proceedings.mlr.press/v162/tsuji22a)).
    ///
    /// Treats the coordinates of `x` as convex-combination weights on the
    /// canonical basis vectors and assumes `v` is a canonical basis vector.
    /// Among the coordinates with nonzero weight it picks the away atom `a`
    /// (largest gradient entry) and the local Frank-Wolfe atom `s` (smallest
    /// gradient entry). If `⟨∇f(x), s - a⟩ ≤ ⟨∇f(x), v - x⟩` it returns a
    /// pairwise step [`StepSize::BpfwMoveWeights`] in direction `s - a` with
    /// line search on `[0, x_a]`; otherwise it returns a Frank-Wolfe step
    /// [`StepSize::Normal`] with line search on `[0, 1]`.
    ///
    /// [`MlpBoost`](crate::MlpBoost) and
    /// [`CorrectiveErlpBoost`](crate::CorrectiveErlpBoost) reject this rule
    /// (their `update_rule` methods panic on it), since their points are not
    /// weight vectors over canonical basis vectors. See [`StepSize`] for a
    /// worked example.
    BlendedPairwise,
}

impl fmt::Display for FwUpdateRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rule = match self {
            Self::Classic => "Classic update",
            Self::ShortStep => "Short-step update",
            Self::LineSearch => "Line-search update",
            Self::BlendedPairwise => "Blended Pairwise update",
        };
        write!(f, "{rule}")
    }
}

/// A step returned by [`FrankWolfe::get_stepsize_mut`].
///
/// # Examples
/// Two blended pairwise steps on `f(x) = ‖x - c‖² / 2` with `c = (0.2, 0.3,
/// 0.5)`, computed by hand:
/// 1. At `x = (1/2, 1/2, 0)`, `∇f = (0.3, 0.2, -0.5)` and the vertex is
///    `e_3`. The pairwise direction `e_2 - e_1` has slope `-0.1`, the
///    Frank-Wolfe direction `e_3 - x` has slope `-0.75`, so a Frank-Wolfe
///    step is taken; exact line search gives `λ = 0.75 / 1.5 = 1/2`.
/// 2. At `x = (1/4, 1/4, 1/2)`, `∇f = (0.05, -0.05, 0)` and the vertex is
///    `e_2`. Now the pairwise slope `-0.1` beats `-0.05`, so weight moves
///    from `e_1` to `e_2`; line search on `[0, 1/4]` gives `0.05`, which
///    reaches the optimum `c`.
///
/// Line search bisects to a bracket of width `1e-10`, hence the tolerance.
/// ```
/// # /// `f(x) = ‖x - c‖² / 2`, 1-smooth w.r.t. the ℓ2-norm.
/// # struct SquaredDistance(Vec<f64>);
/// # impl miniboosts::ObjectiveFunction for SquaredDistance {
/// #     fn name(&self) -> &str { "Squared distance" }
/// #     fn smooth(&self) -> (f64, f64) { (1.0, 2.0) }
/// #     fn objective_value(&self, x: &[f64]) -> f64 {
/// #         x.iter().zip(&self.0).map(|(a, c)| (a - c).powi(2)).sum::<f64>() / 2.0
/// #     }
/// #     fn gradient(&self, x: &[f64]) -> Vec<f64> {
/// #         x.iter().zip(&self.0).map(|(a, c)| a - c).collect()
/// #     }
/// # }
/// use miniboosts::{FrankWolfe, FwUpdateRule, StepSize};
///
/// let mut fw = FrankWolfe::new(
///     SquaredDistance(vec![0.2, 0.3, 0.5]),
///     FwUpdateRule::BlendedPairwise,
/// );
///
/// let x = [0.5, 0.5, 0.0];
/// let StepSize::Normal(lambda) = fw.get_stepsize_mut(&x, &[0.0, 0.0, 1.0]) else {
///     panic!("expected a Frank-Wolfe step");
/// };
/// assert!((lambda - 0.5).abs() < 1e-9);
///
/// let x = [0.25, 0.25, 0.5];
/// match fw.get_stepsize_mut(&x, &[0.0, 1.0, 0.0]) {
///     StepSize::BpfwMoveWeights { stepsize, dir } => {
///         assert_eq!(dir, vec![-1.0, 1.0, 0.0]);
///         assert!((stepsize - 0.05).abs() < 1e-9);
///     }
///     StepSize::Normal(_) => panic!("expected a pairwise step"),
/// }
///
/// // `next` applies the step: x + stepsize * dir = c.
/// fw.iteration(0);
/// let x = fw.next(vec![0.25, 0.25, 0.5], vec![0.0, 1.0, 0.0]);
/// for (xi, ci) in x.iter().zip([0.2, 0.3, 0.5]) {
///     assert!((xi - ci).abs() < 1e-9);
/// }
/// ```
#[derive(Clone)]
pub enum StepSize {
    /// An ordinary Frank-Wolfe step `x + λ (v - x)` with the given `λ`.
    Normal(f64),
    /// A pairwise step `x + stepsize * dir` of
    /// [`FwUpdateRule::BlendedPairwise`], which moves weight from an away atom
    /// to a local Frank-Wolfe atom.
    BpfwMoveWeights {
        /// Step length along `dir`, in `[0, x_a]` for the away atom `a`.
        stepsize: f64,
        /// Direction `s - a`, the difference of two canonical basis vectors.
        dir: Vec<f64>,
    },
}

/// Step-size computation for the Frank-Wolfe method on an objective `F`.
///
/// Holds the objective, the [`FwUpdateRule`], and an iteration counter `t`
/// that starts at `0` and is incremented by every call to
/// [`get_stepsize_mut`](Self::get_stepsize_mut) (and hence
/// [`next`](Self::next)). The counter is used only by
/// [`FwUpdateRule::Classic`].
///
/// Convergence guarantees of the Frank-Wolfe method (e.g., the `O(1/t)` rate
/// for smooth convex objectives over a compact convex set, Jaggi 2013) hold
/// only under their assumptions, which are not checked here.
///
/// # Examples
/// The caller runs the Frank-Wolfe loop: compute the gradient, pick the
/// vertex that minimizes the linear model, and let [`next`](Self::next)
/// take the step. Here `f(x) = ‖x - c‖² / 2` is minimized over the
/// probability simplex, stopping when the Frank-Wolfe gap
/// `⟨∇f(x), x - v⟩ ≥ f(x) - f*` is small.
/// ```
/// # /// `f(x) = ‖x - c‖² / 2`, 1-smooth w.r.t. the ℓ2-norm.
/// # struct SquaredDistance(Vec<f64>);
/// # impl miniboosts::ObjectiveFunction for SquaredDistance {
/// #     fn name(&self) -> &str { "Squared distance" }
/// #     fn smooth(&self) -> (f64, f64) { (1.0, 2.0) }
/// #     fn objective_value(&self, x: &[f64]) -> f64 {
/// #         x.iter().zip(&self.0).map(|(a, c)| (a - c).powi(2)).sum::<f64>() / 2.0
/// #     }
/// #     fn gradient(&self, x: &[f64]) -> Vec<f64> {
/// #         x.iter().zip(&self.0).map(|(a, c)| a - c).collect()
/// #     }
/// # }
/// use miniboosts::{FrankWolfe, FwUpdateRule, ObjectiveFunction};
///
/// let c = vec![0.2, 0.3, 0.5];
/// let f = SquaredDistance(c.clone());
/// let mut fw = FrankWolfe::new(SquaredDistance(c.clone()), FwUpdateRule::LineSearch);
///
/// let mut x = vec![1.0, 0.0, 0.0];
/// for _ in 0..1000 {
///     let grad = f.gradient(&x);
///     let k = (0..3).min_by(|&i, &j| grad[i].total_cmp(&grad[j])).unwrap();
///     let gap = grad.iter().zip(&x).map(|(g, xi)| g * xi).sum::<f64>() - grad[k];
///     if gap < 1e-8 {
///         break;
///     }
///     let mut vertex = vec![0.0; 3];
///     vertex[k] = 1.0;
///     x = fw.next(x, vertex);
/// }
/// // The gap bounds the suboptimality: f(x) - 0 ≤ 1e-8.
/// assert!(f.objective_value(&x) <= 1e-8);
/// ```
pub struct FrankWolfe<F> {
    objective: F,
    update_rule: FwUpdateRule,
    iteration: usize,
}

impl<F> FrankWolfe<F> {
    /// Creates a new instance with iteration counter `t = 0`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{Entropy, FrankWolfe, FwUpdateRule, StepSize};
    ///
    /// let fw = FrankWolfe::new(Entropy::new(), FwUpdateRule::Classic);
    /// assert!(matches!(fw.current_update_rule(), FwUpdateRule::Classic));
    /// // t = 0, so the first classic step is λ = 2 / (0 + 2) = 1.
    /// assert!(matches!(fw.classic(&[], &[]), StepSize::Normal(l) if l == 1.0));
    /// ```
    pub fn new(objective: F, update_rule: FwUpdateRule) -> Self {
        let iteration = 0;
        Self {
            objective,
            update_rule,
            iteration,
        }
    }

    /// Replaces the step-size rule; the iteration counter is kept.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{Entropy, FrankWolfe, FwUpdateRule};
    ///
    /// let mut fw = FrankWolfe::new(Entropy::new(), FwUpdateRule::Classic);
    /// fw.update_rule(FwUpdateRule::LineSearch);
    /// assert!(matches!(fw.current_update_rule(), FwUpdateRule::LineSearch));
    /// ```
    pub fn update_rule(&mut self, update_rule: FwUpdateRule) {
        self.update_rule = update_rule;
    }

    /// Returns the current Frank-Wolfe strategy.
    pub fn current_update_rule(&self) -> FwUpdateRule {
        self.update_rule
    }

    /// Sets the iteration counter `t` used by [`FwUpdateRule::Classic`].
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{Entropy, FrankWolfe, FwUpdateRule, StepSize};
    ///
    /// let mut fw = FrankWolfe::new(Entropy::new(), FwUpdateRule::Classic);
    /// fw.iteration(3);
    /// // λ = 2 / (3 + 2)
    /// assert!(matches!(fw.classic(&[], &[]), StepSize::Normal(l) if l == 0.4));
    /// ```
    pub fn iteration(&mut self, iteration: usize) {
        self.iteration = iteration;
    }

    /// Returns `self` with the objective replaced, keeping the update rule
    /// and the iteration counter.
    ///
    /// # Examples
    /// Boosters rebuild their objective when a parameter changes:
    /// ```
    /// use miniboosts::{ErlpSoftMarginObjective, FrankWolfe, FwUpdateRule, StepSize};
    ///
    /// let mut fw = FrankWolfe::new(ErlpSoftMarginObjective::new(1.0, 1.0), FwUpdateRule::Classic);
    /// fw.iteration(2);
    /// let fw = fw.objective(ErlpSoftMarginObjective::new(1.0, 10.0));
    /// // The counter t = 2 survives: λ = 2 / (2 + 2).
    /// assert!(matches!(fw.classic(&[], &[]), StepSize::Normal(l) if l == 0.5));
    /// ```
    pub fn objective(self, objective: F) -> Self {
        Self {
            objective,
            update_rule: self.update_rule,
            iteration: self.iteration,
        }
    }

    /// Returns the classic step size `λ = 2 / (t + 2)` for the update
    /// ```txt
    ///     x_{t+1} = x_t + λ (v_t - x_t),
    /// ```
    /// where `t` is the current iteration counter. The points are ignored and
    /// the counter is not incremented.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{Entropy, FrankWolfe, FwUpdateRule, StepSize};
    ///
    /// let mut fw = FrankWolfe::new(Entropy::new(), FwUpdateRule::Classic);
    /// let (x, v) = ([1.0, 0.0], [0.0, 1.0]);
    /// for t in 0..3 {
    ///     let StepSize::Normal(lambda) = fw.get_stepsize_mut(&x, &v) else {
    ///         unreachable!()
    ///     };
    ///     assert_eq!(lambda, 2.0 / (t as f64 + 2.0));
    /// }
    /// // `classic` itself does not advance the counter (now t = 3).
    /// assert!(matches!(fw.classic(&x, &v), StepSize::Normal(l) if l == 0.4));
    /// assert!(matches!(fw.classic(&x, &v), StepSize::Normal(l) if l == 0.4));
    /// ```
    pub fn classic(&self, _cur: &[f64], _nxt: &[f64]) -> StepSize {
        let stepsize = 2f64 / (self.iteration as f64 + 2f64);
        StepSize::Normal(stepsize)
    }
}

impl<F> FrankWolfe<F>
where
    F: ObjectiveFunction,
{
    /// Returns the short step size for current point `cur` and vertex `nxt`:
    /// ```txt
    ///     λ = clamp( ⟨∇f(x), x - v⟩ / (L ‖v - x‖_p^2), 0, 1 ),
    /// ```
    /// where `(L, p) = f.smooth()` (`p == f64::MAX` means the ℓ∞-norm).
    /// Returns `λ = 1` when `cur.len() == 1`. The result is NaN if `v == x`
    /// with a zero numerator or if `L == 0`; this is not checked here. The
    /// counter is not incremented.
    ///
    /// # Examples
    /// For `f(x) = ‖x - c‖² / 2` (`L = 1`, `p = 2`), `x = e_1`, `v = e_3`, and
    /// `c = (0.2, 0.3, 0.5)`: `⟨∇f(x), x - v⟩ = 0.8 + 0.5 = 1.3` and
    /// `‖v - x‖² = 2`, so `λ = 0.65`. For a quadratic the short step is the
    /// exact line minimizer.
    /// ```
    /// # /// `f(x) = ‖x - c‖² / 2`, 1-smooth w.r.t. the ℓ2-norm.
    /// # struct SquaredDistance(Vec<f64>);
    /// # impl miniboosts::ObjectiveFunction for SquaredDistance {
    /// #     fn name(&self) -> &str { "Squared distance" }
    /// #     fn smooth(&self) -> (f64, f64) { (1.0, 2.0) }
    /// #     fn objective_value(&self, x: &[f64]) -> f64 {
    /// #         x.iter().zip(&self.0).map(|(a, c)| (a - c).powi(2)).sum::<f64>() / 2.0
    /// #     }
    /// #     fn gradient(&self, x: &[f64]) -> Vec<f64> {
    /// #         x.iter().zip(&self.0).map(|(a, c)| a - c).collect()
    /// #     }
    /// # }
    /// use miniboosts::{FrankWolfe, FwUpdateRule, StepSize};
    ///
    /// let fw = FrankWolfe::new(SquaredDistance(vec![0.2, 0.3, 0.5]), FwUpdateRule::ShortStep);
    /// let StepSize::Normal(lambda) = fw.short_step(&[1.0, 0.0, 0.0], &[0.0, 0.0, 1.0]) else {
    ///     unreachable!()
    /// };
    /// assert!((lambda - 0.65).abs() < 1e-12);
    /// ```
    pub fn short_step(&self, cur: &[f64], nxt: &[f64]) -> StepSize {
        if cur.len() == 1 {
            return StepSize::Normal(1f64);
        }
        // if cur.len() == 1 { return vec![1f64]; }

        let grad = self.objective.gradient(cur);

        let (s, p) = self.objective.smooth();
        let numer = helpers::inner_product(&grad[..], cur) - helpers::inner_product(&grad[..], nxt);
        let denom = nxt
            .iter()
            .zip(cur)
            .fold(0f64, |acc, (n, c)| {
                if p == f64::MAX {
                    (n - c).abs().max(acc)
                } else {
                    acc + (n - c).abs().powf(p)
                }
            })
            .powf(if p == f64::MAX { 2f64 } else { 2f64 / p });

        let stepsize = (numer / (s * denom)).clamp(0f64, 1f64);
        StepSize::Normal(stepsize)
    }

    /// Seeks the best step size `λ ∈ [0, 1]` by line-search.
    /// ```txt
    /// λ =   arg min  f( (-Aw) + λ ((-Ae) - (-Aw)) )
    ///     λ ∈ [0, 1]
    /// ```
    fn line_search(&self, cur: &[f64], nxt: &[f64]) -> StepSize {
        let dir = nxt.iter().zip(cur).map(|(n, c)| n - c).collect::<Vec<_>>();

        let stepsize = self.line_search_inner(cur, &dir[..], 1f64);
        StepSize::Normal(stepsize)
    }

    fn line_search_inner(&self, cur: &[f64], dir: &[f64], mut ub: f64) -> f64 {
        let mut lb = 0f64;

        assert!(
            lb < ub,
            "the line segment is empty. [lb, ub] = [{lb}, {ub}]",
        );

        while ub - lb > BINARY_SEARCH_TOLERANCE {
            let stepsize = (lb + ub) / 2f64;
            let mid = cur
                .iter()
                .zip(dir)
                .map(|(c, d)| c + stepsize * d)
                .collect::<Vec<_>>();
            let grad = self.objective.gradient(&mid[..]);
            let val = helpers::inner_product(&grad[..], dir);
            // A NaN derivative would otherwise fall through to `break`
            // and return an unchecked midpoint as the step size.
            assert!(
                val.is_finite(),
                "line-search derivative is not finite: {val}"
            );
            if val < 0f64 {
                lb = stepsize;
            } else if val > 0f64 {
                ub = stepsize;
            } else {
                break;
            }
        }

        (lb + ub) / 2f64
    }

    fn find_away_atom(&self, cur: &[f64], grad: &[f64]) -> (f64, Vec<f64>) {
        let (ix, _) =
            grad.iter()
                .zip(cur)
                .enumerate()
                .fold((usize::MAX, f64::MIN), |acc, (i, (&g, &c))| {
                    if c == 0f64 || g < acc.1 { acc } else { (i, g) }
                });
        assert_ne!(ix, usize::MAX, "failed to find an away atom.");

        let dim = cur.len();
        let mut away = vec![0f64; dim];
        away[ix] = 1f64;
        (cur[ix], away)
    }

    fn find_localfw_atom(&self, cur: &[f64], grad: &[f64]) -> (f64, Vec<f64>) {
        let (ix, _) =
            grad.iter()
                .zip(cur)
                .enumerate()
                .fold((usize::MAX, f64::MAX), |acc, (i, (&g, &c))| {
                    if c == 0f64 || acc.1 < g { acc } else { (i, g) }
                });
        assert_ne!(ix, usize::MAX, "failed to find an local fw atom.");

        let dim = cur.len();
        let mut localfw = vec![0f64; dim];
        localfw[ix] = 1f64;
        (cur[ix], localfw)
    }

    /// The blended pairwise Frank-Wolfe algorithm.
    /// This function assumes that `nxt` is a canonical basis vector.
    fn bpfw(&self, cur: &[f64], nxt: &[f64]) -> StepSize {
        let grad = self.objective.gradient(cur);

        let (ub, a) = self.find_away_atom(cur, &grad[..]);
        let (_, s) = self.find_localfw_atom(cur, &grad[..]);
        let x = cur;
        let w = nxt;
        let sa = s
            .iter()
            .zip(&a[..])
            .map(|(si, ai)| si - ai)
            .collect::<Vec<_>>();
        let wx = w.iter().zip(x).map(|(wi, xi)| wi - xi).collect::<Vec<_>>();

        let stepsize;
        let dir;
        if helpers::inner_product(&grad[..], &sa[..]) <= helpers::inner_product(&grad[..], &wx[..])
        {
            dir = sa;
            stepsize = self.line_search_inner(x, &dir[..], ub);
            StepSize::BpfwMoveWeights { stepsize, dir }
        } else {
            dir = wx;
            stepsize = self.line_search_inner(x, &dir[..], 1f64);
            StepSize::Normal(stepsize)
        }
    }

    /// Computes the step from current point `cur` to vertex `nxt` with the
    /// current [`FwUpdateRule`], then increments the iteration counter.
    ///
    /// `cur` and `nxt` must have equal length. The step is not range-checked
    /// here.
    ///
    /// # Panics
    /// [`FwUpdateRule::LineSearch`] and [`FwUpdateRule::BlendedPairwise`]
    /// panic on a non-finite directional derivative;
    /// [`FwUpdateRule::BlendedPairwise`] also panics if `cur` has no nonzero
    /// coordinate.
    ///
    /// # Examples
    /// The line-search step on `f(x) = ‖x - c‖² / 2` from `e_1` toward `e_3`
    /// (the exact minimizer is `λ = 0.65`; bisection stops at a `1e-10`
    /// bracket):
    /// ```
    /// # /// `f(x) = ‖x - c‖² / 2`, 1-smooth w.r.t. the ℓ2-norm.
    /// # struct SquaredDistance(Vec<f64>);
    /// # impl miniboosts::ObjectiveFunction for SquaredDistance {
    /// #     fn name(&self) -> &str { "Squared distance" }
    /// #     fn smooth(&self) -> (f64, f64) { (1.0, 2.0) }
    /// #     fn objective_value(&self, x: &[f64]) -> f64 {
    /// #         x.iter().zip(&self.0).map(|(a, c)| (a - c).powi(2)).sum::<f64>() / 2.0
    /// #     }
    /// #     fn gradient(&self, x: &[f64]) -> Vec<f64> {
    /// #         x.iter().zip(&self.0).map(|(a, c)| a - c).collect()
    /// #     }
    /// # }
    /// use miniboosts::{FrankWolfe, FwUpdateRule, StepSize};
    ///
    /// let mut fw = FrankWolfe::new(SquaredDistance(vec![0.2, 0.3, 0.5]), FwUpdateRule::LineSearch);
    /// match fw.get_stepsize_mut(&[1.0, 0.0, 0.0], &[0.0, 0.0, 1.0]) {
    ///     StepSize::Normal(lambda) => assert!((lambda - 0.65).abs() < 1e-9),
    ///     StepSize::BpfwMoveWeights { .. } => unreachable!(),
    /// }
    /// ```
    pub fn get_stepsize_mut(&mut self, cur: &[f64], nxt: &[f64]) -> StepSize {
        let stepsize = match self.update_rule {
            FwUpdateRule::Classic => self.classic(cur, nxt),
            FwUpdateRule::ShortStep => self.short_step(cur, nxt),
            FwUpdateRule::LineSearch => self.line_search(cur, nxt),
            FwUpdateRule::BlendedPairwise => self.bpfw(cur, nxt),
        };
        self.iteration += 1;

        stepsize
    }

    /// Computes a step with [`get_stepsize_mut`](Self::get_stepsize_mut) and
    /// returns the next point:
    /// ```txt
    ///     x_{t+1} = x_t + λ (v_t - x_t)        for StepSize::Normal(λ),
    ///     x_{t+1} = x_t + stepsize * dir       for StepSize::BpfwMoveWeights.
    /// ```
    ///
    /// # Panics
    /// Panics if a [`StepSize::Normal`] step is outside `[0, 1]` (including
    /// NaN), and in the cases listed for
    /// [`get_stepsize_mut`](Self::get_stepsize_mut).
    ///
    /// # Examples
    /// One short step from `e_1` toward `e_3` with `λ = 0.65` (see
    /// [`short_step`](Self::short_step)):
    /// ```
    /// # /// `f(x) = ‖x - c‖² / 2`, 1-smooth w.r.t. the ℓ2-norm.
    /// # struct SquaredDistance(Vec<f64>);
    /// # impl miniboosts::ObjectiveFunction for SquaredDistance {
    /// #     fn name(&self) -> &str { "Squared distance" }
    /// #     fn smooth(&self) -> (f64, f64) { (1.0, 2.0) }
    /// #     fn objective_value(&self, x: &[f64]) -> f64 {
    /// #         x.iter().zip(&self.0).map(|(a, c)| (a - c).powi(2)).sum::<f64>() / 2.0
    /// #     }
    /// #     fn gradient(&self, x: &[f64]) -> Vec<f64> {
    /// #         x.iter().zip(&self.0).map(|(a, c)| a - c).collect()
    /// #     }
    /// # }
    /// use miniboosts::{FrankWolfe, FwUpdateRule};
    ///
    /// let mut fw = FrankWolfe::new(SquaredDistance(vec![0.2, 0.3, 0.5]), FwUpdateRule::ShortStep);
    /// let x = fw.next(vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0]);
    /// for (xi, expected) in x.iter().zip([0.35, 0.0, 0.65]) {
    ///     assert!((xi - expected).abs() < 1e-12);
    /// }
    /// ```
    pub fn next(&mut self, cur: Vec<f64>, nxt: Vec<f64>) -> Vec<f64> {
        match self.get_stepsize_mut(&cur[..], &nxt[..]) {
            StepSize::Normal(stepsize) => interior_point(cur, nxt, stepsize),
            StepSize::BpfwMoveWeights { stepsize, dir } => move_to_dir(cur, dir, stepsize),
        }
    }
}

/// Take the interior point of the given two arrays.
pub(crate) fn interior_point(cur: Vec<f64>, nxt: Vec<f64>, stepsize: f64) -> Vec<f64> {
    checkers::stepsize(stepsize);
    cur.iter()
        .zip(nxt)
        .map(|(c, n)| c + stepsize * (n - c))
        .collect()
}
/// Take the interior point of the given two arrays.
pub(crate) fn move_to_dir(cur: Vec<f64>, dir: Vec<f64>, stepsize: f64) -> Vec<f64> {
    cur.iter().zip(dir).map(|(c, d)| c + stepsize * d).collect()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::optimization::objective_function::Entropy;

    const TEST_TOLERANCE: f64 = 1e-3;

    /// ℓ₂-norm objective
    struct L2;
    impl L2 {
        fn new() -> Self {
            Self {}
        }
    }
    impl ObjectiveFunction for L2 {
        fn name(&self) -> &str {
            "L2-norm objective"
        }

        fn smooth(&self) -> (f64, f64) {
            (1f64, 2f64)
        }

        fn objective_value(&self, point: &[f64]) -> f64 {
            point.iter().map(|p| p.powi(2)).sum::<f64>() / 2f64
        }

        fn gradient(&self, point: &[f64]) -> Vec<f64> {
            point.to_vec()
        }
    }

    /// Returns a point `v ∈ [-1, +1]ⁿ` that minimizes
    /// the inner product `v · ∇f(x).`
    fn linear_minimization_oracle_l2(grad: &[f64]) -> Vec<f64> {
        grad.iter()
            .map(|&g| if g < 0f64 { 1f64 } else { -1f64 })
            .collect::<Vec<_>>()
    }

    /// Returns a point `v ∈ [0, +1]ⁿ` that minimizes
    /// the inner product `v · ∇f(x).`
    fn linear_minimization_oracle_entropy(grad: &[f64]) -> Vec<f64> {
        let dim = grad.len();
        let (i, v) =
            grad.iter().enumerate().fold(
                (dim, f64::MAX),
                |acc, (i, &g)| {
                    if acc.1 > g { (i, g) } else { acc }
                },
            );
        assert!(
            i < dim && v < f64::MAX,
            "failed to call linear minimization oracle."
        );
        let mut minimizer = vec![0f64; dim];
        minimizer[i] = 1f64;
        minimizer
    }

    /// Generate an initial point `e₁ ∈ [-1, +1]ⁿ`, a basis vector.
    fn initial_point_l2(dim: usize) -> Vec<f64> {
        let mut point = vec![0f64; dim];
        point[0] = 1f64;
        point
    }

    /// Generate an initial point `x₁ ∈ [0, +1]ⁿ`, a basis vector.
    /// In order not to take infinite value, we choose the following one
    /// as the initial point.
    /// ```txt
    ///     x₁[i] = 1 - (n-1)ε,     if i = 1,
    ///             ε,              if i ≠ 1.
    /// ```
    fn initial_point_entropy(dim: usize) -> Vec<f64> {
        const EPS: f64 = 1e-9;
        let mut point = vec![EPS; dim];
        point[0] = 1f64 - (dim - 1) as f64 * EPS;
        point
    }

    /// Returns the optimal solution to the test optimization problem
    fn optimal_solution_l2(dim: usize) -> Vec<f64> {
        vec![0f64; dim]
    }

    /// Returns the optimal solution to the test optimization problem
    fn optimal_solution_entropy(dim: usize) -> Vec<f64> {
        vec![1f64 / dim as f64; dim]
    }

    fn stopping_criterion(nxt: &[f64], cur: &[f64], grad: &[f64]) -> bool {
        let duality_gap = nxt
            .iter()
            .zip(&cur[..])
            .zip(&grad[..])
            .map(|((n, c), g)| (c - n) * g)
            .sum::<f64>();
        duality_gap < TEST_TOLERANCE
    }

    #[test]
    fn classic_frank_wolfe_l2_objective() {
        let dim = 100usize;
        let f = L2::new();
        let mut x = initial_point_l2(dim);
        let mut algo = FrankWolfe::new(L2::new(), FwUpdateRule::Classic);

        loop {
            let g = f.gradient(&x[..]);
            let v = linear_minimization_oracle_l2(&g[..]);
            if stopping_criterion(&v[..], &x[..], &g[..]) {
                break;
            }
            x = algo.next(x, v);
        }

        let xoptimal = optimal_solution_l2(dim);
        let optval = f.objective_value(&xoptimal[..]);
        let objval = f.objective_value(&x[..]);

        assert!(
            objval - optval < TEST_TOLERANCE,
            "expected {optval}, got {objval}."
        );
    }

    #[test]
    fn shortstep_frank_wolfe_l2_objective() {
        let dim = 100usize;
        let f = L2::new();
        let mut x = initial_point_l2(dim);
        let mut algo = FrankWolfe::new(L2::new(), FwUpdateRule::ShortStep);

        loop {
            let g = f.gradient(&x[..]);
            let v = linear_minimization_oracle_l2(&g[..]);
            if stopping_criterion(&v[..], &x[..], &g[..]) {
                break;
            }
            x = algo.next(x, v);
        }

        let xoptimal = optimal_solution_l2(dim);
        let optval = f.objective_value(&xoptimal[..]);
        let objval = f.objective_value(&x[..]);

        assert!(
            objval - optval < TEST_TOLERANCE,
            "expected {optval}, got {objval}."
        );
    }

    #[test]
    fn line_search_frank_wolfe_l2_objective() {
        let dim = 100usize;
        let f = L2::new();
        let mut x = initial_point_l2(dim);
        let mut algo = FrankWolfe::new(L2::new(), FwUpdateRule::LineSearch);

        loop {
            let g = f.gradient(&x[..]);
            let v = linear_minimization_oracle_l2(&g[..]);
            if stopping_criterion(&v[..], &x[..], &g[..]) {
                break;
            }
            x = algo.next(x, v);
        }

        let xoptimal = optimal_solution_l2(dim);
        let optval = f.objective_value(&xoptimal[..]);
        let objval = f.objective_value(&x[..]);

        assert!(
            objval - optval < TEST_TOLERANCE,
            "expected {optval}, got {objval}."
        );
    }

    #[test]
    fn bpfw_frank_wolfe_l2_objective() {
        let dim = 100usize;
        let f = L2::new();
        let mut x = initial_point_l2(dim);
        let mut algo = FrankWolfe::new(L2::new(), FwUpdateRule::BlendedPairwise);

        loop {
            let g = f.gradient(&x[..]);
            let v = linear_minimization_oracle_l2(&g[..]);
            if stopping_criterion(&v[..], &x[..], &g[..]) {
                break;
            }
            x = algo.next(x, v);
        }

        let xoptimal = optimal_solution_l2(dim);
        let optval = f.objective_value(&xoptimal[..]);
        let objval = f.objective_value(&x[..]);

        assert!(
            objval - optval < TEST_TOLERANCE,
            "expected {optval}, got {objval}."
        );
    }

    #[test]
    fn classic_frank_wolfe_entropy_objective() {
        let dim = 100usize;
        let f = Entropy::new();
        let mut x = initial_point_entropy(dim);
        let mut algo = FrankWolfe::new(Entropy::new(), FwUpdateRule::Classic);

        loop {
            let g = f.gradient(&x[..]);
            let v = linear_minimization_oracle_entropy(&g[..]);
            if stopping_criterion(&v[..], &x[..], &g[..]) {
                break;
            }
            x = algo.next(x, v);
        }

        let xoptimal = optimal_solution_entropy(dim);
        let optval = f.objective_value(&xoptimal[..]);
        let objval = f.objective_value(&x[..]);

        assert!(
            objval - optval < TEST_TOLERANCE,
            "expected {optval}, got {objval}."
        );
    }

    #[test]
    fn line_search_frank_wolfe_entropy_objective() {
        let dim = 100usize;
        let f = Entropy::new();
        let mut x = initial_point_entropy(dim);
        let mut algo = FrankWolfe::new(Entropy::new(), FwUpdateRule::LineSearch);

        loop {
            let g = f.gradient(&x[..]);
            let v = linear_minimization_oracle_entropy(&g[..]);
            if stopping_criterion(&v[..], &x[..], &g[..]) {
                break;
            }
            x = algo.next(x, v);
        }

        let xoptimal = optimal_solution_entropy(dim);
        let optval = f.objective_value(&xoptimal[..]);
        let objval = f.objective_value(&x[..]);

        assert!(
            objval - optval < TEST_TOLERANCE,
            "expected {optval}, got {objval}."
        );
    }

    #[test]
    fn bpfw_frank_wolfe_entropy_objective() {
        let dim = 100usize;
        let f = Entropy::new();
        let mut x = initial_point_entropy(dim);
        let mut algo = FrankWolfe::new(Entropy::new(), FwUpdateRule::BlendedPairwise);

        loop {
            let g = f.gradient(&x[..]);
            let v = linear_minimization_oracle_entropy(&g[..]);
            if stopping_criterion(&v[..], &x[..], &g[..]) {
                break;
            }
            x = algo.next(x, v);
        }

        let xoptimal = optimal_solution_entropy(dim);
        let optval = f.objective_value(&xoptimal[..]);
        let objval = f.objective_value(&x[..]);

        assert!(
            objval - optval < TEST_TOLERANCE,
            "expected {optval}, got {objval}."
        );
    }
}
