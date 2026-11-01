//! Implementation of [`LpBoost`], the column-generation soft-margin booster
//! of Demiriz, Bennett, and Shawe-Taylor (2002).
//!
//! See the [`LpBoost`] struct for the algorithm, guarantees, and an example.
use crate::core::CurrentHypothesis;
use crate::core::{
    Booster, Classifier, Sample, WeakLearner,
    constants::{DEFAULT_CAPPING, DEFAULT_TOLERANCE},
    tools::checkers,
    tools::helpers,
};
use crate::hypotheses::WeightedMajority;
use crate::optimization::{ColumnGeneration, soft_margin_optimization};

use std::ops::ControlFlow;

/// LPBoost: totally corrective soft-margin boosting by column generation.
///
/// # Algorithm
///
/// LPBoost solves the soft-margin linear program over the hypotheses
/// collected so far and uses the optimal dual solution as the next
/// distribution over examples. With `m` examples, capping parameter `ν`,
/// and hypotheses `h_1, ..., h_t`, the restricted problem is
/// ```txt
/// max_{ρ, w, ξ}  ρ - (1/ν) Σ_i ξ_i
/// s.t.           y_i Σ_j w_j h_j(x_i) ≥ ρ - ξ_i,   i = 1, ..., m,
///                w ∈ simplex,  ξ ≥ 0,
/// ```
/// whose dual is the edge minimization over the capped simplex
/// ```txt
/// γ*_t = min_{d ∈ Δ_{m,ν}} max_{j ≤ t} Σ_i d_i y_i h_j(x_i),
/// Δ_{m,ν} = { d : Σ_i d_i = 1, 0 ≤ d_i ≤ 1/ν }.
/// ```
/// Starting from the uniform distribution `d_1` and `γ̂ = 1`, round `t`:
///
/// 1. obtains `h_t` from the weak learner on `d_t`;
/// 2. updates `γ̂ ← min(γ̂, Σ_i d_{t,i} y_i h_t(x_i))`, the smallest edge
///    returned so far;
/// 3. adds `h_t` as a new column, solves the restricted LP to get `γ*_t`,
///    and sets `d_{t+1}` to its optimal dual solution;
/// 4. stops if `γ*_t ≥ γ̂ - tolerance` (the `γ̂` rule follows the LPBoost
///    description in Warmuth, Glocer, and Rätsch, NeurIPS 2007).
///
/// The hypothesis received in the stopping round is kept. After stopping,
/// [`Booster::postprocess`] re-solves the same soft-margin LP over all
/// collected hypotheses with residual and duality-gap checks and returns the
/// resulting [`WeightedMajority`]. There is no iteration budget: training
/// continues until the stopping rule holds.
///
/// Implementation notes: each LP is solved from scratch by the Clarabel
/// interior-point solver through [`ColumnGeneration`]; negative primal
/// weights caused by roundoff are clipped to zero. The LP value is asserted to
/// be finite and within `[-1, 1]` up to
/// [`NUMERIC_TOLERANCE`](crate::core::constants::NUMERIC_TOLERANCE).
///
/// # Theoretical guarantee
///
/// The restricted value `γ*_t` never exceeds the optimal soft-margin value
/// `g*` over the whole hypothesis class. If the weak learner returns a
/// hypothesis of maximum edge on every distribution, each returned edge is at
/// least `g*`, so `γ̂ ≥ g*`; hence at termination
/// `γ*_t ≥ g* - tolerance`, i.e. the output is a `tolerance`-accurate solution
/// of the soft-margin problem (assuming the LPs are solved exactly).
///
/// No iteration bound polynomial in `1/tolerance` is known for LPBoost.
/// Warmuth, Glocer, and Rätsch (NeurIPS 2007) show instances on which it
/// needs `Ω(m)` iterations. LPBoost is often fast in practice; see
/// [`ErlpBoost`](crate::ErlpBoost) and [`MlpBoost`](crate::MlpBoost) for
/// variants with an `O(ln(m/ν)/ε²)` iteration bound.
///
/// # Limitations
///
/// - The optimality certificate assumes a maximum-edge weak learner; this is
///   not checked at runtime. With a heuristic weak learner such as a decision
///   tree, `γ̂` can underestimate the best achievable edge and the run can
///   stop before reaching the soft-margin optimum.
/// - `tolerance` is an optimization gap on the soft-margin objective, not a
///   bound on training or test error.
/// - `nu` caps every example weight at `1 / nu`; it does not guarantee a
///   particular number of margin errors.
/// - Labels must be `-1` or `+1` with both classes present; preprocessing
///   panics otherwise.
/// - Solver failures **panic**: the per-round LP must report `Solved` or
///   `AlmostSolved` and yield a valid capped distribution, and the final LP
///   must additionally pass residual and duality-gap checks.
/// - Each round solves an LP with `m + t + 1` variables, so the per-round cost
///   grows with the number of hypotheses.
/// - [`CurrentHypothesis::current_hypothesis`] (used by loggers) uses the
///   weights of the last per-round LP; the final output re-solves the same
///   problem, so the two agree up to solver accuracy.
///
/// # Parameters
///
/// | Builder method | Meaning | Default |
/// | --- | --- | --- |
/// | [`nu`](LpBoost::nu) | capping parameter `ν ∈ [1, m]` | `1` ([`DEFAULT_CAPPING`]) |
/// | [`tolerance`](LpBoost::tolerance) | stopping gap `γ̂ - γ*_t` | `1e-3` ([`DEFAULT_TOLERANCE`]) |
///
/// # Reference
///
/// Ayhan Demiriz, Kristin P. Bennett, and John Shawe-Taylor.
/// [Linear Programming Boosting via Column Generation](https://link.springer.com/content/pdf/10.1023/A:1012470815092.pdf).
/// Machine Learning 46, 2002.
///
/// The stopping rule and the `Ω(m)` lower bound: Manfred K. Warmuth,
/// Karen A. Glocer, and Gunnar Rätsch.
/// [Boosting Algorithms for Maximizing the Soft Margin](https://proceedings.neurips.cc/paper/2007/file/cfbce4c1d7c425baf21d6b6f2babe6be-Paper.pdf).
/// NeurIPS 2007.
///
/// # Examples
/// Train on a held-out split with decision stumps and evaluate on the test
/// part.
/// ```
/// use std::io::BufReader;
/// use miniboosts::{Sample, prelude::*};
///
/// // Labels follow the sign of `x + y`, so no single stump separates them.
/// let csv = b"x,y,class\n\
///     -2,1,-1\n-1,-2,-1\n1,-3,-1\n-3,2,-1\n0,-1,-1\n-2,-1,-1\n\
///     2,1,1\n1,2,1\n-1,3,1\n3,-1,1\n0,2,1\n2,2,1\n";
/// let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
///     .unwrap()
///     .set_target("class");
///
/// // A fixed permutation keeps the split reproducible:
/// // rows ix[0..4] form the test set, the other 8 rows the training set.
/// let ix = [0, 6, 4, 10, 1, 2, 3, 5, 7, 8, 9, 11];
/// let (train, test) = sample.split(&ix, 0, 4);
///
/// // The weak learner sees only the training split.
/// let learner = DecisionTreeBuilder::new(&train).max_depth(1).build();
///
/// // `nu = 2` caps every example weight at 1/2 (soft margin); training stops
/// // when the LP value is within `tolerance` of the smallest edge seen.
/// let mut booster = LpBoost::init(&train).nu(2.0).tolerance(0.01);
/// let f = booster.run(&learner);
///
/// let predictions = f.predict_all(&test);
/// assert!(predictions.iter().all(|&p| p == -1 || p == 1));
/// let correct = predictions
///     .iter()
///     .zip(test.target())
///     .filter(|&(&p, &y)| p as f64 == y)
///     .count();
/// let accuracy = correct as f64 / test.shape().0 as f64;
/// println!("LpBoost test accuracy: {accuracy}");
///
/// // `terminated` is the round at which the stopping rule held; it equals
/// // the number of collected hypotheses (the final LP may assign some of
/// // them weight zero, and those are dropped from `f`).
/// let rounds = booster.terminated();
/// assert!(rounds >= 1 && rounds != usize::MAX);
/// assert!(f.hypotheses.len() <= rounds);
/// // The LP weights are nonnegative and normalized.
/// assert!(f.weights.iter().all(|&w| w > 0.0));
/// assert!((f.weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
/// ```
pub struct LpBoost<'a, F> {
    // Training sample
    sample: &'a Sample,

    // Distribution over examples
    dist: Vec<f64>,

    // Smallest edge returned by the weak learner so far (`γ̂`)
    gamma_hat: f64,

    // Tolerance parameter
    tolerance: f64,

    // Number of examples
    n_examples: usize,

    // Capping parameter
    nu: f64,

    solver: ColumnGeneration<'a>,

    hypotheses: Vec<F>,
    weights: Vec<f64>,

    terminated: usize,
}

impl<'a, F> LpBoost<'a, F>
where
    F: Classifier,
{
    /// Constructs a new instance of `LpBoost` for `sample` with the default
    /// capping parameter `nu = 1` and `tolerance = 1e-3`.
    ///
    /// The sample is validated (binary `±1` labels) when training starts,
    /// not here.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{Sample, prelude::*};
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    /// // Defaults: hard margin (`nu = 1`) and `tolerance = 1e-3`.
    /// let mut booster = LpBoost::init(&sample);
    /// let f = booster.run(&learner);
    /// assert_eq!(f.predict_all(&sample).len(), 6);
    /// ```
    pub fn init(sample: &'a Sample) -> Self {
        let n_examples = sample.shape().0;

        Self {
            sample,

            dist: Vec::new(),
            gamma_hat: 1.0,
            tolerance: DEFAULT_TOLERANCE,
            n_examples,
            nu: DEFAULT_CAPPING,
            solver: ColumnGeneration::new(&sample),

            hypotheses: Vec::new(),
            weights: Vec::new(),

            terminated: usize::MAX,
        }
    }

    /// Sets the capping parameter `nu`.
    ///
    /// Every example weight is capped at `1 / nu`; `nu = 1` gives the
    /// hard-margin LP and larger values allow more margin violations.
    /// `nu` must lie in `[1, m]`, where `m` is the number of training
    /// examples. Default: `1`.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Panics
    /// Panics if `nu` is not in `[1, m]`.
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{Sample, prelude::*};
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    /// // m = 6: tolerate up to about 2 margin violations (weights ≤ 1/3).
    /// let mut booster = LpBoost::init(&sample).nu(3.0);
    /// let f = booster.run(&learner);
    /// assert!(!f.hypotheses.is_empty());
    /// ```
    pub fn nu(mut self, nu: f64) -> Self {
        checkers::capping_parameter(nu, self.n_examples);
        self.nu = nu;

        self
    }

    /// Sets the stopping tolerance.
    ///
    /// Training stops once the restricted LP value is within `tolerance` of
    /// the smallest edge returned so far. With a maximum-edge weak learner and
    /// exact LP solutions, the output is then a `tolerance`-accurate solution
    /// of the soft-margin problem. This is not a classification-error
    /// threshold. Default: `1e-3`.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Panics
    /// Panics if `tolerance` is not finite and strictly positive.
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{Sample, prelude::*};
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    /// let mut booster = LpBoost::init(&sample).tolerance(0.01);
    /// let f = booster.run(&learner);
    /// assert!(booster.terminated() >= 1);
    /// assert_eq!(f.predict_all(&sample).len(), 6);
    /// ```
    #[inline(always)]
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        assert!(
            tolerance.is_finite() && tolerance > 0.0,
            "tolerance must be finite and strictly positive"
        );
        self.tolerance = tolerance;
        self
    }

    /// Returns the round at which the stopping rule held, which equals the
    /// number of hypotheses collected.
    /// Returns `usize::MAX` before training and while the rule has not held.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{Sample, prelude::*};
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    /// let mut booster = LpBoost::init(&sample).tolerance(0.01);
    /// assert_eq!(booster.terminated(), usize::MAX); // not trained yet
    /// let f = booster.run(&learner);
    /// // The stopping rule held after `terminated()` rounds.
    /// assert!(f.hypotheses.len() <= booster.terminated());
    /// ```
    #[inline(always)]
    pub fn terminated(&self) -> usize {
        self.terminated
    }

    /// Adds `h` as a column, solves the restricted soft-margin LP, stores the
    /// optimal dual solution in `self.dist`, and returns the optimal value.
    ///
    /// Time complexity depends on the LP solver.
    #[inline(always)]
    fn update_distribution_mut(&mut self, h: &F) -> f64 {
        self.solver.solve(h);
        self.dist = self.solver.distribution_on_examples();

        self.solver.optimal_value()
    }
}

impl<F> Booster<F> for LpBoost<'_, F>
where
    F: Classifier + Clone,
{
    type Output = WeightedMajority<F>;

    fn name(&self) -> &str {
        "LpBoost"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_examples, n_feature) = self.sample.shape();
        let ratio = self.nu * 100f64 / n_examples as f64;
        let nu = helpers::format_unit(self.nu);
        let info = Vec::from([
            ("# of examples", format!("{n_examples}")),
            ("# of features", format!("{n_feature}")),
            ("Tolerance", format!("{}", self.tolerance)),
            ("Max iteration", "-".to_string()),
            ("Capping (outliers)", format!("{nu} ({ratio: >7.3} %)")),
        ]);
        Some(info)
    }

    fn preprocess(&mut self) {
        self.sample.is_valid_binary_instance();
        let n_examples = self.sample.shape().0;
        let uni = 1.0_f64 / self.n_examples as f64;

        self.n_examples = n_examples;
        self.dist = vec![uni; n_examples];
        self.gamma_hat = 1.0;
        self.solver.initialize(self.n_examples, self.nu);
        self.hypotheses = Vec::new();
        self.terminated = usize::MAX;
    }

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize) -> ControlFlow<usize>
    where
        W: WeakLearner<Hypothesis = F>,
    {
        let h = weak_learner.produce(self.sample, &self.dist);

        // Edge of `h` on the current distribution.
        let ghat = helpers::edge(self.sample, &self.dist[..], &h);

        self.gamma_hat = ghat.min(self.gamma_hat);

        let gamma_star = self.update_distribution_mut(&h);
        self.hypotheses.push(h);
        // The LP optimum is in [-1, 1] for confidence-rated hypotheses,
        // but the numerical solver can exceed a boundary by roundoff.
        let slack = crate::core::constants::NUMERIC_TOLERANCE;
        assert!(
            gamma_star.is_finite() && (-1.0 - slack..=1.0 + slack).contains(&gamma_star),
            "LP objective outside its numerical bounds: {gamma_star}"
        );

        if gamma_star >= self.gamma_hat - self.tolerance {
            self.terminated = self.hypotheses.len();
            return ControlFlow::Break(iteration);
        }

        ControlFlow::Continue(())
    }

    fn postprocess(&mut self) -> Self::Output {
        let (_, weights) = soft_margin_optimization(self.nu, self.sample, &self.hypotheses[..]);
        self.weights = weights;
        WeightedMajority::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}

impl<H> CurrentHypothesis for LpBoost<'_, H>
where
    H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        let weights = self.solver.weights_on_hypotheses();

        WeightedMajority::from_slices(&weights[..], &self.hypotheses[..])
    }
}

#[cfg(test)]
mod parameter_tests {
    use super::*;

    struct Hypothesis;
    impl Classifier for Hypothesis {
        fn confidence(&self, _: &Sample, _: usize) -> f64 {
            0.0
        }
    }

    #[test]
    fn tolerance_rejects_nonpositive_and_nonfinite_values() {
        let sample = Sample::dummy(2);
        for value in [0.0, -0.0, -0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                std::panic::catch_unwind(|| {
                    LpBoost::<Hypothesis>::init(&sample).tolerance(value);
                })
                .is_err(),
                "accepted invalid tolerance {value}"
            );
        }
    }

    #[test]
    fn tolerance_preserves_positive_finite_values() {
        let sample = Sample::dummy(2);
        for value in [f64::MIN_POSITIVE, 0.25, 2.0, f64::MAX] {
            let booster = LpBoost::<Hypothesis>::init(&sample).tolerance(value);
            assert_eq!(booster.tolerance, value);
        }
    }
}
