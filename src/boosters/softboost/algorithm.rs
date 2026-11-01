//! Implementation of [`SoftBoost`] (Warmuth, Glocer, and Rätsch, 2007,
//! "Boosting Algorithms for Maximizing the Soft Margin").
//!
//! See [`SoftBoost`] for the algorithm, guarantees, and limitations.
use crate::boosters::softboost::solver::SoftBoostSolver;
use crate::core::CurrentHypothesis;
use crate::core::{
    Booster, Classifier, Sample, WeakLearner,
    constants::{DEFAULT_CAPPING, DEFAULT_TOLERANCE},
    tools::checkers,
    tools::helpers,
};
use crate::hypotheses::WeightedMajority;
use crate::optimization::soft_margin_optimization;

use std::ops::ControlFlow;

/// A totally corrective booster that approximately maximizes the soft margin
/// by entropy projections with capped example weights.
///
/// # Algorithm
///
/// This follows Algorithm 2 of Warmuth, Glocer, and Rätsch (2007). Let
/// `u^q_i = y_i h_q(x_i)`, let `δ` be [`SoftBoost::tolerance`], and let `ν`
/// be [`SoftBoost::nu`]. Starting from the uniform distribution `d^0` and
/// `γ̂_0 = 1`, round `t`:
///
/// 1. Sends `d^{t-1}` to the weak learner and receives `h_t`.
/// 2. Updates the smallest edge seen so far:
///    ```txt
///    γ̂_t = min{ γ̂_{t-1}, Σ_i d^{t-1}_i u^t_i }
///    ```
/// 3. Projects the uniform distribution onto the constraints of all
///    hypotheses received so far (minimum relative entropy to uniform):
///    ```txt
///    d^t = argmin_d Σ_i d_i ln(m d_i)
///          s.t. Σ_i d_i u^q_i ≤ γ̂_t - δ,  q = 1, ..., t,
///               Σ_i d_i = 1,  0 ≤ d_i ≤ 1/ν.
///    ```
/// 4. Stops if this problem is infeasible or if `d^t` contains an entry equal
///    to zero (step 3(c) of the paper).
///
/// After stopping, the hypothesis weights are obtained by solving the
/// soft-margin LP over all collected hypotheses (see
/// [`soft_margin_optimization`](crate::optimization::soft_margin_optimization())), and the result is
/// returned as a [`WeightedMajority`] (exactly-zero weights are dropped).
///
/// Implementation choices:
///
/// - The projection is solved with the Clarabel interior-point solver using
///   exponential cones, instead of the sequential quadratic programming
///   suggested in the paper. A solve counts as successful only if the solver
///   reports `Solved` and the returned distribution passes independent checks
///   (finite values, bounds, sum to one, and edge constraints, each within
///   `1e-7`). Negative round-off is replaced with `0.0`, which can trigger the
///   zero-entry stopping rule; positive entries are not rounded.
/// - A `PrimalInfeasible` status is the paper's infeasibility stop. Any other
///   solver status (for example `AlmostSolved` or `MaxIterations`) or a failed
///   check stops boosting early and is recorded in
///   [`SoftBoost::numerical_stop_reason`]; see `# Limitations`.
/// - Boosting also stops after [`SoftBoost::max_loop`] rounds, the paper's
///   iteration bound. It is at least one so that a hypothesis is always
///   obtained, even when `ν = m`.
///
/// # Theoretical guarantee
///
/// Assume the weak learner returns, for every distribution with
/// `d ≤ (1/ν) 1`, a hypothesis with edge at least `g`, and that the
/// projections are solved exactly. Theorem 2 of the paper shows that SoftBoost
/// terminates after at most `⌈2 ln(m/ν) / δ²⌉` iterations with a convex
/// combination whose soft-margin objective (LP (1) of the paper) is at most
/// `δ` below `g`. Since `g` can be at most the optimal soft margin over the
/// whole hypothesis class, this is `δ`-optimal when the weak learner returns a
/// hypothesis with maximum edge. With `ν = 1` the bound coincides with the
/// bound for TotalBoost (see [`crate::boosters::totalboost::TotalBoost`]).
///
/// # Limitations
///
/// - The weak-learning assumption is not checked at runtime. If it fails (for
///   example, with a depth-limited decision tree), the guarantee does not
///   apply.
/// - `tolerance` bounds a gap in the soft-margin objective, not the
///   classification error.
/// - `nu` caps every example weight at `1/ν`, with `1 ≤ ν ≤ m`. It does not
///   guarantee a particular number of misclassified training examples.
/// - On an entropy-solver accuracy failure, boosting stops early with a
///   warning on standard error, and [`SoftBoost::numerical_stop_reason`]
///   returns the reason. Such a stop does not certify convergence; the output
///   is the LP-weighted combination of the hypotheses collected so far
///   (including the latest one).
/// - The final soft-margin LP panics if it does not reach an accurate solution.
/// - Each round solves a conic program with `2m` variables and
///   `O(m + t)` constraints, built from scratch; this can be slow for large
///   samples. Earlier hypotheses are re-evaluated on the sample every round.
/// - Binary labels must be `-1` and `+1`.
///
/// # Parameters
///
/// | Method | Meaning | Default |
/// | --- | --- | --- |
/// | [`SoftBoost::tolerance`] | accuracy `δ > 0` | `1e-3` |
/// | [`SoftBoost::nu`] | capping parameter `ν ∈ [1, m]` | `1.0` (no capping) |
///
/// # Reference
///
/// Manfred K. Warmuth, Karen A. Glocer, and Gunnar Rätsch.
/// [Boosting Algorithms for Maximizing the Soft Margin](https://proceedings.neurips.cc/paper/2007/file/cfbce4c1d7c425baf21d6b6f2babe6be-Paper.pdf).
/// Advances in Neural Information Processing Systems 20 (NeurIPS 2007).
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
/// // `nu = 2` caps every example weight at 1/2; `tolerance` is the
/// // soft-margin accuracy δ.
/// let mut booster = SoftBoost::init(&train).nu(2.0).tolerance(0.1);
/// // Budget ceil(2 ln(m/ν) / δ²) = ceil(2 ln 4 / 0.01) = 278 rounds.
/// assert_eq!(booster.max_loop(), 278);
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
/// println!("SoftBoost test accuracy: {accuracy}");
///
/// // An entropy-solver accuracy failure stops training early without
/// // certifying convergence; check it before trusting the result.
/// if let Some(reason) = booster.numerical_stop_reason() {
///     eprintln!("SoftBoost stopped early: {reason}");
/// }
/// assert!(!f.hypotheses.is_empty() && f.hypotheses.len() <= 278);
/// // The final LP weights are nonnegative and normalized.
/// assert!(f.weights.iter().all(|&w| w > 0.0));
/// assert!((f.weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
/// ```
pub struct SoftBoost<'a, H> {
    sample: &'a Sample,

    pub(crate) dist: Vec<f64>,

    // Smallest edge observed so far, `γ̂_t = min_{q ≤ t} Σ_i d^{q-1}_i y_i h_q(x_i)`.
    gamma_hat: f64,
    tolerance: f64,
    nu: f64,

    solver: SoftBoostSolver<'a>,

    hypotheses: Vec<H>,

    max_iter: usize,
    terminated: usize,

    weights: Vec<f64>,
    numerical_stop_reason: Option<String>,
}

impl<'a, H> SoftBoost<'a, H>
where
    H: Classifier,
{
    /// Constructs a new `SoftBoost` for `sample` with
    /// `tolerance = 1e-3` and `nu = 1.0`.
    ///
    /// # Panics
    ///
    /// Panics if `sample` has no examples.
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
    /// let mut booster = SoftBoost::init(&sample).tolerance(0.1);
    /// let f = booster.run(&learner);
    /// assert_eq!(f.predict_all(&sample).len(), 6);
    /// ```
    pub fn init(sample: &'a Sample) -> Self {
        let n_examples = sample.shape().0;
        assert_ne!(n_examples, 0);

        SoftBoost {
            sample,

            gamma_hat: 1f64,
            tolerance: DEFAULT_TOLERANCE,
            nu: DEFAULT_CAPPING,

            solver: SoftBoostSolver::new(sample),

            dist: Vec::new(),
            weights: Vec::new(),
            numerical_stop_reason: None,
            hypotheses: Vec::new(),

            max_iter: usize::MAX,
            terminated: usize::MAX,
        }
    }

    /// Returns the reason boosting stopped early because of an entropy-solver
    /// accuracy failure, or `None` if no such failure occurred.
    ///
    /// A `Some` value means convergence was not certified; the final model
    /// is still fitted to the hypotheses collected before stopping. `None`
    /// does not imply zero training error. The value is reset by
    /// [`Booster::preprocess`].
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
    /// let mut booster = SoftBoost::init(&sample).tolerance(0.1);
    /// assert!(booster.numerical_stop_reason().is_none()); // not trained yet
    /// let f = booster.run(&learner);
    /// match booster.numerical_stop_reason() {
    ///     Some(reason) => eprintln!("convergence not certified: {reason}"),
    ///     None => assert!(!f.hypotheses.is_empty()),
    /// }
    /// ```
    pub fn numerical_stop_reason(&self) -> Option<&str> {
        self.numerical_stop_reason.as_deref()
    }

    /// Sets the capping parameter `nu`, which caps each example weight at
    /// `1 / nu`. Larger values allow more examples to violate the margin.
    ///
    /// # Panics
    ///
    /// Panics unless `1 ≤ nu ≤ m`, where `m` is the number of examples.
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
    /// // m = 6: every example weight is at most 1/3.
    /// let mut booster = SoftBoost::init(&sample).nu(3.0).tolerance(0.1);
    /// let f = booster.run(&learner);
    /// assert!(!f.hypotheses.is_empty());
    /// ```
    #[inline(always)]
    pub fn nu(mut self, nu: f64) -> Self {
        let n_examples = self.sample.shape().0;
        checkers::capping_parameter(nu, n_examples);

        self.nu = nu;
        self
    }

    /// Sets the accuracy parameter `δ` (the `tolerance`).
    ///
    /// Each projection requires every collected hypothesis to have edge at
    /// most `γ̂_t - δ`. This is a soft-margin objective gap, not a
    /// classification-error bound.
    ///
    /// # Panics
    ///
    /// Panics unless `tolerance` is finite and positive.
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
    /// let mut booster = SoftBoost::init(&sample).tolerance(0.5);
    /// // Budget ceil(2 ln 6 / 0.5²) = 15 rounds.
    /// assert_eq!(booster.max_loop(), 15);
    /// let f = booster.run(&learner);
    /// assert!(f.hypotheses.len() <= 15);
    /// ```
    #[inline(always)]
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        assert!(
            tolerance.is_finite() && tolerance > 0.0,
            "SoftBoost tolerance must be finite and positive"
        );
        self.tolerance = tolerance;
        self
    }

    fn initialize_solver(&mut self) {
        self.solver.initialize(self.nu);
    }

    /// Returns the iteration budget `max(⌈2 ln(m / ν) / δ²⌉, 1)`.
    ///
    /// This is the iteration bound for the soft-margin objective gap
    /// (Theorem 2 in Warmuth, Glocer, and Rätsch, 2007), not a bound on
    /// classification error. The paper's bound is zero when `ν = m`; the
    /// budget is at least one so that a hypothesis is always obtained.
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
    /// use miniboosts::DecisionTreeClassifier;
    ///
    /// // The hypothesis type must be known before training.
    /// let booster = SoftBoost::<DecisionTreeClassifier>::init(&sample)
    ///     .nu(2.0)
    ///     .tolerance(0.5);
    /// // m = 6: ceil(2 ln(6/2) / 0.5²) = ceil(8.79) = 9.
    /// assert_eq!(booster.max_loop(), 9);
    /// // With ν = m the paper's bound is zero; the budget is still one round.
    /// let booster = SoftBoost::<DecisionTreeClassifier>::init(&sample).nu(6.0);
    /// assert_eq!(booster.max_loop(), 1);
    /// ```
    pub fn max_loop(&self) -> usize {
        let m = self.sample.shape().0 as f64;

        let ln_m = (m / self.nu).ln();
        // Even at nu=m, obtain one hypothesis before forming the output.
        ((2f64 * ln_m / self.tolerance.powi(2)).ceil() as usize).max(1)
    }
}

impl<H> SoftBoost<'_, H>
where
    H: Classifier,
{
    /// Updates `self.dist`.
    /// Returns `None` if a stopping criterion is satisfied.
    fn update_params_mut(&mut self) -> Option<()> {
        match self
            .solver
            .solve(self.gamma_hat, self.tolerance, &self.hypotheses[..])
        {
            Ok(Some(())) => {
                self.dist = self.solver.distribution_on_examples();
                if self.dist.iter().any(|&d| d == 0.0) {
                    return None;
                }
                Some(())
            }
            Ok(None) => None,
            Err(reason) => {
                eprintln!(
                    "[WARN] SoftBoost/TotalBoost stopped early: {reason}. Returning a model from collected hypotheses; convergence is not certified."
                );
                self.numerical_stop_reason = Some(reason);
                None
            }
        }
    }
}

impl<H> Booster<H> for SoftBoost<'_, H>
where
    H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;

    fn name(&self) -> &str {
        "SoftBoost"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_examples, n_feature) = self.sample.shape();
        let ratio = self.nu * 100f64 / n_examples as f64;
        let nu = helpers::format_unit(self.nu);
        let info = Vec::from([
            ("# of examples", format!("{n_examples}")),
            ("# of features", format!("{n_feature}")),
            ("Tolerance", format!("{}", self.tolerance)),
            ("Max iteration", format!("{}", self.max_iter)),
            ("Capping (outliers)", format!("{nu} ({ratio: >7.3} %)")),
        ]);
        Some(info)
    }

    fn preprocess(&mut self) {
        self.sample.is_valid_binary_instance();
        let n_examples = self.sample.shape().0;

        let uni = 1.0 / n_examples as f64;

        self.dist = vec![uni; n_examples];

        self.max_iter = self.max_loop();
        self.terminated = self.max_iter;
        self.hypotheses = Vec::new();

        self.gamma_hat = 1.0;
        self.numerical_stop_reason = None;
        self.initialize_solver();
    }

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize) -> ControlFlow<usize>
    where
        W: WeakLearner<Hypothesis = H>,
    {
        if self.max_iter < iteration {
            return ControlFlow::Break(self.max_iter);
        }

        // Receive a hypothesis from the base learner
        let h = weak_learner.produce(self.sample, &self.dist);

        let edge = helpers::edge(self.sample, &self.dist, &h);
        self.gamma_hat = self.gamma_hat.min(edge);

        self.hypotheses.push(h);

        // Update the parameters
        if self.update_params_mut().is_none() {
            self.terminated = iteration;
            return ControlFlow::Break(self.terminated);
        }

        ControlFlow::Continue(())
    }

    fn postprocess(&mut self) -> Self::Output {
        let (_, weights) = soft_margin_optimization(self.nu, self.sample, &self.hypotheses[..]);
        self.weights = weights;
        WeightedMajority::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}

impl<H> CurrentHypothesis for SoftBoost<'_, H>
where
    H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        // With no hypotheses the soft-margin LP (Σ w = 1) is infeasible;
        // the snapshot before the first round is the empty ensemble.
        if self.hypotheses.is_empty() {
            return WeightedMajority::from_slices(&[], &[]);
        }
        let (_, weights) = soft_margin_optimization(self.nu, self.sample, &self.hypotheses[..]);
        WeightedMajority::from_slices(&weights[..], &self.hypotheses[..])
    }
}

#[cfg(test)]
mod endpoint_tests {
    use super::*;

    #[derive(Clone)]
    struct Oracle;
    impl Classifier for Oracle {
        fn confidence(&self, sample: &Sample, row: usize) -> f64 {
            [1.0, 0.0, -1.0][row] * sample.target()[row]
        }
    }
    impl WeakLearner for Oracle {
        type Hypothesis = Self;
        fn produce(&self, _: &Sample, dist: &[f64]) -> Self {
            assert!(dist.iter().all(|d| (*d - 1.0 / 3.0).abs() < 1e-12));
            Self
        }
    }
    #[test]
    fn full_capping_obtains_a_hypothesis_before_stopping() {
        let sample = Sample::dummy(3);
        let mut booster = SoftBoost::init(&sample).nu(3.0);
        let output = booster.run(&Oracle);
        assert_eq!(output.hypotheses.len(), 1);
        assert!((output.weights[0] - 1.0).abs() < 1e-7);
        assert!(output.confidence(&sample, 0).is_finite());
    }
}
