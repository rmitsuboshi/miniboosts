//! Implementation of [`TotalBoost`] (Warmuth, Liao, and Rätsch, 2006,
//! "Totally Corrective Boosting Algorithms that Maximize the Margin").
//!
//! See [`TotalBoost`] for the algorithm, guarantees, and limitations.
use crate::boosters::softboost::SoftBoost;
use crate::core::CurrentHypothesis;
use crate::core::{Booster, Classifier, Sample, WeakLearner};
use crate::hypotheses::WeightedMajority;

use std::ops::ControlFlow;

/// A totally corrective booster that approximately maximizes the hard margin.
///
/// # Algorithm
///
/// This type wraps [`SoftBoost`] with capping parameter `ν = 1`, under which
/// the capping constraint `d_i ≤ 1/ν` is vacuous and SoftBoost reduces to
/// TotalBoost (Warmuth, Glocer, and Rätsch, 2007, Section 4). With
/// `u^q_i = y_i h_q(x_i)` and `δ` = [`TotalBoost::tolerance`], round `t`:
///
/// 1. Sends `d^{t-1}` to the weak learner, receives `h_t`, and sets
///    `γ̂_t = min{ γ̂_{t-1}, Σ_i d^{t-1}_i u^t_i }` (with `γ̂_0 = 1`).
/// 2. Computes the relative-entropy projection of the uniform distribution:
///    ```txt
///    d^t = argmin_d Σ_i d_i ln(m d_i)
///          s.t. Σ_i d_i u^q_i ≤ γ̂_t - δ,  q = 1, ..., t,
///               Σ_i d_i = 1,  d_i ≥ 0.
///    ```
/// 3. Stops if the projection is infeasible or `d^t` has a zero entry.
///
/// The output weights maximize the margin over the collected hypotheses
/// (the soft-margin LP with `ν = 1`). Solver use, numerical checks, and the
/// iteration budget are those of [`SoftBoost`]; the budget is
/// `max(⌈2 ln(m) / δ²⌉, 1)` rounds.
///
/// # Theoretical guarantee
///
/// If the weak learner always returns a hypothesis with edge at least `g` and
/// the projections are solved exactly, TotalBoost terminates after at most
/// `⌈2 ln(m) / δ²⌉` iterations with a convex combination whose minimum
/// margin is at least `g - δ` (Warmuth, Liao, and Rätsch, 2006; restated as
/// the `ν = 1` case of Theorem 2 in Warmuth, Glocer, and Rätsch, 2007).
///
/// # Limitations
///
/// - The weak-learning assumption is not checked at runtime.
/// - `tolerance` bounds a margin gap, not the classification error.
/// - No capping is used, so noisy or non-separable data can concentrate all
///   weight on a few hard examples; consider [`SoftBoost`] with `nu > 1`.
/// - On an entropy-solver accuracy failure, boosting stops early with a
///   warning on standard error, and [`TotalBoost::numerical_stop_reason`]
///   returns the reason. Such a stop does not certify convergence. The final
///   LP panics if it does not reach an accurate solution.
/// - [`Booster::info`] reports SoftBoost's fields, including the capping value.
/// - Binary labels must be `-1` and `+1`.
///
/// # Parameters
///
/// | Method | Meaning | Default |
/// | --- | --- | --- |
/// | [`TotalBoost::tolerance`] | accuracy `δ > 0` | `1e-3` |
///
/// # Reference
///
/// Manfred K. Warmuth, Jun Liao, and Gunnar Rätsch.
/// [Totally Corrective Boosting Algorithms that Maximize the Margin](https://dl.acm.org/doi/10.1145/1143844.1143970).
/// Proceedings of the 23rd International Conference on Machine Learning
/// (ICML 2006).
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
/// // `tolerance` is the margin accuracy δ; the budget is
/// // ceil(2 ln(m) / δ²) = ceil(2 ln 8 / 0.01) = 416 rounds.
/// let mut booster = TotalBoost::init(&train).tolerance(0.1);
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
/// println!("TotalBoost test accuracy: {accuracy}");
///
/// // An entropy-solver accuracy failure stops training early without
/// // certifying convergence; check it before trusting the result.
/// if let Some(reason) = booster.numerical_stop_reason() {
///     eprintln!("TotalBoost stopped early: {reason}");
/// }
/// assert!(!f.hypotheses.is_empty() && f.hypotheses.len() <= 416);
/// // The hard-margin LP weights are nonnegative and normalized, so every
/// // training margin y_i f(x_i) lies in [-1, 1].
/// assert!(f.weights.iter().all(|&w| w > 0.0));
/// assert!((f.weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
/// let min_margin = f
///     .confidence_all(&train)
///     .iter()
///     .zip(train.target())
///     .map(|(c, y)| c * y)
///     .fold(f64::INFINITY, f64::min);
/// assert!((-1.0..=1.0).contains(&min_margin));
/// println!("minimum training margin: {min_margin}");
/// ```
pub struct TotalBoost<'a, H> {
    softboost: SoftBoost<'a, H>,
}

impl<'a, H> TotalBoost<'a, H>
where
    H: Classifier,
{
    /// Constructs a new `TotalBoost` for `sample` with `tolerance = 1e-3`.
    ///
    /// # Panics
    ///
    /// Panics if `sample` has no examples.
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
    /// let mut booster = TotalBoost::init(&sample).tolerance(0.1);
    /// let f = booster.run(&learner);
    /// assert_eq!(f.predict_all(&sample).len(), 6);
    /// ```
    pub fn init(sample: &'a Sample) -> Self {
        let softboost = SoftBoost::init(sample).nu(1f64);

        Self { softboost }
    }

    /// Returns the reason boosting stopped early because of an entropy-solver
    /// accuracy failure, or `None` if no such failure occurred.
    ///
    /// A `Some` value means convergence was not certified. `None` does not
    /// imply zero training error. See [`SoftBoost::numerical_stop_reason`].
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
    /// let mut booster = TotalBoost::init(&sample).tolerance(0.1);
    /// assert!(booster.numerical_stop_reason().is_none()); // not trained yet
    /// let f = booster.run(&learner);
    /// match booster.numerical_stop_reason() {
    ///     Some(reason) => eprintln!("convergence not certified: {reason}"),
    ///     None => assert!(!f.hypotheses.is_empty()),
    /// }
    /// ```
    pub fn numerical_stop_reason(&self) -> Option<&str> {
        self.softboost.numerical_stop_reason()
    }

    /// Sets the accuracy parameter `δ` (the `tolerance`), a margin gap rather
    /// than a classification-error bound.
    ///
    /// # Panics
    ///
    /// Panics unless `tol` is finite and positive.
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
    /// // Budget ceil(2 ln 6 / 0.5²) = 15 rounds.
    /// let mut booster = TotalBoost::init(&sample).tolerance(0.5);
    /// let f = booster.run(&learner);
    /// assert!(!f.hypotheses.is_empty() && f.hypotheses.len() <= 15);
    /// ```
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.softboost = self.softboost.tolerance(tol);
        self
    }
}

impl<H> Booster<H> for TotalBoost<'_, H>
where
    H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;

    fn name(&self) -> &str {
        "TotalBoost"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        self.softboost.info()
    }

    fn preprocess(&mut self) {
        self.softboost.preprocess();
    }

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize) -> ControlFlow<usize>
    where
        W: WeakLearner<Hypothesis = H>,
    {
        self.softboost.boost(weak_learner, iteration)
    }

    fn postprocess(&mut self) -> Self::Output {
        self.softboost.postprocess()
    }
}

impl<H> CurrentHypothesis for TotalBoost<'_, H>
where
    H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        self.softboost.current_hypothesis()
    }
}
