//! The [`Booster`] trait: the booster's side of the boosting protocol.

use crate::core::WeakLearner;
use std::ops::ControlFlow;

/// The booster's side of the boosting protocol.
///
/// Boosting is a repeated game between a **booster** and a
/// **weak learner** ([`WeakLearner`]). In each round `t = 1, 2, ...`,
/// 1. the booster chooses a probability distribution `d_t` over the
///    training examples, and
/// 2. the weak learner returns a hypothesis `h_t` that is (hopefully) better
///    than random guessing with respect to `d_t`.
///
/// After sufficiently many rounds, the booster outputs a combined hypothesis.
/// What "sufficiently many" means and what the combined hypothesis achieves
/// (training error, margin, soft margin) depend on the algorithm and on its
/// weak-learning assumption; neither is checked by this trait.
///
/// # Run protocol
/// [`Booster::run`] is provided and performs:
/// ```txt
/// preprocess()
/// for t = 1, 2, 3, ...:
///     if boost(weak_learner, t) is ControlFlow::Break(_): stop
/// return postprocess()
/// ```
/// External drivers such as [`crate::Logger`] follow the same protocol but
/// may also stop early (time or iteration limits) and inspect intermediate
/// models via [`crate::core::CurrentHypothesis`].
///
/// # Implementor contract
/// - [`Booster::preprocess`] must (re)initialize all per-run state
///   (distribution, ensemble, iteration budget) and validate the sample, so
///   that `run` can be called more than once and each call starts fresh.
/// - [`Booster::boost`] is called with `iteration = 1, 2, ...`. It performs
///   at most one round and returns `ControlFlow::Break(_)` when the algorithm
///   stops (stopping criterion met, budget exhausted, or no further progress
///   possible). **It must eventually return `Break`**; otherwise `run` never
///   returns. Callers do not call `boost` again after `Break`.
/// - The `Break` payload is informational (boosters in this crate return the
///   round at which they stopped or their iteration budget); `run` ignores it.
/// - Distributions passed to the weak learner should be nonnegative, finite,
///   and sum to one. Non-finite values or solver failures must not be
///   reported as successful results: panic, or stop and leave the ensemble
///   in a well-defined state.
/// - [`Booster::postprocess`] builds the final hypothesis from the rounds
///   performed. It may be called even if `Break` was never returned
///   (e.g., when an external driver stops early).
/// - Boosters must remain independent of concrete weak learners: they may
///   rely only on the [`WeakLearner`] interface and the hypothesis type `H`.
///
/// # Required methods
/// - [`Booster::name`]
/// - [`Booster::preprocess`]
/// - [`Booster::boost`]
/// - [`Booster::postprocess`]
///
/// # Provided methods
/// - [`Booster::info`] (returns `None` by default)
/// - [`Booster::run`]
///
/// # Examples
/// A minimal custom booster. It is **not** a real boosting algorithm: it
/// keeps the uniform distribution in every round and outputs the uniform
/// average of the `n_rounds` hypotheses. It only illustrates the protocol.
/// ```
/// use std::io::BufReader;
/// use std::ops::ControlFlow;
/// use miniboosts::prelude::*;
/// use miniboosts::{Sample, WeightedMajority};
///
/// struct UniformAverage<'a, H> {
///     sample: &'a Sample,
///     n_rounds: usize,
///     dist: Vec<f64>,
///     hypotheses: Vec<H>,
/// }
///
/// impl<'a, H> UniformAverage<'a, H> {
///     fn init(sample: &'a Sample, n_rounds: usize) -> Self {
///         assert!(n_rounds > 0, "the output needs at least one hypothesis");
///         Self { sample, n_rounds, dist: Vec::new(), hypotheses: Vec::new() }
///     }
/// }
///
/// impl<H: Classifier + Clone> Booster<H> for UniformAverage<'_, H> {
///     type Output = WeightedMajority<H>;
///
///     fn name(&self) -> &str {
///         "Uniform average"
///     }
///
///     fn info(&self) -> Option<Vec<(&str, String)>> {
///         Some(vec![("# of rounds", self.n_rounds.to_string())])
///     }
///
///     // Validate the sample and reset all per-run state.
///     fn preprocess(&mut self) {
///         self.sample.is_valid_binary_instance();
///         let n_sample = self.sample.shape().0;
///         self.dist = vec![1.0 / n_sample as f64; n_sample];
///         self.hypotheses.clear();
///     }
///
///     // One round; `Break` once the budget is used up.
///     fn boost<W>(&mut self, weak_learner: &W, iteration: usize) -> ControlFlow<usize>
///     where
///         W: WeakLearner<Hypothesis = H>,
///     {
///         if iteration > self.n_rounds {
///             return ControlFlow::Break(self.n_rounds);
///         }
///         let h = weak_learner.produce(self.sample, &self.dist);
///         self.hypotheses.push(h);
///         ControlFlow::Continue(())
///     }
///
///     // Equal weights, normalized to sum to one by `from_slices`.
///     fn postprocess(&mut self) -> Self::Output {
///         let weights = vec![1.0; self.hypotheses.len()];
///         WeightedMajority::from_slices(&weights, &self.hypotheses)
///     }
/// }
///
/// let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
/// let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
///     .unwrap()
///     .set_target("class");
/// let tree = DecisionTreeBuilder::new(&sample).max_depth(1).build();
///
/// let mut booster = UniformAverage::init(&sample, 3);
/// let f = booster.run(&tree);
/// assert_eq!(f.hypotheses.len(), 3);
/// assert_eq!(f.weights, vec![1.0 / 3.0; 3]);
///
/// // `run` starts afresh because `preprocess` resets the state.
/// let f = booster.run(&tree);
/// assert_eq!(f.hypotheses.len(), 3);
/// // The same separating stump is returned in every round.
/// assert_eq!(f.predict_all(&sample), vec![-1, -1, -1, 1, 1, 1]);
/// ```
pub trait Booster<H> {
    /// The final hypothesis output by the boosting algorithm,
    /// typically a weighted combination of the weak hypotheses `H`.
    type Output;

    /// Returns the name of the boosting algorithm, used in logs.
    fn name(&self) -> &str;

    /// Returns human-readable `(label, value)` pairs describing the booster's
    /// configuration (e.g., tolerance, capping parameter), used in logs.
    /// Returns `None` by default.
    fn info(&self) -> Option<Vec<(&str, String)>> {
        None
    }

    /// Runs the boosting algorithm with `weak_learner` and returns the
    /// combined hypothesis.
    ///
    /// Calls [`Booster::preprocess`], then [`Booster::boost`] with
    /// `iteration = 1, 2, ...` until it returns `ControlFlow::Break`, then
    /// [`Booster::postprocess`]. Does not return if `boost` never breaks.
    ///
    /// # Panics
    /// Propagates any panic raised by the booster or the weak learner, e.g.,
    /// for invalid labels or solver failures.
    ///
    /// # Examples
    /// Train AdaBoost with decision stumps and evaluate it on held-out data.
    /// ```
    /// # use std::io::BufReader;
    /// use miniboosts::prelude::*;
    /// use miniboosts::Sample;
    ///
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// let (train, test) = sample.split([0, 3, 1, 2, 4, 5], 0, 2);
    /// let tree = DecisionTreeBuilder::new(&train).max_depth(1).build();
    /// let mut booster = AdaBoost::init(&train).tolerance(0.1);
    /// let f = booster.run(&tree);
    ///
    /// assert_eq!(f.predict_all(&test), vec![-1, 1]);
    /// assert!(f.confidence_all(&test).iter().all(|c| (-1.0..=1.0).contains(c)));
    /// ```
    ///
    /// The same protocol written out by hand, as external drivers such as
    /// [`crate::Logger`] do:
    /// ```
    /// # use std::io::BufReader;
    /// use miniboosts::prelude::*;
    /// use miniboosts::Sample;
    ///
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// let tree = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    /// let mut booster = AdaBoost::init(&sample);
    ///
    /// booster.preprocess();
    /// let mut iteration = 1;
    /// while booster.boost(&tree, iteration).is_continue() {
    ///     iteration += 1;
    /// }
    /// let f = booster.postprocess();
    ///
    /// // A separating stump has edge 1, so AdaBoost stops in round 1.
    /// assert_eq!(iteration, 1);
    /// assert_eq!(f.predict_all(&sample), vec![-1, -1, -1, 1, 1, 1]);
    /// ```
    fn run<W>(&mut self, weak_learner: &W) -> Self::Output
    where
        W: WeakLearner<Hypothesis = H>,
    {
        self.preprocess();

        let _ = (1..).try_for_each(|iter| self.boost(weak_learner, iter));

        self.postprocess()
    }

    /// Initializes per-run state before the first round.
    /// Called once at the start of [`Booster::run`].
    fn preprocess(&mut self);

    /// Performs boosting round `iteration` (1-based): obtains a hypothesis
    /// from `weak_learner` for the current distribution and updates the
    /// booster's state.
    ///
    /// Returns `ControlFlow::Continue(())` to request another round, or
    /// `ControlFlow::Break(k)` to stop, where `k` is an informational round
    /// count.
    fn boost<W>(&mut self, weak_learner: &W, iteration: usize) -> ControlFlow<usize>
    where
        W: WeakLearner<Hypothesis = H>;

    /// Builds the final combined hypothesis from the rounds performed.
    fn postprocess(&mut self) -> Self::Output;
}
