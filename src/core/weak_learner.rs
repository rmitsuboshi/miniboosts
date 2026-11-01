//! The [`WeakLearner`] trait: the weak learner's side of the boosting protocol.
use crate::core::Sample;

/// The weak learner's side of the boosting protocol.
///
/// Given the training sample and a distribution over its examples chosen by a
/// [`crate::Booster`], a weak learner returns a hypothesis of type
/// [`WeakLearner::Hypothesis`]. For classification this is a type implementing
/// [`crate::Classifier`] whose confidence values lie in `[-1, 1]`.
///
/// Boosting guarantees are typically stated under a *weak-learning
/// assumption*, e.g., that the returned hypothesis has edge
/// `Σ_i d_i y_i h(x_i) >= γ > 0` for **every** distribution `d` the booster
/// requests. Nothing in this trait enforces or checks that assumption; for
/// example, a depth-limited [`crate::DecisionTree`] need not satisfy it.
///
/// # Required methods
/// - [`WeakLearner::produce`]
///
/// # Provided methods
/// - [`WeakLearner::name`] (returns `"Not specified"` by default)
/// - [`WeakLearner::info`] (returns `None` by default)
///
/// # Examples
/// A custom weak learner: decision stumps `h(x) = s * sign(x_f - t)` that
/// maximize the edge over every feature `f`, every observed value `t`, and
/// both signs `s`.
/// ```
/// use std::io::BufReader;
/// use miniboosts::prelude::*;
/// use miniboosts::Sample;
/// use miniboosts::core::helpers;
///
/// #[derive(Clone, Debug)]
/// struct Stump {
///     feature: String,
///     threshold: f64,
///     sign: f64,
/// }
///
/// impl Classifier for Stump {
///     fn confidence(&self, sample: &Sample, row: usize) -> f64 {
///         let x = sample[&self.feature][row];
///         if x >= self.threshold { self.sign } else { -self.sign }
///     }
/// }
///
/// struct StumpLearner;
///
/// impl WeakLearner for StumpLearner {
///     type Hypothesis = Stump;
///
///     fn name(&self) -> &str {
///         "Decision stump"
///     }
///
///     fn produce(&self, sample: &Sample, dist: &[f64]) -> Stump {
///         let n_sample = sample.shape().0;
///         let mut best: Option<(f64, Stump)> = None;
///         for feature in sample.features() {
///             for row in 0..n_sample {
///                 let mut h = Stump {
///                     feature: feature.name().to_string(),
///                     threshold: feature[row],
///                     sign: 1.0,
///                 };
///                 // Flipping the sign negates the edge.
///                 let mut edge = helpers::edge(sample, dist, &h);
///                 if edge < 0.0 {
///                     h.sign = -1.0;
///                     edge = -edge;
///                 }
///                 if best.as_ref().is_none_or(|(e, _)| edge > *e) {
///                     best = Some((edge, h));
///                 }
///             }
///         }
///         best.expect("the sample has no examples or no features").1
///     }
/// }
///
/// let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
/// let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
///     .unwrap()
///     .set_target("class");
///
/// // Called directly with a distribution ...
/// let n_sample = sample.shape().0;
/// let uniform = vec![1.0 / n_sample as f64; n_sample];
/// let h = StumpLearner.produce(&sample, &uniform);
/// // `x >= 1` separates the classes, so the best stump has edge 1.
/// assert_eq!((h.feature.as_str(), h.threshold, h.sign), ("x", 1.0, 1.0));
/// // Tolerance: the edge sums six terms of 1/6, so it is 1 up to rounding.
/// assert!((helpers::edge(&sample, &uniform, &h) - 1.0).abs() < 1e-12);
///
/// // ... or plugged into any booster.
/// let f = AdaBoost::init(&sample).run(&StumpLearner);
/// assert_eq!(f.predict_all(&sample), vec![-1, -1, -1, 1, 1, 1]);
/// ```
pub trait WeakLearner {
    /// The hypothesis type returned by [`WeakLearner::produce`].
    type Hypothesis;

    /// Returns the name of the weak learning algorithm, used in logs.
    fn name(&self) -> &str {
        "Not specified"
    }

    /// Returns human-readable `(label, value)` pairs describing the weak
    /// learner's configuration, used in logs. Returns `None` by default.
    fn info(&self) -> Option<Vec<(&str, String)>> {
        None
    }

    /// Returns a hypothesis trained on `sample` weighted by `dist`.
    ///
    /// `dist[i]` is the weight of the `i`-th example of `sample`;
    /// boosters pass a probability vector (nonnegative entries summing to
    /// one) whose length equals the number of examples. For classification,
    /// the hypothesis should achieve a large weighted edge
    /// `Σ_i dist[i] y_i h(x_i)` (equivalently, small weighted error).
    /// Implementations may panic on malformed input; see each implementation.
    fn produce(&self, sample: &Sample, dist: &[f64]) -> Self::Hypothesis;
}
