//! Implementation of [`AdaBoost`] (Freund and Schapire, 1997).
//!
//! See the [`AdaBoost`] struct for the update rule, guarantees, and an example.
use rayon::prelude::*;

use crate::core::CurrentHypothesis;
use crate::core::{
    Booster, Classifier, Sample, WeakLearner, constants::DEFAULT_TOLERANCE, tools::helpers,
};
use crate::hypotheses::WeightedMajority;

use std::ops::ControlFlow;

/// AdaBoost: exponential-weight boosting by Freund and Schapire.
///
/// # Algorithm
/// Let `m` be the number of training examples. Labels are `y_i ∈ {-1, +1}`
/// and weak hypotheses return confidences
/// `h_t(x_i) ∈ [-1, 1]`. Starting from the uniform distribution
/// `d_{1,i} = 1/m`, round `t` does the following.
///
/// 1. Call the weak learner on `d_t` to obtain `h_t`.
/// 2. Compute the edge and the hypothesis weight
///    ```txt
///    γ_t = Σ_i d_{t,i} y_i h_t(x_i),
///    α_t = atanh(γ_t) = (1/2) ln((1 + γ_t) / (1 - γ_t)).
///    ```
///    For `{-1, +1}`-valued hypotheses with weighted error `ε_t`,
///    `γ_t = 1 - 2ε_t` and `α_t = (1/2) ln((1 - ε_t) / ε_t)`.
/// 3. Update the distribution
///    ```txt
///    d_{t+1,i} ∝ d_{t,i} exp(-α_t y_i h_t(x_i)).
///    ```
///
/// The output is the [`WeightedMajority`] `sign(Σ_t α_t h_t(x))`
/// with weights normalized by [`WeightedMajority::from_slices`].
///
/// This matches the original algorithm up to parametrization: Freund and
/// Schapire multiply the weight of each correctly classified example by
/// `β_t = ε_t / (1 - ε_t)` and weight `h_t` by `ln(1 / β_t)`; after
/// normalization this gives the same distributions and the same final vote
/// for binary hypotheses. For confidence-rated hypotheses, the choice
/// `α_t = atanh(γ_t)` is the one analyzed by Schapire and Singer (1999).
///
/// Implementation choices (not part of the original method):
/// - **Log-space update.** The booster stores `ln d_{t,i}` and applies
///   `ln d_{t+1,i} = ln d_{t,i} - α_t y_i h_t(x_i)` before normalizing with a
///   shifted log-sum-exp. An example whose probability underflows to `0.0`
///   keeps a finite log weight, so it can regain mass in later rounds.
///   `α_t` is also computed from the log weights, so rounding of small
///   probabilities does not produce a spurious edge of `±1`.
/// - **Perfect hypotheses.** If every example with nonzero (log-space) mass
///   has margin exactly `+1` (or exactly `-1`), `α_t` would be infinite.
///   The booster then stops and returns `h_t` alone with weight `+1`
///   (or `-1`), discarding earlier hypotheses.
/// - **Negative edges.** A hypothesis with `γ_t < 0` is kept with the
///   negative weight `α_t`, which is equivalent to adding `-h_t`.
/// - **Iteration budget.** Training stops after
///   ```txt
///   T = max(1, ceil(ln m / tolerance²))
///   ```
///   rounds (see [`AdaBoost::max_loop`]), or after the count set by
///   [`AdaBoost::force_quit_at`]. The training error is never measured; there
///   is no other stopping test.
///
/// # Theoretical guarantee
/// Freund and Schapire show that, if the weak hypotheses have errors
/// `ε_t = 1/2 - g_t` on the distributions `d_t`, the training error of the
/// combined hypothesis after `T` rounds is at most
/// ```txt
/// Π_t 2 sqrt(ε_t (1 - ε_t)) = Π_t sqrt(1 - γ_t²) ≤ exp(-Σ_t γ_t² / 2),
/// ```
/// where `γ_t = 2 g_t` is the edge. The same bound holds for confidences in
/// `[-1, 1]` with `α_t = atanh(γ_t)` (Schapire and Singer, 1999).
/// If every edge satisfies `γ_t ≥ γ > 0`, the training error is below `1/m`,
/// hence zero, once `T > 2 ln m / γ²`.
///
/// The budget `ceil(ln m / tolerance²)` coincides with this count when
/// `tolerance = γ / √2` (that is, `√2` times the advantage `g = γ/2`).
/// Thus, if the weak learner achieves edge at least `√2 · tolerance` on
/// every distribution it receives, the full budget suffices for zero training
/// error. The rounding up and the one-round minimum are budget policies, not
/// extra guarantees.
///
/// # Limitations
/// - The edge assumption above is not checked at runtime. A decision tree of
///   a given depth does not by itself guarantee any edge.
/// - `tolerance` is neither a training-error threshold nor a margin; it only
///   sets the number of rounds. The default `tolerance = 1e-3` gives a budget
///   of about `10⁶ · ln m` rounds, so set `tolerance` or `force_quit_at`
///   explicitly in practice.
/// - AdaBoost does not maximize the margin and is sensitive to label noise;
///   see [`AdaBoostV`](crate::AdaBoostV) and the soft-margin boosters.
/// - Labels must be `-1` or `+1` and both classes must be present
///   (checked by [`Sample::is_valid_binary_instance`]).
///   Margins `y_i h_t(x_i)` outside `[-1, 1]` cause a panic.
/// - Each round costs `O(m)` time beyond the weak learner, and the booster
///   stores every hypothesis.
///
/// # Parameters
/// | Builder method | Default | Meaning |
/// | --- | --- | --- |
/// | [`tolerance`](AdaBoost::tolerance) | `1e-3` ([`DEFAULT_TOLERANCE`]) | Sets the budget `ceil(ln m / tolerance²)`. |
/// | [`force_quit_at`](AdaBoost::force_quit_at) | none | Replaces the budget with an exact round limit. |
///
/// # Reference
/// Yoav Freund and Robert E. Schapire.
/// *A Decision-Theoretic Generalization of On-Line Learning and an
/// Application to Boosting.* Journal of Computer and System Sciences,
/// 55(1):119–139, 1997.
/// <https://www.sciencedirect.com/science/article/pii/S002200009791504X>
///
/// Robert E. Schapire and Yoram Singer.
/// *Improved Boosting Algorithms Using Confidence-rated Predictions.*
/// Machine Learning, 37(3):297–336, 1999.
///
/// # Examples
/// Train on a held-out split with decision stumps and evaluate on the test
/// part. The labels follow the sign of `x + y`, so no single stump separates
/// them and the ensemble needs several rounds.
/// ```
/// use std::io::BufReader;
/// use miniboosts::{Sample, prelude::*};
///
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
/// // `tolerance = 0.1` gives a budget of ceil(ln 8 / 0.01) = 208 rounds;
/// // `force_quit_at` replaces it with exactly 20 rounds.
/// let mut booster = AdaBoost::init(&train).tolerance(0.1).force_quit_at(20);
/// assert_eq!(booster.max_loop(), 208);
/// let f = booster.run(&learner);
///
/// // At most 20 hypotheses; the weights are normalized by `from_slices`.
/// assert!(!f.weights.is_empty() && f.weights.len() <= 20);
/// assert!(f.weights.iter().all(|w| w.is_finite()));
///
/// // Predictions are labels in {-1, +1}; confidences lie in [-1, 1].
/// let predictions = f.predict_all(&test);
/// assert!(predictions.iter().all(|&p| p == -1 || p == 1));
/// assert!(f.confidence_all(&test).iter().all(|c| (-1.0..=1.0).contains(c)));
///
/// let correct = predictions
///     .iter()
///     .zip(test.target())
///     .filter(|&(&p, &y)| p as f64 == y)
///     .count();
/// let accuracy = correct as f64 / test.shape().0 as f64;
/// println!("AdaBoost test accuracy: {accuracy}");
/// ```
///
/// Boosters depend only on the [`WeakLearner`] and [`Classifier`] traits,
/// so any weak learner works. Here is a minimal decision stump that picks
/// the feature, threshold, and orientation with the largest weighted edge
/// `Σ_i d_i y_i h(x_i)`.
/// ```
/// use std::io::BufReader;
/// use miniboosts::{Sample, prelude::*};
///
/// /// `h(x) = sign` if `x[feature] > threshold`, and `-sign` otherwise.
/// #[derive(Clone)]
/// struct Stump {
///     feature: usize,
///     threshold: f64,
///     sign: f64,
/// }
///
/// impl Classifier for Stump {
///     fn confidence(&self, sample: &Sample, row: usize) -> f64 {
///         let x = sample.features()[self.feature][row];
///         if x > self.threshold { self.sign } else { -self.sign }
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
///         let (n_examples, n_features) = sample.shape();
///         let y = sample.target();
///         let mut best = Stump { feature: 0, threshold: f64::NEG_INFINITY, sign: 1.0 };
///         let mut best_edge = f64::NEG_INFINITY;
///         for feature in 0..n_features {
///             let column = &sample.features()[feature];
///             // Candidate thresholds: every observed value of the feature.
///             for row in 0..n_examples {
///                 let threshold = column[row];
///                 // Edge of `x > threshold => +1`; the flipped stump has `-edge`.
///                 let edge: f64 = (0..n_examples)
///                     .map(|i| {
///                         let h = if column[i] > threshold { 1.0 } else { -1.0 };
///                         dist[i] * y[i] * h
///                     })
///                     .sum();
///                 for (sign, e) in [(1.0, edge), (-1.0, -edge)] {
///                     if e > best_edge {
///                         best_edge = e;
///                         best = Stump { feature, threshold, sign };
///                     }
///                 }
///             }
///         }
///         best
///     }
/// }
///
/// let csv = b"x,y,class\n\
///     -2,1,-1\n-1,-2,-1\n1,-3,-1\n-3,2,-1\n0,-1,-1\n-2,-1,-1\n\
///     2,1,1\n1,2,1\n-1,3,1\n3,-1,1\n0,2,1\n2,2,1\n";
/// let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
///     .unwrap()
///     .set_target("class");
///
/// // The booster code is unchanged; only the weak learner differs.
/// let mut booster = AdaBoost::init(&sample).tolerance(0.1).force_quit_at(10);
/// let f = booster.run(&StumpLearner);
///
/// // `f` is a `WeightedMajority<Stump>`.
/// let predictions = f.predict_all(&sample);
/// assert_eq!(predictions.len(), 12);
/// assert!(f.hypotheses.len() <= 10);
/// ```
pub struct AdaBoost<'a, F> {
    // Training sample
    sample: &'a Sample,

    // Distribution on sample.
    dist: Vec<f64>,

    // Keep finite log mass even when exp(log_mass) underflows to zero.
    log_dist: Vec<f64>,

    // Tolerance parameter
    tolerance: f64,

    // Weights on hypotheses in `hypotheses`
    weights: Vec<f64>,

    // Hypotheses obtained by the weak-learner.
    hypotheses: Vec<F>,

    // Iteration budget; see `max_loop`.
    max_iter: usize,

    // Optional. If this value is `Some(it)`,
    // the algorithm terminates after `it` iterations.
    force_quit_at: Option<usize>,

    // Terminated iteration.
    // AdaBoost terminates early if a hypothesis has edge exactly +1 or -1.
    terminated: usize,
}

impl<'a, F> AdaBoost<'a, F> {
    /// Constructs a new instance of `AdaBoost` for `sample`.
    ///
    /// The tolerance defaults to [`DEFAULT_TOLERANCE`] and no forced round
    /// limit is set. The sample is validated when training starts.
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
    /// // Always configure the budget; the default tolerance `1e-3` allows
    /// // about `10⁶ · ln m` rounds.
    /// let mut booster = AdaBoost::init(&sample).force_quit_at(5);
    /// let f = booster.run(&learner);
    /// assert_eq!(f.predict_all(&sample).len(), 6);
    /// ```
    #[inline]
    pub fn init(sample: &'a Sample) -> Self {
        Self {
            sample,

            tolerance: DEFAULT_TOLERANCE,

            dist: Vec::new(),
            log_dist: Vec::new(),
            weights: Vec::new(),
            hypotheses: Vec::new(),

            force_quit_at: None,
            max_iter: usize::MAX,
            terminated: usize::MAX,
        }
    }

    /// Returns the configured iteration budget, `ceil(ln(m) / tolerance²)`,
    /// with at least one round for a nonempty sample. Budgets exceeding
    /// `usize::MAX` saturate at that value.
    /// This is not a measured training-error stopping condition. Convergence
    /// guarantees require the weak-learning assumptions in the source method.
    /// See [Freund and Schapire, 1997](https://www.sciencedirect.com/science/article/pii/S002200009791504X).
    /// Rounding up and the one-round minimum are implementation budget policies,
    /// not additional convergence guarantees. A singleton budget can be queried,
    /// but training still requires the core's valid binary sample contract.
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
    /// let booster = AdaBoost::<DecisionTreeClassifier>::init(&sample).tolerance(0.5);
    /// // m = 6: ceil(ln 6 / 0.25) = ceil(7.17) = 8.
    /// assert_eq!(booster.max_loop(), 8);
    /// // `force_quit_at` does not change the reported budget.
    /// assert_eq!(booster.force_quit_at(3).max_loop(), 8);
    /// ```
    pub fn max_loop(&self) -> usize {
        let n_sample = self.sample.shape().0 as f64;

        let budget = (n_sample.ln() / self.tolerance.powi(2)).ceil() as usize;
        if n_sample > 0.0 {
            budget.max(1)
        } else {
            budget
        }
    }

    /// Runs exactly `it` rounds unless a perfect hypothesis stops training
    /// earlier. This replaces the budget of [`AdaBoost::max_loop`], so `it`
    /// may be smaller or larger than it. `it = 0` yields an empty ensemble.
    /// If `it` is smaller than the bound in the struct documentation,
    /// the returned hypothesis has no training-error guarantee.
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
    /// // Exactly 3 rounds, unless a perfect hypothesis stops training earlier
    /// // (on this separable sample the first stump is already perfect).
    /// let mut booster = AdaBoost::init(&sample).force_quit_at(3);
    /// let f = booster.run(&learner);
    /// assert!(f.hypotheses.len() <= 3);
    /// ```
    pub fn force_quit_at(mut self, it: usize) -> Self {
        self.force_quit_at = Some(it);
        self
    }

    /// Sets the parameter used to compute the iteration budget
    /// `ceil(ln m / tolerance²)`; see [`AdaBoost::max_loop`].
    /// Training error is not checked against this value each round.
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
    /// // Budget ceil(ln 6 / 0.2²) = 45 rounds.
    /// let mut booster = AdaBoost::init(&sample).tolerance(0.2);
    /// assert_eq!(booster.max_loop(), 45);
    /// let f = booster.run(&learner);
    /// assert!(f.hypotheses.len() <= 45);
    /// ```
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        assert!(
            tolerance.is_finite() && tolerance > 0.0,
            "tolerance must be finite and strictly positive"
        );
        self.tolerance = tolerance;
        self
    }

    /// Applies the exponential update for a hypothesis with the given
    /// margins and weight `edge_coefficient = atanh(edge)`, and returns that
    /// weight. Updates `self.log_dist` in log space and refreshes `self.dist`
    /// from it, so underflowed probabilities keep their log mass.
    ///
    /// Time complexity: `O(m)` for `m` training examples.
    #[inline]
    fn update_params(&mut self, margins: Vec<f64>, edge_coefficient: f64) -> f64 {
        // Compute the weight on new hypothesis.
        // This is the returned value of this function.
        assert!(edge_coefficient.is_finite());
        let weight = edge_coefficient;

        // This is the original exponential update in log space. Never recover
        // logs from rounded probabilities: an underflowed mass can grow later.
        assert_eq!(margins.len(), self.log_dist.len());
        self.log_dist
            .par_iter_mut()
            .zip(margins)
            .for_each(|(log_d, margin)| *log_d -= weight * margin);
        self.dist = helpers::normalize_log_weights(&mut self.log_dist);

        weight
    }
}

impl<F> Booster<F> for AdaBoost<'_, F>
where
    F: Classifier + Clone,
{
    type Output = WeightedMajority<F>;

    fn name(&self) -> &str {
        "AdaBoost"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_sample, n_feature) = self.sample.shape();
        let quit = if let Some(it) = self.force_quit_at {
            format!("At round {it}")
        } else {
            "-".to_string()
        };
        let info = Vec::from([
            ("# of examples", format!("{}", n_sample)),
            ("# of features", format!("{}", n_feature)),
            ("Tolerance", format!("{}", self.tolerance)),
            ("Max iteration", format!("{}", self.max_loop())),
            ("Force quit", quit),
        ]);
        Some(info)
    }

    fn preprocess(&mut self) {
        self.sample.is_valid_binary_instance();
        // Initialize parameters
        let n_sample = self.sample.shape().0;
        let uni = 1.0 / n_sample as f64;
        self.dist = vec![uni; n_sample];
        self.log_dist = vec![-(n_sample as f64).ln(); n_sample];

        self.weights = Vec::new();
        self.hypotheses = Vec::new();

        self.max_iter = self.max_loop();

        if let Some(it) = self.force_quit_at {
            self.max_iter = it;
        }
    }

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize) -> ControlFlow<usize>
    where
        W: WeakLearner<Hypothesis = F>,
    {
        if self.max_iter < iteration {
            return ControlFlow::Break(self.max_iter);
        }

        // Get a new hypothesis
        let h = weak_learner.produce(self.sample, &self.dist);

        // Each element in `margins` is the product of
        // the predicted vector and the correct vector
        let margins = helpers::margins(self.sample, &h).collect::<Vec<_>>();

        let coefficient = helpers::log_weighted_edge_coefficient(&self.log_dist, &margins);

        // Only exact perfect margins on the mathematical support terminate;
        // rounded probability zeros must not erase finite log mass.
        if coefficient.is_infinite() {
            self.terminated = iteration;
            self.weights = vec![coefficient.signum()];
            self.hypotheses = vec![h];
            return ControlFlow::Break(iteration);
        }

        // Compute the weight on the new hypothesis
        let weight = self.update_params(margins, coefficient);
        self.weights.push(weight);
        self.hypotheses.push(h);

        ControlFlow::Continue(())
    }

    fn postprocess(&mut self) -> Self::Output {
        WeightedMajority::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}

impl<H> CurrentHypothesis for AdaBoost<'_, H>
where
    H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        WeightedMajority::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}

#[cfg(test)]
mod log_weight_tests {
    use super::*;

    #[derive(Clone)]
    struct Hypothesis;
    impl Classifier for Hypothesis {
        fn confidence(&self, _: &Sample, _: usize) -> f64 {
            0.0
        }
    }

    #[test]
    fn budget_rounds_up_and_allows_a_round_for_nonempty_samples() {
        let sample = Sample::dummy(2);
        // ln(2) / 0.5² = 2.77258... must allow the third round.
        let mut booster = AdaBoost::<Hypothesis>::init(&sample).tolerance(0.5);
        assert_eq!(booster.max_loop(), 3);
        booster.preprocess();
        assert_eq!(booster.max_iter, 3);
        assert_eq!(AdaBoost::<()>::init(&sample).tolerance(2.0).max_loop(), 1);
        let singleton = Sample::dummy(1);
        assert_eq!(AdaBoost::<()>::init(&singleton).max_loop(), 1);
        assert_eq!(
            AdaBoost::<()>::init(&sample)
                .tolerance(f64::MIN_POSITIVE)
                .max_loop(),
            usize::MAX
        );
    }

    #[test]
    fn explicit_iteration_override_is_preserved() {
        let sample = Sample::dummy(2);
        for rounds in [0, 1, 10] {
            let mut booster = AdaBoost::<Hypothesis>::init(&sample)
                .tolerance(0.5)
                .force_quit_at(rounds);
            booster.preprocess();
            assert_eq!(booster.max_iter, rounds);
        }
    }

    #[test]
    fn underflowed_mass_recovers_and_preprocessing_resets_logs() {
        let sample = Sample::dummy(2);
        let mut booster = AdaBoost::<Hypothesis>::init(&sample).tolerance(0.25);
        booster.preprocess();
        // Exercise the update kernel with a fixed finite coefficient and
        // opposite margin sequences. Their products cancel exactly in theory.
        for _ in 0..1600 {
            booster.update_params(vec![1.0, -1.0], 0.5f64.atanh());
        }
        assert_eq!(booster.dist[0], 0.0);
        assert!(booster.log_dist[0].is_finite());
        for _ in 0..1600 {
            booster.update_params(vec![-1.0, 1.0], 0.5f64.atanh());
        }
        // Roundoff accumulates over 3200 updates; no mass should be lost.
        assert!((booster.dist[0] - 0.5).abs() < 1e-10, "{:?}", booster.dist);
        booster.preprocess();
        assert_eq!(booster.dist, vec![0.5; 2]);
        assert_eq!(booster.log_dist, vec![-2f64.ln(); 2]);
    }

    #[test]
    fn recurrence_matches_direct_exponential_weights() {
        let sample = Sample::dummy(4);
        let mut booster = AdaBoost::<Hypothesis>::init(&sample).tolerance(0.25);
        booster.preprocess();
        let mut reference = vec![0.25; 4];
        for margins in [[1.0, 1.0, 1.0, -1.0], [-1.0, 1.0, 1.0, 1.0]] {
            let edge: f64 = margins.iter().zip(&reference).map(|(m, d)| m * d).sum();
            let alpha = edge.atanh();
            let actual_alpha = booster.update_params(margins.to_vec(), edge.atanh());
            assert!((actual_alpha - alpha).abs() < 1e-14);
            for (d, margin) in reference.iter_mut().zip(margins) {
                *d *= (-alpha * margin).exp();
            }
            let sum: f64 = reference.iter().sum();
            reference.iter_mut().for_each(|d| *d /= sum);
            for (actual, expected) in booster.dist.iter().zip(&reference) {
                assert!((actual - expected).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn common_log_shift_does_not_change_the_update() {
        let sample = Sample::dummy(3);
        let mut first = AdaBoost::<Hypothesis>::init(&sample).tolerance(0.25);
        let mut shifted = AdaBoost::<Hypothesis>::init(&sample).tolerance(0.25);
        first.preprocess();
        shifted.preprocess();
        first.log_dist = vec![-2.0, -1.0, 0.0];
        shifted.log_dist = vec![998.0, 999.0, 1000.0];
        first.update_params(vec![1.0, -1.0, 1.0], 0.5f64.atanh());
        shifted.update_params(vec![1.0, -1.0, 1.0], 0.5f64.atanh());
        for (a, b) in first.dist.iter().zip(&shifted.dist) {
            assert!((a - b).abs() < 1e-13);
        }
    }
    #[derive(Clone)]
    struct RecoversMass;
    impl Classifier for RecoversMass {
        fn confidence(&self, sample: &Sample, row: usize) -> f64 {
            [-1.0, 1.0][row] * sample.target()[row]
        }
    }
    impl WeakLearner for RecoversMass {
        type Hypothesis = Self;
        fn produce(&self, _: &Sample, dist: &[f64]) -> Self {
            assert_eq!(dist, &[0.0, 1.0]);
            Self
        }
    }

    #[test]
    fn public_boost_recovers_underflowed_misclassified_mass() {
        let sample = Sample::dummy(2);
        let mut booster = AdaBoost::init(&sample).tolerance(0.25);
        booster.preprocess();
        booster.log_dist = vec![-1000.0, 0.0];
        booster.dist = helpers::softmax(&booster.log_dist);
        // The ordinary weighted edge rounds to +1 even though row zero is
        // misclassified and has positive mathematical mass exp(-1000).
        assert!(booster.boost(&RecoversMass, 1).is_continue());
        assert!(booster.weights[0].is_finite());
        assert!(booster.dist[0] > 0.1);
        assert!((booster.dist.iter().sum::<f64>() - 1.0).abs() < 1e-14);
    }
}

#[cfg(test)]
mod negative_coefficient_tests {
    use super::*;

    // Margins `y h(x)` per row; the sign pattern fixes the edge.
    #[derive(Clone)]
    struct Margins(&'static [f64]);
    impl Classifier for Margins {
        fn confidence(&self, sample: &Sample, row: usize) -> f64 {
            self.0[row] * sample.target()[row]
        }
    }
    impl WeakLearner for Margins {
        type Hypothesis = Self;
        fn produce(&self, _: &Sample, _: &[f64]) -> Self {
            self.clone()
        }
    }

    #[test]
    fn negative_edge_keeps_its_negative_coefficient() {
        let sample = Sample::dummy(4);
        // Uniform edge -0.5, so alpha = atanh(-0.5) < 0 and the
        // normalized ensemble is -h.
        let learner = Margins(&[-1.0, -1.0, -1.0, 1.0]);
        let f = AdaBoost::init(&sample).force_quit_at(1).run(&learner);
        assert_eq!(f.weights, vec![-1.0]);
        assert_eq!(f.predict_all(&sample), vec![1, 1, -1, 1]);
    }

    #[test]
    fn perfectly_wrong_hypothesis_terminates_with_its_negation() {
        let sample = Sample::dummy(4);
        let learner = Margins(&[-1.0; 4]);
        let mut booster = AdaBoost::init(&sample);
        let f = booster.run(&learner);
        assert_eq!(booster.terminated, 1);
        assert_eq!(f.weights, vec![-1.0]);
        assert_eq!(f.predict_all(&sample), vec![1, 1, -1, -1]);
    }
}

#[cfg(test)]
mod parameter_tests {
    use super::*;

    #[test]
    fn tolerance_rejects_nonpositive_and_nonfinite_values() {
        let sample = Sample::dummy(2);
        for value in [0.0, -0.0, -0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                std::panic::catch_unwind(|| {
                    AdaBoost::<()>::init(&sample).tolerance(value);
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
            let booster = AdaBoost::<()>::init(&sample).tolerance(value);
            assert_eq!(booster.tolerance, value);
        }
    }
}
