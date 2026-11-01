//! Implementation of `AdaBoost*_ν` (Rätsch and Warmuth, 2005) as [`AdaBoostV`].
//!
//! `*` cannot appear in a Rust identifier, so the algorithm is named
//! `AdaBoostV`, a name also used in the SparsiBoost paper.
//! See the [`AdaBoostV`] struct for details and an example.
use rayon::prelude::*;

use crate::core::CurrentHypothesis;
use crate::core::{Booster, Classifier, Sample, WeakLearner, tools::helpers};
use crate::hypotheses::WeightedMajority;

use std::ops::ControlFlow;

/// `AdaBoost*_ν` by Rätsch and Warmuth: AdaBoost with an adaptive margin
/// estimate that approximately maximizes the hard margin.
///
/// # Algorithm
/// Let `m` be the number of training examples, `y_i ∈ {-1, +1}` the labels,
/// `h_t(x_i) ∈ [-1, 1]` the weak hypotheses' confidences, and
/// `ν = tolerance`. Starting from `d_{1,i} = 1/m`, round `t` follows
/// Algorithm 2 of the paper.
///
/// 1. Call the weak learner on `d_t` to obtain `h_t`.
/// 2. Compute the edge `γ_t = Σ_i d_{t,i} y_i h_t(x_i)`. If `|γ_t| = 1`,
///    stop and return `h_t` alone with weight `sign(γ_t)`, discarding
///    earlier hypotheses.
/// 3. Update the margin estimate and the hypothesis weight
///    ```txt
///    γ_t^min = min_{r ≤ t} γ_r,
///    ρ_t     = γ_t^min - ν,
///    α_t     = atanh(γ_t) - atanh(ρ_t)
///            = (1/2) ln((1 + γ_t) / (1 - γ_t)) - (1/2) ln((1 + ρ_t) / (1 - ρ_t)).
///    ```
///    Because `ρ_t < γ_t`, every `α_t` is positive.
/// 4. Update the distribution
///    ```txt
///    d_{t+1,i} ∝ d_{t,i} exp(-α_t y_i h_t(x_i)).
///    ```
///
/// The output is the [`WeightedMajority`] `sign(Σ_t α_t h_t(x))` with
/// normalized weights. Training stops after `floor(2 ln m / ν²)` rounds
/// (see [`AdaBoostV::max_loop`]) or after the count set by
/// [`AdaBoostV::force_quit_at`].
///
/// Implementation choices (numerical, not part of the original method):
/// - The distribution is kept in log space and normalized with a shifted
///   log-sum-exp, so an example whose probability underflows to `0.0` keeps
///   a finite log weight and can regain mass later.
/// - `atanh(γ_t)` is computed from the log weights, and the minimum edge is
///   tracked as `min_r atanh(γ_r)` (equivalent because `atanh` is
///   increasing). The term `atanh(ρ_t)` is evaluated with `ln_1p` and
///   log-space identities so it stays accurate when `γ_t^min` is close to
///   `±1`. The test `|γ_t| = 1` is exact on the log-space support, so
///   underflowed probabilities do not create a spurious perfect edge.
///
/// # Theoretical guarantee
/// Let `ρ*` be the maximum hard margin achievable by a convex combination of
/// hypotheses in the class. Theorem 6 of the paper states: if the weak
/// learner always returns a hypothesis with edge `γ_t ≥ ρ*`, then after
/// `2 ln m / ν²` iterations `AdaBoost*_ν` produces a combined hypothesis with
/// margin at least `ρ* - ν`. By the min-max theorem, an edge of `ρ*` is
/// always attainable by a weak learner that returns a hypothesis of maximum
/// edge over the class.
///
/// # Limitations
/// - The edge assumption `γ_t ≥ ρ*` is not checked at runtime. Separability
///   of the data or the choice of a decision tree does not establish it;
///   with a non-optimal weak learner the margin guarantee does not apply.
/// - `tolerance` is the margin accuracy `ν`, not a training-error threshold.
///   The default `ν = 1/m` gives a budget of about `2 m² ln m` rounds.
/// - The budget truncates `2 ln m / ν²` toward zero, so it can be one round
///   short of the paper's count when this value is not an integer. It is `0`
///   for `m = 1`.
/// - [`AdaBoostV::force_quit_at`] with a smaller count voids the guarantee.
/// - Training panics if `ρ_t ≤ -1` (that is, `ν ≥ 1 + γ_t^min`, which can
///   happen with large `ν` or negative edges), if a hypothesis weight is not
///   a finite `f64`, or if a margin lies outside `[-1, 1]`.
/// - Labels must be `-1` or `+1` and both classes must be present.
/// - Each round costs `O(m)` time beyond the weak learner, and every
///   hypothesis is stored.
///
/// # Parameters
/// | Builder method | Default | Meaning |
/// | --- | --- | --- |
/// | [`tolerance`](AdaBoostV::tolerance) | `1/m` | Margin accuracy `ν`; also sets the budget `floor(2 ln m / ν²)`. |
/// | [`force_quit_at`](AdaBoostV::force_quit_at) | none | Replaces the budget with an exact round limit. |
///
/// # Reference
/// Gunnar Rätsch and Manfred K. Warmuth.
/// *Efficient Margin Maximizing with Boosting.*
/// Journal of Machine Learning Research, 6:2131–2152, 2005.
/// <http://jmlr.org/papers/v6/ratsch05a.html>
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
/// // `tolerance` is the margin accuracy ν. The budget floor(2 ln 8 / 0.1²)
/// // = 415 rounds is replaced by exactly 20 rounds with `force_quit_at`.
/// let mut booster = AdaBoostV::init(&train).tolerance(0.1).force_quit_at(20);
/// assert_eq!(booster.max_loop(), 415);
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
/// println!("AdaBoostV test accuracy: {accuracy}");
///
/// // At most 20 hypotheses with finite weights; confidences lie in [-1, 1].
/// assert!(!f.hypotheses.is_empty() && f.hypotheses.len() <= 20);
/// assert!(f.weights.iter().all(|w| w.is_finite()));
/// assert!(f.confidence_all(&test).iter().all(|c| (-1.0..=1.0).contains(c)));
/// ```
pub struct AdaBoostV<'a, F> {
    // Training sample
    sample: &'a Sample,

    // Margin accuracy `ν`.
    tolerance: f64,

    // Current margin estimate `ρ_t = γ_t^min - ν`.
    rho: f64,

    // Minimum edge observed so far, `γ_t^min`.
    gamma: f64,

    // atanh is monotone and preserves the distance from rounded +/-1 edges.
    min_edge_coefficient: f64,

    // Distribution on sample.
    dist: Vec<f64>,

    // Keep finite log mass even when exp(log_mass) underflows to zero.
    log_dist: Vec<f64>,

    // Weights on hypotheses in `hypotheses`
    weights: Vec<f64>,

    // Hypotheses obtained by the weak learner.
    hypotheses: Vec<F>,

    // Iteration budget.
    max_iter: usize,

    // Optional. If this value is `Some(iteration)`,
    // the algorithm terminates after `iteration` iterations.
    force_quit_at: Option<usize>,

    terminated: usize,
}

impl<'a, F> AdaBoostV<'a, F> {
    /// Constructs a new instance of `AdaBoostV` for `sample`.
    ///
    /// The tolerance `ν` defaults to `1/m` for `m` examples, and no forced
    /// round limit is set. The sample is validated when training starts.
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
    /// // The default ν = 1/m gives about 2 m² ln m rounds; cap them.
    /// let mut booster = AdaBoostV::init(&sample).force_quit_at(5);
    /// let f = booster.run(&learner);
    /// assert_eq!(f.predict_all(&sample).len(), 6);
    /// ```
    #[inline]
    pub fn init(sample: &'a Sample) -> Self {
        let n_examples = sample.shape().0;
        let default_tolerance = 1.0 / n_examples as f64;
        Self {
            sample,

            tolerance: default_tolerance,
            rho: 1.0,
            gamma: 1.0,
            min_edge_coefficient: f64::INFINITY,

            dist: Vec::new(),
            log_dist: Vec::new(),
            weights: Vec::new(),
            hypotheses: Vec::new(),

            max_iter: usize::MAX,
            terminated: usize::MAX,
            force_quit_at: None,
        }
    }

    /// Sets the margin accuracy `ν` used in the margin estimate
    /// `ρ_t = γ_t^min - ν` and in the budget `floor(2 ln m / ν²)`.
    /// This is not a threshold on observed classification error.
    /// Training panics later if `ν ≥ 1 + γ_t^min` in some round.
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
    /// // ν = 0.5 gives a budget of floor(2 ln 6 / 0.25) = 14 rounds.
    /// let mut booster = AdaBoostV::init(&sample).tolerance(0.5);
    /// assert_eq!(booster.max_loop(), 14);
    /// let f = booster.run(&learner);
    /// assert!(f.hypotheses.len() <= 14);
    /// ```
    #[inline]
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        assert!(
            tolerance.is_finite() && tolerance > 0.0,
            "AdaBoostV tolerance must be finite and positive"
        );
        self.tolerance = tolerance;

        self
    }

    /// Returns the configured iteration budget, `floor(2 ln(m) / tolerance²)`
    /// for `m` training examples. Values exceeding `usize::MAX` saturate.
    /// Margin guarantees require the weak-learning assumptions of AdaBoostV;
    /// separability alone does not certify this particular weak learner.
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
    /// let booster = AdaBoostV::<DecisionTreeClassifier>::init(&sample).tolerance(0.5);
    /// // m = 6: floor(2 ln 6 / 0.5²) = floor(14.33) = 14.
    /// assert_eq!(booster.max_loop(), 14);
    /// ```
    #[inline]
    pub fn max_loop(&self) -> usize {
        let n_examples = self.sample.shape().0 as f64;

        (2.0 * n_examples.ln() / self.tolerance.powi(2)) as usize
    }

    /// Set the maximal number of rounds.
    #[inline]
    #[allow(unused)]
    pub(crate) fn set_max_loop(&mut self, max_loop: usize) {
        self.max_iter = max_loop;
    }

    /// Runs exactly `iteration` rounds unless a perfect hypothesis stops
    /// training earlier. This replaces the budget of
    /// [`AdaBoostV::max_loop`], so `iteration` may be smaller or larger.
    /// If it is smaller than the iteration bound for AdaBoostV,
    /// the returned hypothesis has no guarantee about the margin.
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
    /// // At most 3 rounds, fewer if a perfect hypothesis appears.
    /// let mut booster = AdaBoostV::init(&sample).tolerance(0.1).force_quit_at(3);
    /// let f = booster.run(&learner);
    /// assert!(f.hypotheses.len() <= 3);
    /// ```
    pub fn force_quit_at(mut self, iteration: usize) -> Self {
        self.force_quit_at = Some(iteration);
        self
    }

    /// Updates `γ_t^min` and `ρ_t`, computes
    /// `α_t = atanh(γ_t) - atanh(ρ_t)` from `edge_coefficient = atanh(γ_t)`,
    /// applies the exponential update in log space, and returns `α_t`.
    ///
    /// Panics if `ρ_t ≤ -1` or `α_t` is not finite.
    ///
    /// Time complexity: `O(m)` for `m` training examples.
    #[inline]
    fn update_params(&mut self, margins: Vec<f64>, edge_coefficient: f64) -> f64 {
        // Update edge & margin estimation parameters
        assert!(edge_coefficient.is_finite());
        self.min_edge_coefficient = self.min_edge_coefficient.min(edge_coefficient);
        self.gamma = self.min_edge_coefficient.tanh();
        self.rho = self.gamma - self.tolerance;

        // AdaBoost*: alpha = atanh(edge) - atanh(gamma - tolerance).
        // Evaluate the second term before rounding gamma or rho to +/-1.
        let (log_plus, log_minus) = log_one_plus_minus_tanh(self.min_edge_coefficient);
        let log_tolerance = self.tolerance.ln();
        // rho > -1 iff tolerance < 1+gamma. rho < 1 follows from tolerance>0.
        assert!(
            log_tolerance < log_plus,
            "AdaBoostV requires the estimated margin rho to lie in (-1, 1)"
        );
        let correction = if self.rho.abs() < 0.5 {
            // Preserve tiny corrections around zero without subtracting logs
            // of quantities close to one.
            0.5 * (self.rho.ln_1p() - (-self.rho).ln_1p())
        } else {
            let ratio_log = log_tolerance - log_plus;
            let log_complement = if ratio_log < -std::f64::consts::LN_2 {
                (-ratio_log.exp()).ln_1p()
            } else {
                (-ratio_log.exp_m1()).ln()
            };
            let log_one_plus_rho = log_plus + log_complement;
            let log_one_minus_rho = helpers::logaddexp(log_minus, log_tolerance);
            0.5 * log_one_plus_rho - 0.5 * log_one_minus_rho
        };
        let weight = edge_coefficient - correction;
        assert!(
            weight.is_finite(),
            "AdaBoostV coefficient exceeds the finite f64 range"
        );

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

// log(1 +/- tanh(a)), using log1p near zero and a softplus identity
// near the endpoints. Doubling |a| occurs only in a negative exponential;
// overflow there gives the correct limiting zero instead of inf/inf.
fn log_one_plus_minus_tanh(a: f64) -> (f64, f64) {
    if a.abs() < 0.5 {
        let gamma = a.tanh();
        return (gamma.ln_1p(), (-gamma).ln_1p());
    }
    let tail = (-2.0 * a.abs()).exp().ln_1p();
    let large = std::f64::consts::LN_2 - tail;
    let small = std::f64::consts::LN_2 - a.abs() - a.abs() - tail;
    if a >= 0.0 {
        (large, small)
    } else {
        (small, large)
    }
}

impl<F> Booster<F> for AdaBoostV<'_, F>
where
    F: Classifier + Clone,
{
    type Output = WeightedMajority<F>;

    fn name(&self) -> &str {
        "AdaBoostV"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_examples, n_feature) = self.sample.shape();
        let quit = if let Some(it) = self.force_quit_at {
            format!("At round {it}")
        } else {
            "-".to_string()
        };
        let info = Vec::from([
            ("# of examples", format!("{}", n_examples)),
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
        let n_examples = self.sample.shape().0;
        self.dist = vec![1.0 / n_examples as f64; n_examples];
        self.log_dist = vec![-(n_examples as f64).ln(); n_examples];

        self.rho = 1.0;
        self.gamma = 1.0;
        self.min_edge_coefficient = f64::INFINITY;

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

        // Each element in `predictions` is the product of
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

impl<H> CurrentHypothesis for AdaBoostV<'_, H>
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
    fn underflowed_mass_recovers_and_preprocessing_resets_logs() {
        let sample = Sample::dummy(2);
        let mut booster = AdaBoostV::<Hypothesis>::init(&sample).tolerance(0.25);
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
        let mut booster = AdaBoostV::<Hypothesis>::init(&sample).tolerance(0.25);
        booster.preprocess();
        let mut reference = vec![0.25; 4];
        let mut minimum_edge: f64 = 1.0;
        for margins in [[1.0, 1.0, 1.0, -1.0], [-1.0, 1.0, 1.0, 1.0]] {
            let edge: f64 = margins.iter().zip(&reference).map(|(m, d)| m * d).sum();
            minimum_edge = minimum_edge.min(edge);
            let alpha = edge.atanh() - (minimum_edge - 0.25).atanh();
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
        let mut first = AdaBoostV::<Hypothesis>::init(&sample).tolerance(0.25);
        let mut shifted = AdaBoostV::<Hypothesis>::init(&sample).tolerance(0.25);
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
        let mut booster = AdaBoostV::init(&sample).tolerance(0.25);
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

    #[test]
    fn public_boost_preserves_sub_epsilon_margin_tolerance() {
        let sample = Sample::dummy(2);
        let tolerance = 1e-18;
        let mut booster = AdaBoostV::init(&sample).tolerance(tolerance);
        booster.preprocess();
        booster.log_dist = vec![-1000.0, 0.0];
        booster.dist = helpers::softmax(&booster.log_dist);
        assert!(booster.boost(&RecoversMass, 1).is_continue());
        assert!(booster.weights[0].is_finite());
        // gamma differs from one by order exp(-1000). Ignoring that negligible
        // term, the recovered mass is tolerance/2; the relative check ensures
        // that no machine-epsilon floor replaced the requested tolerance.
        assert!((booster.dist[0] / tolerance - 0.5).abs() < 1e-12);
        booster.preprocess();
        assert_eq!(booster.min_edge_coefficient, f64::INFINITY);
    }

    #[test]
    fn rejects_nonpositive_and_nonfinite_tolerance() {
        let sample = Sample::dummy(2);
        for tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                std::panic::catch_unwind(|| {
                    AdaBoostV::<Hypothesis>::init(&sample).tolerance(tolerance)
                })
                .is_err()
            );
        }
    }

    #[test]
    #[should_panic(expected = "estimated margin rho")]
    fn rejects_margin_estimate_outside_its_true_domain() {
        let sample = Sample::dummy(2);
        let mut booster = AdaBoostV::<Hypothesis>::init(&sample).tolerance(0.5);
        booster.preprocess();
        booster.update_params(vec![1.0, -1.0], (-0.75f64).atanh());
    }

    #[test]
    fn tiny_margin_correction_and_negative_endpoint_remain_finite() {
        let sample = Sample::dummy(2);
        let mut booster = AdaBoostV::<Hypothesis>::init(&sample).tolerance(1e-18);
        booster.preprocess();
        let weight = booster.update_params(vec![1.0, -1.0], 1e-18);
        assert!((weight / 1e-18 - 1.0).abs() < 1e-14);
        booster.preprocess();
        // tanh(-20) rounds to -1, although 1+gamma exceeds the tolerance.
        let weight = booster.update_params(vec![1.0, -1.0], -20.0);
        assert!(weight.is_finite() && weight > 0.0);
    }
}
