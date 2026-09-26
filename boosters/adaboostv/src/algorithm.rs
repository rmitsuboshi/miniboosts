//! Provides `AdaBoost*` by Rätsch & Warmuth, 2005.
//! Since one cannot use `*` as a struct name,
//! We call `AdaBoost*` as `AdaBoostV`.
//! (I found this name in the paper of `SparsiBoost`)
use rayon::prelude::*;

use hypotheses::WeightedMajority;
use miniboosts_core::CurrentHypothesis;
use miniboosts_core::{Booster, Classifier, Sample, WeakLearner, tools::helpers};

use std::ops::ControlFlow;

pub struct AdaBoostV<'a, F> {
    /// Training sample
    sample: &'a Sample,

    /// Tolerance parameter
    tolerance: f64,

    rho: f64,

    gamma: f64,

    // atanh is monotone and preserves the distance from rounded +/-1 edges.
    min_edge_coefficient: f64,

    /// Distribution on sample.
    dist: Vec<f64>,

    // Keep finite log mass even when exp(log_mass) underflows to zero.
    log_dist: Vec<f64>,

    /// Weights on hypotheses in `hypotheses`
    weights: Vec<f64>,

    /// Hypohteses obtained by the weak-learner.
    hypotheses: Vec<F>,

    max_iter: usize,

    /// Optional. If this value is `Some(iteration)`,
    /// the algorithm terminates after `iteration` iterations.
    force_quit_at: Option<usize>,

    terminated: usize,
}

impl<'a, F> AdaBoostV<'a, F> {
    /// Constructs a new instance of `AdaBoostV`.
    ///
    /// Time complexity: `O(1)`.
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

    /// Set the margin-accuracy parameter used in updates and the iteration budget.
    /// Must be finite and positive. This is not a threshold on observed
    /// classification error.
    ///
    /// Time complexity: `O(1)`.
    #[inline]
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        assert!(
            tolerance.is_finite() && tolerance > 0.0,
            "AdaBoostV tolerance must be finite and positive"
        );
        self.tolerance = tolerance;

        self
    }

    /// Returns the configured iteration budget, `floor(2 ln(m) / tolerance²)`.
    /// Margin guarantees require the weak-learning assumptions of AdaBoostV;
    /// separability alone does not certify this particular weak learner.
    ///
    /// Time complexity: `O(1)`.
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

    /// Force quits after `iteration` iterations.
    /// Note that if `iteration` is smaller than the iteration bound
    /// for AdaBoostV,
    /// the returned hypothesis has no guarantee about the margin.
    ///
    /// Time complexity: `O(1)`.
    pub fn force_quit_at(mut self, iteration: usize) -> Self {
        self.force_quit_at = Some(iteration);
        self
    }

    /// Returns a weight on the new hypothesis.
    /// `update_params` also updates `self.dist`.
    ///
    /// `AdaBoostV` uses exponential update,
    /// which is numerically unstable so that I adopt a logarithmic computation.
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
