//! Provides [`AdaBoost`] by Freund & Schapire, 1995.
use rayon::prelude::*;

use hypotheses::WeightedMajority;
use miniboosts_core::CurrentHypothesis;
use miniboosts_core::{
    Booster, Classifier, Sample, WeakLearner, constants::DEFAULT_TOLERANCE, tools::helpers,
};

use std::ops::ControlFlow;

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

    // Hypohteses obtained by the weak-learner.
    hypotheses: Vec<F>,

    // Max iteration until AdaBoost guarantees the optimality.
    max_iter: usize,

    // Optional. If this value is `Some(it)`,
    // the algorithm terminates after `it` iterations.
    force_quit_at: Option<usize>,

    // Terminated iteration.
    // AdaBoost terminates in eary step
    // if the training set is linearly separable.
    terminated: usize,
}

impl<'a, F> AdaBoost<'a, F> {
    /// Constructs a new instance of `AdaBoost`.
    ///
    /// Time complexity: `O(1)`.
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

    /// Returns the configured iteration budget, `ceil(ln(m) / tolerance²)`.
    /// This is not a measured training-error stopping condition. Convergence
    /// guarantees require the weak-learning assumptions in the source method.
    ///
    /// Time complexity: `O(1)`.
    pub fn max_loop(&self) -> usize {
        let n_sample = self.sample.shape().0 as f64;

        (n_sample.ln() / self.tolerance.powi(2)) as usize
    }

    /// Force quits after at most `it` iterations.
    /// Note that if `it` is smaller than the iteration bound
    /// for AdaBoost, the returned hypothesis has no guarantee.
    ///
    /// Time complexity: `O(1)`.
    pub fn force_quit_at(mut self, it: usize) -> Self {
        self.force_quit_at = Some(it);
        self
    }

    /// Set the parameter used to compute the iteration budget.
    /// Training error is not checked against this value each round.
    ///
    /// Time complexity: `O(1)`.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Returns a weight on the new hypothesis.
    /// `update_params` also updates `self.dist`.
    ///
    /// `AdaBoost` uses exponential update,
    /// which is numerically unstable so that I adopt a logarithmic computation.
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
