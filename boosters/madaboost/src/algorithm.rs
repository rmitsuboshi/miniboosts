//! Provides [`MadaBoost`] by Domingo and Watanabe, 2000.
use rayon::prelude::*;

use hypotheses::WeightedMajority;
use miniboosts_core::CurrentHypothesis;
use miniboosts_core::{
    Booster, Classifier, Sample, WeakLearner, checkers, constants::DEFAULT_TOLERANCE,
    tools::helpers,
};

use std::ops::ControlFlow;

pub struct MadaBoost<'a, F> {
    // Training sample
    sample: &'a Sample,

    // Weights for each instances in `sample.`
    // At the end of round `t,`
    // the `i`th element of `betas` holds
    // `ln Bt[i] = sum_{k=1}^{T} ( y[i] hk(x[i]) ln beta[k] ).`
    betas: Vec<f64>,

    // Tolerance parameter
    tolerance: f64,

    // Weights on hypotheses in `hypotheses`
    alphas: Vec<f64>,

    // Hypohteses obtained by the weak-learner.
    hypotheses: Vec<F>,

    // Iteration budget.
    max_iter: usize,

    // Optional. If this value is `Some(it)`,
    // the algorithm terminates after `it` iterations.
    force_quit_at: Option<usize>,

    // Terminated iteration.
    // MadaBoost terminates in eary step
    // if the training set is linearly separable.
    terminated: usize,
}

impl<'a, F> MadaBoost<'a, F> {
    /// Constructs a new instance of `MadaBoost`.
    ///
    /// Time complexity: `O(1)`.
    #[inline]
    pub fn init(sample: &'a Sample) -> Self {
        Self {
            sample,

            tolerance: DEFAULT_TOLERANCE,

            alphas: Vec::new(),
            betas: Vec::new(),
            hypotheses: Vec::new(),

            force_quit_at: None,
            max_iter: usize::MAX,
            terminated: usize::MAX,
        }
    }

    /// Returns the iteration budget `(m - 1) / tolerance^2`, rounded up.
    /// This implementation uses MadaBoost's capped-product weights (Section 3),
    /// not the modified variant analyzed in the paper's convergence proof.
    /// The budget does not guarantee zero training error.
    ///
    /// Time complexity: `O(1)`.
    pub fn max_loop(&self) -> usize {
        let n_examples = self.sample.shape().0 as f64;

        ((n_examples - 1f64) / self.tolerance.powi(2)).ceil() as usize
    }

    /// Force quits after at most `it` iterations.
    /// Overrides the default iteration budget.
    ///
    /// Time complexity: `O(1)`.
    pub fn force_quit_at(mut self, it: usize) -> Self {
        self.force_quit_at = Some(it);
        self
    }

    /// Set the tolerance used to compute the iteration budget.
    /// This does not test the training error after each round.
    ///
    /// Time complexity: `O(1)`.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Returns a alpha on the new hypothesis.
    /// `update_params` also updates `self.betas`.
    ///
    /// `MadaBoost` uses exponential update,
    /// which is numerically unstable so that I adopt a logarithmic computation.
    ///
    /// Time complexity: `O(m)` for `m` training examples.
    #[inline]
    fn update_params(&mut self, margins: Vec<f64>, alpha: f64, h: F) {
        assert!(alpha.is_finite());
        // log(beta) = -alpha; retain the uncapped product history.
        assert_eq!(self.betas.len(), margins.len());
        self.betas.par_iter_mut().zip(margins).for_each(|(b, yh)| {
            *b -= yh * alpha;
        });

        self.alphas.push(alpha);
        self.hypotheses.push(h);
    }

    // Domingo & Watanabe (2000), Section 3, capped weighting scheme:
    // https://www.learningtheory.org/colt2000/papers/DomingoWatanabe.pdf
    // Keep log(min(1, B_t)) in log space until the final normalization.
    fn distribution(&self) -> Vec<f64> {
        // Do not normalize betas themselves: their absolute scale determines
        // which product weights are capped at one in the original method.
        let capped_logs = self
            .betas
            .iter()
            .map(|&b| {
                assert!(b.is_finite(), "MadaBoost log weights must be finite");
                b.min(0.0)
            })
            .collect::<Vec<_>>();
        let dist = helpers::softmax(&capped_logs);
        checkers::capped_simplex_condition(&dist[..], 1f64);
        dist
    }
}

impl<F> Booster<F> for MadaBoost<'_, F>
where
    F: Classifier + Clone,
{
    type Output = WeightedMajority<F>;

    fn name(&self) -> &str {
        "MadaBoost"
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
        self.betas = vec![0f64; n_examples];

        self.alphas = Vec::new();
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

        let dist = self.distribution();
        // Get a new hypothesis
        let h = weak_learner.produce(self.sample, &dist[..]);

        // Each element in `margins` is the product of
        // the predicted vector and the correct vector
        let margins = helpers::margins(self.sample, &h).collect::<Vec<_>>();

        let capped_logs: Vec<_> = self.betas.iter().map(|b| b.min(0.0)).collect();
        let coefficient = helpers::log_weighted_edge_coefficient(&capped_logs, &margins);

        // Test perfect margins on log support, including underflowed mass.
        if coefficient.is_infinite() {
            self.terminated = iteration;
            self.alphas = vec![coefficient.signum()];
            self.hypotheses = vec![h];
            return ControlFlow::Break(iteration);
        }

        self.update_params(margins, coefficient, h);

        ControlFlow::Continue(())
    }

    fn postprocess(&mut self) -> Self::Output {
        WeightedMajority::from_slices(&self.alphas[..], &self.hypotheses[..])
    }
}

impl<H> CurrentHypothesis for MadaBoost<'_, H>
where
    H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        WeightedMajority::from_slices(&self.alphas[..], &self.hypotheses[..])
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    #[test]
    fn capped_weights_are_normalized_once() {
        let sample = Sample::dummy(2);
        let mut booster = MadaBoost::<()>::init(&sample);
        booster.betas = vec![-4f64.ln(), 0.0];
        let dist = booster.distribution();
        assert!((dist[0] - 0.2).abs() < 1e-14);
        assert!((dist[1] - 0.8).abs() < 1e-14);
        booster.betas = vec![-1000.0, -1000.0];
        assert!(
            booster
                .distribution()
                .iter()
                .all(|d| (*d - 0.5).abs() < 1e-12)
        );
    }

    #[test]
    fn recurrence_matches_capped_product_of_betas() {
        let sample = Sample::dummy(4);
        let mut booster = MadaBoost::init(&sample);
        booster.betas = vec![0.0; 4];
        let first = vec![1.0, 1.0, 1.0, -1.0];
        booster.update_params(first.clone(), 0.5f64.atanh(), ());
        let second = vec![-1.0, 1.0, 1.0, 1.0];
        let dist = booster.distribution();
        let edge: f64 = second.iter().zip(&dist).map(|(m, d)| m * d).sum();
        booster.update_params(second.clone(), edge.atanh(), ());
        let beta1 = (1f64 / 3.0).sqrt();
        let beta2 = ((1.0 - edge) / (1.0 + edge)).sqrt();
        let raw: Vec<_> = first
            .iter()
            .zip(second)
            .map(|(a, b)| (beta1.powf(*a) * beta2.powf(b)).min(1.0))
            .collect();
        let sum: f64 = raw.iter().sum();
        for (actual, raw) in booster.distribution().iter().zip(raw) {
            assert!((actual - raw / sum).abs() < 1e-14);
        }
        assert!((booster.alphas[0] + beta1.ln()).abs() < 1e-14);
        assert!((booster.alphas[1] + beta2.ln()).abs() < 1e-14);
    }
    #[test]
    fn uncapped_log_shift_is_invariant_without_mutating_products() {
        let sample = Sample::dummy(2);
        let mut booster = MadaBoost::<()>::init(&sample);
        booster.betas = vec![-1000.0, -1001.0];
        let first = booster.distribution();
        assert_eq!(booster.betas, vec![-1000.0, -1001.0]);
        booster.betas = vec![-2.0, -3.0];
        for (a, b) in first.iter().zip(booster.distribution()) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn capped_product_recovers_after_probability_underflow() {
        let sample = Sample::dummy(2);
        let mut booster = MadaBoost::init(&sample);
        booster.betas = vec![0.0; 2];
        for _ in 0..1600 {
            booster.update_params(vec![1.0, -1.0], 0.5f64.atanh(), ());
        }
        assert_eq!(booster.distribution()[0], 0.0);
        for _ in 0..1600 {
            booster.update_params(vec![-1.0, 1.0], 0.5f64.atanh(), ());
        }
        assert!((booster.distribution()[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn small_edge_retains_a_nonzero_coefficient() {
        let sample = Sample::dummy(2);
        let mut booster = MadaBoost::init(&sample);
        booster.betas = vec![0.0; 2];
        let margins = vec![1e-18; 2];
        let coefficient = helpers::log_weighted_edge_coefficient(&booster.betas, &margins);
        booster.update_params(margins, coefficient, ());
        assert!((booster.alphas[0] / 1e-18 - 1.0).abs() < 1e-14);
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
        let mut booster = MadaBoost::init(&sample).tolerance(0.25);
        booster.preprocess();
        booster.betas = vec![-1000.0, 0.0];
        // The ordinary weighted edge rounds to +1 even though row zero is
        // misclassified and has positive mathematical mass exp(-1000).
        assert!(booster.boost(&RecoversMass, 1).is_continue());
        assert!(booster.alphas[0].is_finite());
        assert!((booster.distribution()[0] - 0.5).abs() < 1e-12);
    }
}
