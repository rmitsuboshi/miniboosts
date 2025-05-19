//! Provides [`MadaBoost`] by Domingo and Watanabe, 2000.
use rayon::prelude::*;

use miniboosts_core::{
    Booster,
    WeakLearner,
    Classifier,
    Sample,
    checkers,
    tools::helpers,
    constants::DEFAULT_TOLERANCE,
};
use logging::CurrentHypothesis;
use hypotheses::WeightedMajority;

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

    // Max iteration until MadaBoost guarantees the optimality.
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
    /// Constructs a new instance of `MadaBoostV`.
    /// 
    /// Time complexity: `O(1)`.
    #[inline]
    pub fn init(sample: &'a Sample) -> Self {
        Self {
            sample,

            tolerance: DEFAULT_TOLERANCE,

            alphas:        Vec::new(),
            betas:         Vec::new(),
            hypotheses:    Vec::new(),

            force_quit_at: None,
            max_iter:      usize::MAX,
            terminated:    usize::MAX,
        }
    }

    /// Returns the maximum iteration
    /// of the `MadaBoost` to find a combined hypothesis
    /// that has error at most `tolerance`.
    /// After the `self.max_loop()` iterations,
    /// `MadaBoost` guarantees zero training error in terms of zero-one loss
    /// if the training examples are linearly separable.
    /// 
    /// Time complexity: `O(1)`.
    pub fn max_loop(&self) -> usize {
        let n_examples = self.sample.shape().0 as f64;

        ((n_examples - 1f64) / self.tolerance.powi(2)).ceil() as usize
    }

    /// Force quits after at most `it` iterations.
    /// Note that if `it` is smaller than the iteration bound
    /// for MadaBoost, the returned hypothesis has no guarantee.
    /// 
    /// Time complexity: `O(1)`.
    pub fn force_quit_at(mut self, it: usize) -> Self {
        self.force_quit_at = Some(it);
        self
    }

    /// Set the tolerance parameter.
    /// `MadaBoostV` terminates immediately
    /// after reaching the specified `tolerance`.
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
    /// Time complexity: `O( m ln(m) )`,
    /// where `m` is the number of training examples.
    /// The additional `ln(m)` term comes from the numerical stabilization.
    #[inline]
    fn update_params(&mut self, margins: Vec<f64>, edge: f64, h: F) {
        // Defines the `epsilon` by transforming `edge.`
        // We assume that `eps` is in `[0.0, 1.0).`
        let eps = 0.5f64 * (1f64 - edge);
        assert!((0f64..1f64).contains(&eps), "EPS: {}", eps);
        let beta = (eps / (1f64 - eps)).sqrt();


        // To prevent overflow, take the logarithm.
        self.betas.par_iter_mut()
            .zip(margins)
            .for_each(|(b, yh)| { *b += yh * beta.ln(); });


        // Compute the weight on new hypothesis.
        // This is the returned value of this function.
        let alpha = (1f64 / beta).ln();
        self.alphas.push(alpha);
        self.hypotheses.push(h);
    }

    fn distribution(&self) -> Vec<f64> {
        let n_examples = self.sample.shape().0;

        let weights = {
            let mut weights = self.betas.iter()
                .copied()
                .map(|b| b.min(0f64).exp())
                .collect::<Vec<_>>();
            weights.shrink_to_fit();
            weights
        };

        // Sort indices by ascending order
        let mut indices = (0..n_examples).collect::<Vec<usize>>();
        indices.sort_by(|&i, &j| weights[i].partial_cmp(&weights[j]).unwrap());

        let mut normalizer = weights[indices[0]];
        for i in indices.into_iter().skip(1) {
            let mut a = normalizer;
            let mut b = weights[i];
            if a < b {
                std::mem::swap(&mut a, &mut b);
            }

            normalizer = a + (1.0 + (b - a).exp()).ln();
        }

        let dist = weights.into_iter()
            .map(|b| (b - normalizer).exp())
            .collect::<Vec<_>>();
        checkers::capped_simplex_condition(&dist[..], 1f64);
        dist
    }
}

impl<F> Booster<F> for MadaBoost<'_, F>
    where F: Classifier + Clone,
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

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize)
        -> ControlFlow<usize>
        where W: WeakLearner<Hypothesis = F>,
    {
        if self.max_iter < iteration {
            return ControlFlow::Break(self.max_iter);
        }

        let dist = self.distribution();
        // Get a new hypothesis
        let h = weak_learner.produce(self.sample, &dist[..]);

        // Each element in `margins` is the product of
        // the predicted vector and the correct vector
        let margins = helpers::margins(self.sample, &h)
            .collect::<Vec<_>>();

        let edge = helpers::inner_product(&margins, &dist[..]);

        // If `h` predicted all the examples in `sample` correctly,
        // use it as the combined classifier.
        if edge.abs() >= 1.0 {
            self.terminated = iteration;
            self.alphas = vec![edge.signum()];
            self.hypotheses = vec![h];
            return ControlFlow::Break(iteration);
        }

        self.update_params(margins, edge, h);

        ControlFlow::Continue(())
    }

    fn postprocess(&mut self) -> Self::Output {
        WeightedMajority::from_slices(&self.alphas[..], &self.hypotheses[..])
    }
}

impl<H> CurrentHypothesis for MadaBoost<'_, H>
    where H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        WeightedMajority::from_slices(&self.alphas[..], &self.hypotheses[..])
    }
}

