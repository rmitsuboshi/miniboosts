//! Provides [`AdaBoost`] by Freund & Schapire, 1995.
use rayon::prelude::*;

use miniboosts_core::{
    Booster,
    WeakLearner,
    Classifier,
    Sample,
    tools::helpers,
    constants::DEFAULT_TOLERANCE,
};
use logging::CurrentHypothesis;
use hypotheses::WeightedMajority;

use std::ops::ControlFlow;

pub struct AdaBoost<'a, F> {
    // Training sample
    sample: &'a Sample,

    // Distribution on sample.
    dist: Vec<f64>,

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
    /// Constructs a new instance of `AdaBoostV`.
    /// 
    /// Time complexity: `O(1)`.
    #[inline]
    pub fn init(sample: &'a Sample) -> Self {
        Self {
            sample,

            tolerance:     DEFAULT_TOLERANCE,

            dist:          Vec::new(),
            weights:       Vec::new(),
            hypotheses:    Vec::new(),

            force_quit_at: None,
            max_iter:      usize::MAX,
            terminated:    usize::MAX,
        }
    }

    /// Returns the maximum iteration
    /// of the `AdaBoost` to find a combined hypothesis
    /// that has error at most `tolerance`.
    /// After the `self.max_loop()` iterations,
    /// `AdaBoost` guarantees zero training error in terms of zero-one loss
    /// if the training examples are linearly separable.
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

    /// Set the tolerance parameter.
    /// `AdaBoostV` terminates immediately
    /// after reaching the specified `tolerance`.
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
    /// Time complexity: `O( m ln(m) )`,
    /// where `m` is the number of training examples.
    /// The additional `ln(m)` term comes from the numerical stabilization.
    #[inline]
    fn update_params(
        &mut self,
        margins: Vec<f64>,
        edge: f64
    ) -> f64
    {
        let n_sample = self.sample.shape().0;

        // Compute the weight on new hypothesis.
        // This is the returned value of this function.
        let weight = ((1.0 + edge) / (1.0 - edge)).ln() / 2.0;

        // To prevent overflow, take the logarithm.
        self.dist.par_iter_mut()
            .zip(margins)
            .for_each(|(d, p)| *d = d.ln() - weight * p);

        // Sort indices by ascending order
        let mut indices = (0..n_sample).into_par_iter()
            .collect::<Vec<usize>>();
        indices.sort_unstable_by(|&i, &j| {
            self.dist[i].partial_cmp(&self.dist[j]).unwrap()
        });

        let mut normalizer = self.dist[indices[0]];
        for i in indices.into_iter().skip(1) {
            let mut a = normalizer;
            let mut b = self.dist[i];
            if a < b {
                std::mem::swap(&mut a, &mut b);
            }

            normalizer = a + (1.0 + (b - a).exp()).ln();
        }

        // Update distribution over training examples.
        self.dist.par_iter_mut()
            .for_each(|d| *d = (*d - normalizer).exp());

        weight
    }
}

impl<F> Booster<F> for AdaBoost<'_, F>
    where F: Classifier + Clone,
{
    type Output = WeightedMajority<F>;

    fn name(&self) -> &str { "AdaBoost" }

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

        self.weights = Vec::new();
        self.hypotheses = Vec::new();

        self.max_iter = self.max_loop();

        if let Some(it) = self.force_quit_at {
            self.max_iter = it;
        }
    }

    fn boost<W>(
        &mut self,
        weak_learner: &W,
        iteration: usize,
    ) -> ControlFlow<usize>
        where W: WeakLearner<Hypothesis = F>,
    {
        if self.max_iter < iteration {
            return ControlFlow::Break(self.max_iter);
        }

        // Get a new hypothesis
        let h = weak_learner.produce(self.sample, &self.dist);

        // Each element in `margins` is the product of
        // the predicted vector and the correct vector
        let margins = helpers::margins(self.sample, &h)
            .collect::<Vec<_>>();

        let edge = helpers::inner_product(&margins, &self.dist);

        // If `h` predicted all the examples in `sample` correctly,
        // use it as the combined classifier.
        if edge.abs() >= 1.0 {
            self.terminated = iteration;
            self.weights = vec![edge.signum()];
            self.hypotheses = vec![h];
            return ControlFlow::Break(iteration);
        }

        // Compute the weight on the new hypothesis
        let weight = self.update_params(margins, edge);
        self.weights.push(weight);
        self.hypotheses.push(h);

        ControlFlow::Continue(())
    }

    fn postprocess(&mut self) -> Self::Output {
        WeightedMajority::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}

impl<H> CurrentHypothesis for AdaBoost<'_, H>
    where H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        WeightedMajority::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}

