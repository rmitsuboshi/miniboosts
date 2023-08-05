//! Provides `AdaBoost*` by Rätsch & Warmuth, 2005.
//! Since one cannot use `*` as a struct name,
//! We call `AdaBoost*` as `AdaBoostV`.
//! (I found this name in the paper of `SparsiBoost`.
use rayon::prelude::*;


use crate::{
    Sample,
    Booster,
    WeakLearner,

    Classifier,
    CombinedHypothesis,

    common::utils,
    research::Research,
};

use std::ops::ControlFlow;



/// Defines `AdaBoostV`.
/// This struct is based on the paper: 
/// `AdaBoostV`, also known as `AdaBoost_{\nu}^\star`, is a boosting algorithm
/// in the following paper:
/// 
/// [Gunnar Rätsch and Manfred K. Warmuth - Efficient Margin Maximizing with Boosting](https://www.jmlr.org/papers/v6/ratsch05a.html)
/// 
/// `AdaBoostV` aims to maximize the hard-margin 
/// for linearly separable training instances.
/// 
/// # Example
/// The following code shows a small example 
/// for running [`AdaBoostV`](AdaBoostV).  
/// See also:
/// - [`DecisionTree`]
/// - [`DecisionTreeClassifier`]
/// - [`CombinedHypothesis<F>`]
/// - [`Sample`]
/// 
/// [`DecisionTree`]: crate::weak_learner::DecisionTree
/// [`DecisionTreeClassifier`]: crate::weak_learner::DecisionTreeClassifier
/// [`CombinedHypothesis<F>`]: crate::hypothesis::CombinedHypothesis
/// 
/// 
/// ```no_run
/// use miniboosts::prelude::*;
/// 
/// // Read the training sample from the CSV file.
/// // We use the column named `class` as the label.
/// let has_header = true;
/// let sample = Sample::from_csv(path_to_csv_file, has_header)
///     .unwrap()
///     .set_target("class");
/// 
/// // Initialize `AdaBoostV` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis whose training error is
/// // less than `0.01` if the traing examples are linearly separable.
/// let mut booster = AdaBoostV::init(&sample)
///     .tolerance(0.01);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .criterion(Criterion::Entropy)
///     .build();
/// 
/// // Run `AdaBoostV` and obtain the resulting hypothesis `f`.
/// let f = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions = f.predict_all(&sample);
/// 
/// // Get the number of training examples.
/// let n_sample = sample.shape().0 as f64;
/// 
/// // Calculate the training loss.
/// let target = sample.target();
/// let training_loss = target.into_iter()
///     .zip(predictions)
///     .map(|(&y, fx) if y as i64 == fx { 0.0 } else { 1.0 })
///     .sum::<f64>()
///     / n_sample;
/// 
///
/// println!("Training Loss is: {training_loss}");
/// ```
pub struct AdaBoostV<'a, F> {
    /// Training sample
    sample: &'a Sample,

    /// Tolerance parameter
    tolerance: f64,

    rho: f64,

    gamma: f64,

    /// Distribution on sample.
    dist: Vec<f64>,

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
        let n_sample = sample.shape().0;
        let default_tolerance = 1.0 / n_sample as f64;
        Self {
            sample,

            tolerance: default_tolerance,
            rho: 1.0,
            gamma: 1.0,

            dist: Vec::new(),
            weights: Vec::new(),
            hypotheses: Vec::new(),

            max_iter: usize::MAX,
            terminated: usize::MAX,
            force_quit_at: None,
        }
    }



    /// Set the tolerance parameter.
    /// `AdaBoostV` terminates immediately
    /// after reaching the specified `tolerance`.
    /// 
    /// Time complexity: `O(1)`.
    #[inline]
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;

        self
    }


    /// `max_loop` returns the maximum iteration
    /// of the `AdaBoostV` to find a combined hypothesis
    /// that has error at most `tolerance`.
    /// After the `self.max_loop()` iterations,
    /// `AdaBoostV` guarantees zero training error in terms of zero-one loss
    /// if the training examples are linearly separable.
    /// 
    /// Time complexity: `O(1)`.
    #[inline]
    pub fn max_loop(&self) -> usize {
        let n_sample = self.sample.shape().0 as f64;

        (2.0 * n_sample.ln() / self.tolerance.powi(2)) as usize
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
    /// Time complexity: `O( m ln(m) )`,
    /// where `m` is the number of training examples.
    /// The additional `ln(m)` term comes from the numerical stabilization.
    #[inline]
    fn update_params(&mut self, margins: Vec<f64>, edge: f64)
        -> f64
    {


        // Update edge & margin estimation parameters
        self.gamma = edge.min(self.gamma);
        self.rho = self.gamma - self.tolerance;


        let weight = {
            let e = ((1.0 + edge) / (1.0 - edge)).ln() / 2.0;
            let m = ((1.0 + self.rho) / (1.0 - self.rho)).ln() / 2.0;

            e - m
        };


        // To prevent overflow, take the logarithm.
        self.dist.par_iter_mut()
            .zip(margins)
            .for_each(|(d, yh)| *d = d.ln() - weight * yh);


        let m = self.dist.len();
        let mut indices = (0..m).collect::<Vec<usize>>();
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


        self.dist.par_iter_mut()
            .for_each(|d| *d = (*d - normalizer).exp());

        weight
    }
}


impl<F> Booster<F> for AdaBoostV<'_, F>
    where F: Classifier + Clone,
{
    fn preprocess<W>(
        &mut self,
        _weak_learner: &W,
    )
        where W: WeakLearner<Hypothesis = F>
    {
        self.sample.is_valid_binary_instance();
        // Initialize parameters
        let n_sample = self.sample.shape().0;
        self.dist = vec![1.0 / n_sample as f64; n_sample];

        self.rho = 1.0;
        self.gamma = 1.0;


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


        // Each element in `predictions` is the product of
        // the predicted vector and the correct vector
        let margins = utils::margins_of_hypothesis(self.sample, &h);


        let edge = utils::inner_product(&margins, &self.dist);


        // If `h` predicted all the examples in `self.sample` correctly,
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


    fn postprocess<W>(
        &mut self,
        _weak_learner: &W,
    ) -> CombinedHypothesis<F>
        where W: WeakLearner<Hypothesis = F>
    {
        CombinedHypothesis::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}


impl<H> Research<H> for AdaBoostV<'_, H>
    where H: Classifier + Clone,
{
    fn current_hypothesis(&self) -> CombinedHypothesis<H> {
        CombinedHypothesis::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}

