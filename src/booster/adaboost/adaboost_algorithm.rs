//! Provides [`AdaBoost`](AdaBoost) by Freund & Schapire, 1995.
use rayon::prelude::*;


use crate::{
    Booster,
    WeakLearner,
    Classifier,
    CombinedHypothesis,
    Sample,

    common::utils,
    research::Research,
};

use std::ops::ControlFlow;


/// Defines `AdaBoost`.
/// This struct is based on the book: 
/// [Boosting: Foundations and Algorithms](https://direct.mit.edu/books/oa-monograph/5342/BoostingFoundations-and-Algorithms)
/// by Robert E. Schapire and Yoav Freund.
/// 
/// # Example
/// The following code shows a small example 
/// for running [`AdaBoost`](AdaBoost).  
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
/// let mut sample = Sample::from_csv(path_to_csv_file, has_header)
///     .unwrap()
///     .set_target("class");
/// 
/// // Initialize `AdaBoost` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis whose training error is
/// // less than `0.01` if the traing examples are linearly separable.
/// // Note that the default tolerance parameter is set as `1 / n_sample`,
/// // where `n_sample = sample.shape().0` is 
/// // the number of training examples in `sample`.
/// let booster = AdaBoost::init(&sample)
///     .tolerance(0.01);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTree::init(&sample)
///     .max_depth(2)
///     .criterion(Criterion::Edge);
/// 
/// // Run `AdaBoost` and obtain the resulting hypothesis `f`.
/// let f: CombinedHypothesis<DecisionTreeClassifier> = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions: Vec<i64> = f.predict_all(&sample);
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
    /// Initialize the `AdaBoost`.
    /// This method sets some parameters `AdaBoost` holds.
    pub fn init(sample: &'a Sample) -> Self {
        let n_sample = sample.shape().0;

        let uni = 1.0 / n_sample as f64;
        AdaBoost {
            sample,

            dist: vec![uni; n_sample],
            tolerance: 1.0 / (n_sample as f64 + 1.0),

            weights: Vec::new(),
            hypotheses: Vec::new(),

            max_iter: usize::MAX,
            force_quit_at: None,
            terminated: usize::MAX,
        }
    }


    /// Returns the maximum iteration
    /// of the `AdaBoost` to find a combined hypothesis
    /// that has error at most `tolerance`.
    /// After the `self.max_loop()` iterations,
    /// `AdaBoost` guarantees zero training error in terms of zero-one loss
    /// if the training examples are linearly separable.
    pub fn max_loop(&self) -> usize {
        let n_sample = self.sample.shape().0 as f64;

        (n_sample.ln() / self.tolerance.powi(2)) as usize
    }


    /// Force quits after `it` iterations.
    /// Note that if `it` is smaller than the iteration bound
    /// for AdaBoost, the returned hypothesis has no guarantee.
    pub fn force_quit_at(mut self, it: usize) -> Self {
        self.force_quit_at = Some(it);
        self
    }


    /// Set the tolerance parameter.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }


    /// Returns a weight on the new hypothesis.
    /// `update_params` also updates `self.dist`
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



        // Update self.dist
        self.dist.par_iter_mut()
            .for_each(|d| *d = (*d - normalizer).exp());


        weight
    }
}


impl<F> Booster<F> for AdaBoost<'_, F>
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
        let margins = utils::margins_of_hypothesis(self.sample, &h);


        let edge = utils::inner_product(&margins, &self.dist);


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


    fn postprocess<W>(
        &mut self,
        _weak_learner: &W,
    ) -> CombinedHypothesis<F>
        where W: WeakLearner<Hypothesis = F>
    {
        CombinedHypothesis::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}


impl<H> Research<H> for AdaBoost<'_, H>
    where H: Classifier + Clone,
{
    fn current_hypothesis(&self) -> CombinedHypothesis<H> {
        CombinedHypothesis::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}
