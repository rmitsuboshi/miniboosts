//! Provides [`TAdaBoost`] by Nock, Amid, and Warmuth, NeurIPS 2023.
use rayon::prelude::*;


use crate::{
    Booster,
    WeakLearner,
    Classifier,
    WeightedMajority,
    Sample,

    common::utils,
    research::Research,
};

use std::ops::ControlFlow;

const DEFAULT_MAX_ITER: usize = 1_000;


/// The TAdaBoost algorithm 
/// proposed by Richard Nock, Ehsan Amid, and Manfred K. Warmuth.
/// 
/// This struct is based on the paper: 
///
/// [
/// Boosting with Tempered Exponential Measures
/// ](https://papers.nips.cc/paper_files/paper/2023/file/82d3258eb58ceac31744a88005b7ddef-Paper-Conference.pdf)  
/// by Richard Nock, Ehsan Amid, and Manfred K. Warmuth.
/// 
/// TAdaBoost is a boosting algorithm for binary classification 
/// that minimizes the tempered relative entropy.
///
/// ```no_run
/// use miniboosts::prelude::*;
/// 
/// // Read the training sample from the CSV file.
/// // We use the column named `class` as the label.
/// let sample = SampleReader::default()
///     .file(path_to_file)
///     .has_header(true)
///     .target_feature("class")
///     .read()
///     .unwrap();
/// 
/// // Initialize `TAdaBoost` and set the tempered parameter as `0.01`.
/// let mut booster = TAdaBoost::init(&sample)
///     .tempered(0.1)
///     .max_loop(1_000); // number of iterations.
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .criterion(Criterion::Entropy)
///     .build();
/// 
/// // Run `TAdaBoost` and obtain the resulting hypothesis `f`.
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
///     .map(|(&y, fx)| if y as i64 == fx { 0.0 } else { 1.0 })
///     .sum::<f64>()
///     / n_sample;
/// 
///
/// println!("Training Loss is: {training_loss}");
/// ```
pub struct TAdaBoost<'a, F> {
    // Training sample
    sample: &'a Sample,

    // Distribution on sample.
    dist: Vec<f64>,

    // Weights on hypotheses in `hypotheses`
    weights: Vec<f64>,

    // Hypohteses obtained by the weak-learner.
    hypotheses: Vec<F>,

    // parameter `t`
    temp: f64,

    // parameter `t* = 1 / (2 - t)`
    temp_star: f64,

    // Max iteration until TAdaBoost guarantees the optimality.
    max_iter: usize,

    // Terminated iteration.
    // TAdaBoost terminates in eary step 
    // if the training set is linearly separable.
    terminated: usize,
}


impl<'a, F> TAdaBoost<'a, F> {
    /// Constructs a new instance of `TAdaBoostV`.
    /// 
    /// Time complexity: `O(1)`.
    #[inline]
    pub fn init(sample: &'a Sample) -> Self {
        let n_sample = sample.shape().0;

        Self {
            sample,

            dist: Vec::new(),
            weights: Vec::new(),
            hypotheses: Vec::new(),

            temp: 0.5f64,
            temp_star: 2f64 / 3f64, // == 1 / (2 - 0.5)

            max_iter: DEFAULT_MAX_ITER,
            terminated: DEFAULT_MAX_ITER,
        }
    }

    /// Set the number of iterations of this algorithm.
    /// `max_iter` corresponds to `J` in the original paper.
    /// By default, the number of iteration is `1_000.`
    pub fn max_loop(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }


    /// Sets the "tempered" parameter.
    /// This method panics if `temp` is not in [0, 1].`
    pub fn tempered(mut self, temp: f64) -> Self {
        assert!(
            (0f64..=1f64).contains(&temp),
            "The tempered parameter must be in [0, 1]"
        );

        self.temp = temp;
        self.temp_star = 1f64 / (2f64 - self.temp);
        self
    }


    /// Updates the parameters in `TAdaBoost.`
    #[inline]
    fn update_params(
        &mut self,
        margins: Vec<f64>,
        edge: f64
    ) -> f64
    {
        todo!()
    }
}


impl<F> Booster<F> for TAdaBoost<'_, F>
    where F: Classifier + Clone,
{
    // TODO
    // Set the proper struct for this algorithm (clipping).
    type Output = WeightedMajority<F>;


    fn name(&self) -> &str {
        "t-AdaBoost"
    }


    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_sample, n_feature) = self.sample.shape();
        let info = Vec::from([
            ("# of examples", format!("{}", n_sample)),
            ("# of features", format!("{}", n_feature)),
            ("Max iteration", format!("{}", self.max_iter)),
            ("t (parameter)", format!("{}", self.temp)),
            ("t* (parameter)", format!("{}", self.temp_star)),
        ]);
        Some(info)
    }


    fn preprocess<W>(
        &mut self,
        _weak_learner: &W,
    )
        where W: WeakLearner<Hypothesis = F>
    {
        todo!()
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
        todo!()
    }


    fn postprocess<W>(
        &mut self,
        _weak_learner: &W,
    ) -> Self::Output
        where W: WeakLearner<Hypothesis = F>
    {
        todo!()
    }
}


impl<H> Research for TAdaBoost<'_, H>
    where H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        todo!()
    }
}
