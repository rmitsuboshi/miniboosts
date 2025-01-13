//! Provides Gradient Boosting Machine ([`GBM`]) by Friedman, 2001.
use rayon::prelude::*;

use crate::{
    common::loss_functions::*,
    Sample,
    Booster,
    WeakLearner,
    Regressor,
    WeightedMajority
};

use std::ops::ControlFlow;


/// The Gradient Boosting Machine proposed in the following paper:
/// 
/// [Jerome H. Friedman, 2001 - Greedy Function Approximation: A Gradient Boosting Machine](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boostingmachine/10.1214/aos/1013203451.full)
/// 
/// Gradient Boosting Machine, GBM for shorthand, is a boosting algorithm
/// that minimizes the training loss.
/// GBM regards the boosting protocol as the gradient descent 
/// over some functional space
/// (One can see GBM as coordinate descent algorithm,
/// where each coordinate corresponds to some function in that space).
/// 
/// **Note.** Currently, I only implements GBM for regression.
/// 
/// 
/// # Example
/// The following code shows a small example 
/// for running [`GBM`].  
/// See also:
/// - [`Regressor`]
/// - [`WeightedMajority<F>`]
/// 
/// [`RegressionTree`]: crate::weak_learner::RegressionTree
/// [`RegressionTreeRegressor`]: crate::weak_learner::RegressionTreeRegressor
/// [`WeightedMajority<F>`]: crate::hypothesis::WeightedMajority
/// 
/// 
/// ```no_run
/// use miniboosts::prelude::*;
/// 
/// // Read the training sample from the CSV file.
/// // We use the column named `class` as the label.
/// let sample = SampleReader::new()
///     .file(path_to_file)
///     .has_header(true)
///     .target_feature("class")
///     .read()
///     .unwrap();
/// 
/// // Initialize `GBM` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis whose training error is
/// // less than `0.01` if the traing examples are linearly separable.
/// // Note that the default tolerance parameter is set as `1 / n_sample`,
/// // where `n_sample = data.shape().0` is 
/// // the number of training examples in `data`.
/// let booster = GBM::init_with_loss(&sample, GBMLoss::L2);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = RegressionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .loss(LossType::L2)
///     .build();
/// 
/// // Run `GBM` and obtain the resulting hypothesis `f`.
/// let f = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions = f.predict_all(&data);
/// 
/// // Get the number of training examples.
/// let n_sample = data.shape().0 as f64;
/// 
/// // Calculate the L1-training loss.
/// let target = sample.target();
/// let training_loss = sample.target()
///     .into_iter()
///     .zip(predictions)
///     .map(|(y, fx)| (y - fx).powi(2))
///     .sum::<f64>()
///     / n_sample;
/// 
///
/// println!("Training Loss is: {training_loss}");
/// ```
pub struct GBM<'a, F, L> {
    // Training data
    sample: &'a Sample,


    // Tolerance parameter
    tolerance: f64,

    // Weights on hypotheses
    weights: Vec<f64>,

    // Hypohteses obtained by the weak-learner.
    hypotheses: Vec<F>,


    // Some struct that implements `LossFunction` trait
    loss: L,


    // Max iteration until GBM guarantees the optimality.
    max_iter: usize,

    // Terminated iteration.
    // GBM terminates in eary step 
    // if the training set is linearly separable.
    terminated: usize,


    // A prediction vector at a state.
    predictions: Vec<f64>,
}




impl<'a, F, L> GBM<'a, F, L>
{
    /// Initialize the `GBM`.
    /// This method sets some parameters `GBM` holds.
    pub fn init_with_loss(sample: &'a Sample, loss: L) -> Self {

        let n_sample = sample.shape().0;
        let predictions = vec![0.0; n_sample];

        Self {
            sample,
            tolerance: 0.0,

            weights: Vec::new(),
            hypotheses: Vec::new(),

            loss,

            max_iter: 100,

            terminated: usize::MAX,

            predictions,
        }
    }
}


impl<'a, F, L> GBM<'a, F, L> {
    /// Returns the maximum iteration
    /// of the `GBM` to find a combined hypothesis
    /// that has error at most `tolerance`.
    /// Default max loop is `100`.
    pub fn max_loop(&self) -> usize {
        let n_sample = self.sample.shape().0 as f64;

        (n_sample.ln() / self.tolerance.powi(2)) as usize
    }


    /// Set the tolerance parameter.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }


    /// Set the Loss Type.
    pub fn loss(mut self, loss_type: L) -> Self {
        self.loss = loss_type;
        self
    }
}


impl<F, L> Booster<F> for GBM<'_, F, L>
    where F: Regressor + Clone,
          L: LossFunction,
{
    type Output = WeightedMajority<F>;


    fn name(&self) -> &str {
        "Gradient Boosting Machine"
    }


    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_sample, n_feature) = self.sample.shape();
        let info = Vec::from([
            ("# of examples", format!("{n_sample}")),
            ("# of features", format!("{n_feature}")),
            ("Tolerance", format!("{}", self.tolerance)),
            ("Loss", format!("{}", self.loss.name())),
            ("Max iteration", format!("{}", self.max_iter)),
        ]);
        Some(info)
    }


    fn preprocess<W>(
        &mut self,
        _weak_learner: &W,
    )
        where W: WeakLearner<Hypothesis = F>
    {
        // Initialize parameters
        let n_sample = self.sample.shape().0;

        self.weights = Vec::with_capacity(self.max_iter);
        self.hypotheses = Vec::with_capacity(self.max_iter);


        self.terminated = self.max_iter;
        self.predictions = vec![0.0; n_sample];
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
        let h = weak_learner.produce(self.sample, &self.predictions[..]);

        let predictions = h.predict_all(self.sample);
        let coef = self.loss.best_coefficient(
            &self.sample.target(), &predictions[..]
        );

        // If the best coefficient is zero,
        // the newly-attained hypothesis `h` do nothing.
        // Thus, we can terminate the boosting at this point.
        if coef == 0.0 {
            self.terminated = iteration;
            return ControlFlow::Break(iteration);
        }


        self.weights.push(coef);
        self.hypotheses.push(h);


        self.predictions.par_iter_mut()
            .zip(predictions)
            .for_each(|(p, q)| { *p += coef * q; });

        ControlFlow::Continue(())
    }


    fn postprocess<W>(
        &mut self,
        _weak_learner: &W,
    ) -> Self::Output
        where W: WeakLearner<Hypothesis = F>
    {
        WeightedMajority::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}


