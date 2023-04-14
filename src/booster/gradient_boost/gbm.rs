//! Provides [`GBM`](GBM) by Friedman, 2001.

use crate::{
    common::loss_functions::*,
    Sample,
    Booster,
    WeakLearner,
    Regressor,
    CombinedHypothesis
};

use std::ops::ControlFlow;


/// Defines `GBM`.
/// This struct is based on the book: 
/// [Greedy Function Approximation: A Gradient Boosting Machine](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boostingmachine/10.1214/aos/1013203451.full)
/// by Jerome H. Friedman, 2001.
/// 
/// # Example
/// The following code shows a small example 
/// for running [`GBM`](GBM).  
/// See also:
/// - [`Regressor`]
/// - [`CombinedHypothesis<F>`]
/// 
/// [`RTree`]: crate::weak_learner::RTree
/// [`RTreeRegressor`]: crate::weak_learner::RTreeRegressor
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
/// // Initialize `GBM` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis whose training error is
/// // less than `0.01` if the traing examples are linearly separable.
/// // Note that the default tolerance parameter is set as `1 / n_sample`,
/// // where `n_sample = data.shape().0` is 
/// // the number of training examples in `data`.
/// let booster = GBM::init(&sample)
///     .loss(GBMLoss::L1);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = RTree::init(&sample)
///     .max_depth(1)
///     .loss_type(LossType::L1);
/// 
/// // Run `GBM` and obtain the resulting hypothesis `f`.
/// let f: CombinedHypothesis<RTreeRegressor> = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions: Vec<f64> = f.predict_all(&data);
/// 
/// // Get the number of training examples.
/// let n_sample = data.shape().0 as f64;
/// 
/// // Calculate the L1-training loss.
/// let target = sample.target();
/// let training_loss = target.into_iter()
///     .zip(predictions)
///     .map(|(y, fx) (y - fx).abs())
///     .sum::<f64>()
///     / n_sample;
/// 
///
/// println!("Training Loss is: {training_loss}");
/// ```
pub struct GBM<'a, F> {
    // Training data
    sample: &'a Sample,


    // Original labels
    original_target: Vec<f64>,

    // Distribution on examples.
    // Since GBM does not maintain a distribution over examples,
    // we use all-one vector.
    ones: Vec<f64>,

    // Tolerance parameter
    tolerance: f64,

    // Weights on hypotheses
    weights: Vec<f64>,

    // Hypohteses obtained by the weak-learner.
    hypotheses: Vec<F>,


    // Some struct that implements `LossFunction` trait
    loss: GBMLoss,


    // Max iteration until GBM guarantees the optimality.
    max_iter: usize,

    // Terminated iteration.
    // GBM terminates in eary step 
    // if the training set is linearly separable.
    terminated: usize,
}




impl<'a, F> GBM<'a, F>
{
    /// Initialize the `GBM`.
    /// This method sets some parameters `GBM` holds.
    pub fn init(sample: &'a Sample) -> Self {

        let n_sample = sample.shape().0;
        let original_target = sample.target().to_vec();


        Self {
            sample,
            original_target,
            tolerance: 0.0,

            weights: Vec::new(),
            hypotheses: Vec::new(),

            ones: vec![1.0; n_sample],

            loss: GBMLoss::L2,

            max_iter: 100,

            terminated: usize::MAX,
        }
    }
}


impl<'a, F> GBM<'a, F> {
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
    pub fn loss(mut self, loss_type: GBMLoss) -> Self {
        self.loss = loss_type;
        self
    }
}


impl<F> Booster<F> for GBM<'_, F>
    where F: Regressor + Clone,
{
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

        if self.original_target.is_empty() {
            self.original_target = self.sample.target().to_vec();
        }

        self.ones = vec![1.0; n_sample];


        self.terminated = self.max_iter;
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
        let h = weak_learner.produce(self.sample, &self.ones[..]);

        let predictions = h.predict_all(self.sample);
        let coef = {
            let target = self.sample.target();
            self.loss.best_coefficient(&target[..], &predictions[..])
        };

        // If the best coefficient is zero,
        // the newly-attained hypothesis `h` do nothing.
        // Thus, we can terminate the boosting at this point.
        if coef == 0.0 {
            self.terminated = iteration;
            return ControlFlow::Break(iteration);
        }

        // Update the residual vector
        self.sample.target_mut()
            .iter_mut()
            .zip(predictions)
            .for_each(|(r, p)| {
                *r -= coef * p;
            });


        self.weights.push(coef);
        self.hypotheses.push(h);

        ControlFlow::Continue(())
    }


    fn postprocess<W>(
        &mut self,
        _weak_learner: &W,
    ) -> CombinedHypothesis<F>
        where W: WeakLearner<Hypothesis = F>
    {
        // Back the original label to `Sample`.
        self.sample.target_mut()
            .iter_mut()
            .zip(self.original_target.iter().copied())
            .for_each(|(y, orig)| { *y = orig; });

        CombinedHypothesis::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}


