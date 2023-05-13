//! This file defines `SmoothBoost` based on the paper
//! ``Smooth Boosting and Learning with Malicious Noise''
//! by Rocco A. Servedio.


use rayon::prelude::*;

use crate::{
    Sample,
    Booster,
    WeakLearner,

    Classifier,
    CombinedHypothesis,

    research::Research,
};

use std::ops::ControlFlow;


/// `SmoothBoost`.
/// Variable names, such as `kappa`, `gamma`, and `theta`, 
/// come from the original paper.  
/// **Note that** `SmoothBoost` needs to know 
/// the weak learner guarantee `gamma`.  
/// See Figure 1 in this paper: 
/// [Smooth Boosting and Learning with Malicious Noise](https://link.springer.com/chapter/10.1007/3-540-44581-1_31) by Rocco A. Servedio.
/// 
/// # Example
/// The following code shows a small example 
/// for running [`SmoothBoost`](SmoothBoost).  
/// See also:
/// - [`SmoothBoost::tolerance`]
/// - [`SmoothBoost::gamma`]
/// - [`DecisionTree`]
/// - [`DecisionTreeClassifier`]
/// - [`CombinedHypothesis<F>`]
/// 
/// [`SmoothBoost::tolerance`]: SmoothBoost::tolerance
/// [`SmoothBoost::gamma`]: SmoothBoost::gamma
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
/// // Get the number of training examples.
/// let n_sample = sample.shape().0 as f64;
/// 
/// // Initialize `SmoothBoost` and 
/// // set the weak learner guarantee `gamma` as `0.05`.
/// // For this case, weak learner returns a hypothesis
/// // that returns a hypothesis with weighted loss 
/// // at most `0.45 = 0.5 - 0.05`.
/// let booster = SmoothBoost::init(&sample)
///     .tolerance(0.01)
///     .gamma(0.05);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTree::init(&sample)
///     .max_depth(2)
///     .criterion(Criterion::Edge);
/// 
/// // Run `SmoothBoost` and obtain the resulting hypothesis `f`.
/// let f: CombinedHypothesis<DecisionTreeClassifier> = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions: Vec<i64> = f.predict_all(&sample);
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
pub struct SmoothBoost<'a, F> {
    // Training sample
    sample: &'a Sample,

    /// Desired accuracy
    kappa: f64,

    /// Desired margin for the final hypothesis.
    /// To guarantee the convergence rate, `theta` should be
    /// `gamma / (2.0 + gamma)`.
    theta: f64,

    /// Weak-learner guarantee;
    /// for any distribution over the training examples,
    /// the weak-learner returns a hypothesis
    /// with edge at least `gamma`.
    gamma: f64,

    /// The number of training examples.
    n_sample: usize,

    current: usize,

    /// Terminated iteration.
    terminated: usize,

    max_iter: usize,

    hypotheses: Vec<F>,


    m: Vec<f64>,
    n: Vec<f64>,
}


impl<'a, F> SmoothBoost<'a, F> {
    /// Initialize `SmoothBoost`.
    pub fn init(sample: &'a Sample) -> Self {
        let n_sample = sample.shape().0;

        let gamma = 0.5;


        Self {
            sample,

            kappa: 0.5,
            theta: gamma / (2.0 + gamma), // gamma / (2.0 + gamma)
            gamma,

            n_sample,

            current: 0_usize,

            terminated: usize::MAX,
            max_iter: usize::MAX,

            hypotheses: Vec::new(),

            m: Vec::new(),
            n: Vec::new(),
        }
    }


    /// Set the tolerance parameter `kappa`.
    #[inline(always)]
    pub fn tolerance(mut self, kappa: f64) -> Self {
        self.kappa = kappa;

        self
    }


    /// Set the parameter `gamma`.
    /// `gamma` is the weak learner guarantee;  
    /// `SmoothBoost` assumes the weak learner to returns a hypothesis `h`
    /// such that
    /// `0.5 * sum_i D[i] |h(x[i]) - y[i]| <= 0.5 - gamma`
    /// for the given distribution.  
    /// **Note that** this is an extremely assumption.
    #[inline(always)]
    pub fn gamma(mut self, gamma: f64) -> Self {
        // `gamma` must be in [0.0, 0.5)
        assert!((0.0..0.5).contains(&gamma));
        self.gamma = gamma;

        self
    }


    /// Set the parameter `theta`.
    fn theta(&mut self) {
        self.theta = self.gamma / (2.0 + self.gamma);
    }


    /// Returns the maximum iteration
    /// of SmoothBoost to satisfy the stopping criterion.
    fn max_loop(&self) -> usize {
        let denom = self.kappa
            * self.gamma.powi(2)
            * (1.0 - self.gamma).sqrt();


        (2.0 / denom).ceil() as usize
    }


    fn check_preconditions(&self) {
        // Check `kappa`.
        if !(0.0..1.0).contains(&self.kappa) || self.kappa <= 0.0 {
            panic!(
                "Invalid kappa. \
                 The parameter `kappa` must be in (0.0, 1.0)"
            );
        }

        // Check `gamma`.
        if !(self.theta..0.5).contains(&self.gamma) {
            panic!(
                "Invalid gamma. \
                 The parameter `gamma` must be in [self.theta, 0.5)"
            );
        }
    }
}



impl<F> Booster<F> for SmoothBoost<'_, F>
    where F: Classifier + Clone,
{
    fn preprocess<W>(
        &mut self,
        _weak_learner: &W,
    )
        where W: WeakLearner<Hypothesis = F>
    {
        self.sample.is_valid_binary_instance();
        self.n_sample = self.sample.shape().0;
        // Set the paremeter `theta`.
        self.theta();

        // Check whether the parameter satisfies the pre-conditions.
        self.check_preconditions();


        self.current = 0_usize;
        self.max_iter = self.max_loop();
        self.terminated = self.max_iter;

        self.hypotheses = Vec::new();


        self.m = vec![1.0; self.n_sample];
        self.n = vec![1.0; self.n_sample];
    }


    fn boost<W>(
        &mut self,
        weak_learner: &W,
        iteration: usize,
    ) -> ControlFlow<usize>
        where W: WeakLearner<Hypothesis = F>
    {

        if self.max_iter < iteration {
            return ControlFlow::Break(self.max_iter);
        }

        self.current = iteration;


        let sum = self.m.iter().sum::<f64>();
        // Check the stopping criterion.
        if sum < self.n_sample as f64 * self.kappa {
            self.terminated = iteration - 1;
            return ControlFlow::Break(iteration);
        }


        // Compute the distribution.
        let dist = self.m.iter()
            .map(|mj| *mj / sum)
            .collect::<Vec<_>>();


        // Call weak learner to obtain a hypothesis.
        self.hypotheses.push(
            weak_learner.produce(self.sample, &dist[..])
        );
        let h: &F = self.hypotheses.last().unwrap();


        let target = self.sample.target();
        let margins = target.iter()
            .enumerate()
            .map(|(i, y)| y * h.confidence(self.sample, i));


        // Update `n`
        self.n.iter_mut()
            .zip(margins)
            .for_each(|(nj, yh)| {
                *nj = *nj + yh - self.theta;
            });


        // Update `m`
        self.m.par_iter_mut()
            .zip(&self.n[..])
            .for_each(|(mj, nj)| {
                if *nj <= 0.0 {
                    *mj = 1.0;
                } else {
                    *mj = (1.0 - self.gamma).powf(*nj * 0.5);
                }
            });

        ControlFlow::Continue(())
    }


    fn postprocess<W>(
        &mut self,
        _weak_learner: &W,
    ) -> CombinedHypothesis<F>
        where W: WeakLearner<Hypothesis = F>
    {
        let weight = 1.0 / self.terminated as f64;
        let weights = vec![weight; self.n_sample];
        CombinedHypothesis::from_slices(&weights[..], &self.hypotheses[..])
    }
}


impl<H> Research<H> for SmoothBoost<'_, H>
    where H: Classifier + Clone,
{
    fn current_hypothesis(&self) -> CombinedHypothesis<H> {
        let weight = 1.0 / self.terminated as f64;
        let weights = vec![weight; self.n_sample];
        CombinedHypothesis::from_slices(&weights[..], &self.hypotheses[..])
    }
}
