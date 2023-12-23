//! This file defines `TotalBoost` based on the paper
//! "Totally Corrective Boosting Algorithms that Maximize the Margin"
//! by Warmuth et al.
//! 
use crate::{
    Sample,
    Booster,
    WeakLearner,

    Classifier,
    CombinedHypothesis,

    SoftBoost,

    research::Research,
};

use std::ops::ControlFlow;


/// The TotalBoost algorithm proposed in the following paper:
/// [Manfred K. Warmuth, Jun Liao, and Gunnar Rätsch - Totally corrective boosting algorithms that maximize the margin](https://dl.acm.org/doi/10.1145/1143844.1143970)
///
/// Given a set `{(x_{1}, y_{1}), (x_{2}, y_{2}), ..., (x_{m}, y_{m})}`
/// of training examples,
/// [`TotalBoost`] aims to find an optimal solution of
/// the hard-margin optimization problem:
///
/// ```txt
/// max ρ
/// ρ,w
/// s.t. y_{i} Σ_{h ∈ Δ_{H}} w_{h} h(x_{i}) ≥ ρ, for all i ∈ [m],
///      w ∈ Δ_{H}
/// ```
/// 
/// # Convergence rate
/// Assume that there exists a convex combination of hypotheses
/// that perfectly classifies the training examples:
///
/// ```txt
/// ∃ w ∈ Δ_{h},
/// ∀ (x, y) in training examples,
/// y Σ_{h ∈ H} w_{h} h( x ) > 0.
/// ```
///
/// Given a set of training examples of size `m > 0`
/// and an accuracy parameter `ε > 0`,
/// `TotalBoost` finds an `ε`-approximate solution of
/// the hard-margin optimization problem
/// in `o( ln(m) / ε² )` iterations.
/// 
/// # Related information
/// - [`TotalBoost`] is a special case of [`SoftBoost`].
///   That is, `TotalBoost` restricts [`SoftBoost::nu`] as `1.0`.  
///   For this reason, [`TotalBoost`] is 
///   just a wrapper of [`SoftBoost`].
///
/// 
/// # Example
/// The following code shows 
/// a small example for running [`TotalBoost`].  
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
/// 
/// // Get the number of training examples.
/// let n_sample = sample.shape().0 as f64;
/// 
/// // Initialize `TotalBoost` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis whose training error is
/// // less than `0.01` if the traing examples are linearly separable.
/// let mut booster = TotalBoost::init(&sample)
///     .tolerance(0.01);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .criterion(Criterion::Entropy)
///     .build();
/// 
/// // Run `TotalBoost` and obtain the resulting hypothesis `f`.
/// let f = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions = f.predict_all(&sample);
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
pub struct TotalBoost<'a, F> {
    softboost: SoftBoost<'a, F>,
}


impl<'a, F> TotalBoost<'a, F>
    where F: Classifier,
{
    /// Construct a new instance of `TotalBoost`.
    /// 
    /// Time complexity: `O(1)`.
    pub fn init(sample: &'a Sample) -> Self {
        let softboost = SoftBoost::init(sample)
            .nu(1.0);

        Self { softboost }
    }


    /// Set the tolerance parameter.
    /// 
    /// Time complexity: `O(1)`.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.softboost = self.softboost.tolerance(tol);
        self
    }
}


impl<F> Booster<F> for TotalBoost<'_, F>
    where F: Classifier + Clone,
{
    type Output = CombinedHypothesis<F>;


    fn name(&self) -> &str {
        "TotalBoost"
    }


    fn info(&self) -> Option<Vec<(&str, String)>> {
        self.softboost.info()
    }


    fn preprocess<W>(
        &mut self,
        weak_learner: &W,
    )
        where W: WeakLearner<Hypothesis = F>
    {
        self.softboost.preprocess(weak_learner);
    }


    fn boost<W>(
        &mut self,
        weak_learner: &W,
        iteration: usize,
    ) -> ControlFlow<usize>
        where W: WeakLearner<Hypothesis = F>
    {
        self.softboost.boost(weak_learner, iteration)
    }


    fn postprocess<W>(
        &mut self,
        weak_learner: &W,
    ) -> Self::Output
        where W: WeakLearner<Hypothesis = F>
    {
        self.softboost.postprocess(weak_learner)
    }
}

impl<H> Research for TotalBoost<'_, H>
    where H: Classifier + Clone,
{
    type Output = CombinedHypothesis<H>;
    fn current_hypothesis(&self) -> Self::Output {
        self.softboost.current_hypothesis()
    }
}


