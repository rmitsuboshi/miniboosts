//! This file defines `CERLPBoost` based on the paper
//! "On the Equivalence of Weak Learnability and Linaer Separability:
//!     New Relaxations and Efficient Boosting Algorithms"
//! by Shai Shalev-Shwartz and Yoram Singer.
//! I named this algorithm `CERLPBoost`
//! since it is referred as `the Corrective version of CERLPBoost`
//! in "Entropy Regularized LPBoost" by Warmuth et al.
//!
use std::mem;

use crate::{
    Sample,
    Booster,
    WeakLearner,

    Classifier,
    CombinedHypothesis,
    common::utils,
    common::checker,
    common::frank_wolfe::{FrankWolfe, FWType},
    research::Research,
};

use std::ops::ControlFlow;

/// The Corrective ERLPBoost algorithm, proposed in the following paper:
/// 
/// [Shai Shalev-Shwartz and Yoram Singer - On the equivalence of weak learnability and linear separability: new relaxations and efficient boosting algorithms](https://link.springer.com/article/10.1007/s10994-010-5173-z)
/// 
/// Corrective ERLPBoost aims to optimize soft-margin 
/// without using LP/QP solver.
/// ## Strength
/// - Running time per round is 
///   the fastest among soft-margin boosting algorithms.
/// - The iteration bound is the same as the one to ERLPBoost.
/// ## Weakness
/// - Empirically, the number of rounds tend to huge compared to
///   totally corrective algorithms such as [`ERLPBoost`] and [`LPBoost`].
/// 
/// # Example
/// The following code shows a small example 
/// for running [`CERLPBoost`].  
/// See also:
/// - [`CERLPBoost::nu`]
/// - [`DecisionTree`]
/// - [`DecisionTreeClassifier`]
/// - [`CombinedHypothesis<F>`]
/// 
/// [`CERLPBoost::nu`]: CERLPBoost::nu
/// [`DecisionTree`]: crate::weak_learner::DecisionTree
/// [`DecisionTreeClassifier`]: crate::weak_learner::DecisionTreeClassifier
/// [`CombinedHypothesis<F>`]: crate::hypothesis::CombinedHypothesis
/// [`LPBoost`]: crate::prelude::LPBoost
/// [`ERLPBoost`]: crate::prelude::ERLPBoost
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
/// 
/// // Get the number of training examples.
/// let n_sample = sample.shape().0 as f64;
/// 
/// // Set the upper-bound parameter of outliers in `sample`.
/// // Here we assume that the outliers are at most 10% of `sample`.
/// let nu = 0.1 * n_sample;
/// 
/// // Initialize `CERLPBoost` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis whose training error is
/// // less than `0.01` if the traing examples are linearly separable.
/// let mut booster = CERLPBoost::init(&sample)
///     .tolerance(0.01)
///     .nu(0.1 * n_sample);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .criterion(Criterion::Entropy)
///     .build();
/// 
/// // Run `CERLPBoost` and obtain the resulting hypothesis `f`.
/// let f = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions = f.predict_all(&sample);
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
pub struct CERLPBoost<'a, F> {
    // Training sample
    sample: &'a Sample,

    dist: Vec<f64>,
    // A regularization parameter defined in the paper
    eta: f64,

    half_tolerance: f64,
    nu: f64,

    frank_wolfe: FrankWolfe,
    weights: Vec<f64>,
    hypotheses: Vec<F>,

    max_iter: usize,
    terminated: usize,
}

impl<'a, F> CERLPBoost<'a, F> {
    /// Construct a new instance of `CERLPBoost`.
    /// 
    /// Time complexity: `O(1)`.
    pub fn init(sample: &'a Sample) -> Self {
        let n_sample = sample.shape().0;


        // Set tolerance, sub_tolerance
        let half_tolerance = 0.005;

        // Set regularization parameter
        let nu = 1.0;
        let eta = (n_sample as f64 / nu).ln() / half_tolerance;

        let frank_wolfe = FrankWolfe::new(eta, nu, FWType::ShortStep);

        Self {
            sample,

            dist: Vec::new(),
            half_tolerance,
            eta,
            nu: 1.0,

            frank_wolfe,

            weights: Vec::new(),
            hypotheses: Vec::new(),
            max_iter: usize::MAX,
            terminated: usize::MAX,
        }
    }


    /// This method updates the capping parameter.
    /// 
    /// Time complexity: `O(1)`.
    pub fn nu(mut self, nu: f64) -> Self {
        let (n_sample, _) = self.sample.shape();
        checker::check_nu(nu, n_sample);
        self.nu = nu;
        self.frank_wolfe.nu(self.nu);

        self.regularization_param();

        self
    }


    /// Update tolerance parameter `half_tolerance`.
    /// 
    /// Time complexity: `O(1)`.
    #[inline(always)]
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.half_tolerance = tolerance / 2.0;
        self
    }


    /// Set the Frank-Wolfe step size strategy.
    #[inline(always)]
    pub fn fw_type(mut self, fw_type: FWType) -> Self {
        self.frank_wolfe.fw_type(fw_type);
        self
    }


    /// Update regularization parameter.
    /// (the regularization parameter on `self.tolerance` and `self.nu`.)
    /// 
    /// Time complexity: `O(1)`.
    #[inline(always)]
    fn regularization_param(&mut self) {
        let m = self.dist.len() as f64;
        let ln_part = (m / self.nu).ln();
        self.eta = ln_part / self.half_tolerance;

        self.frank_wolfe.eta(self.eta);
    }


    /// returns the maximum iteration of the CERLPBoost
    /// to find a combined hypothesis that has error at most `tolerance`.
    /// 
    /// Time complexity: `O(1)`.
    pub fn max_loop(&mut self) -> usize {

        let m = self.dist.len() as f64;

        let ln_m = (m / self.nu).ln();
        let max_iter = 8.0 * ln_m / self.half_tolerance.powi(2);

        max_iter.ceil() as usize
    }
}


impl<F> CERLPBoost<'_, F>
    where F: Classifier + PartialEq,
{
    /// Updates weight on hypotheses and `self.dist` in this order.
    fn update_distribution_mut(&mut self) {
        self.dist = utils::exp_distribution(
            self.eta, self.nu, self.sample,
            &self.weights[..], &self.hypotheses[..],
        );
    }
}

impl<F> Booster<F> for CERLPBoost<'_, F>
    where F: Classifier + Clone + PartialEq + std::fmt::Debug,
{
    type Output = CombinedHypothesis<F>;


    fn name(&self) -> &str {
        "Corrective ERLPBoost"
    }


    fn preprocess<W>(
        &mut self,
        _weak_learner: &W,
    )
        where W: WeakLearner<Hypothesis = F>
    {
        self.sample.is_valid_binary_instance();
        let n_sample = self.sample.shape().0;
        let uni = 1.0 / n_sample as f64;

        self.dist = vec![uni; n_sample];


        self.regularization_param();
        self.max_iter = self.max_loop();
        self.terminated = self.max_iter;

        // self.classifiers = Vec::new();
        self.weights = Vec::new();
        self.hypotheses = Vec::new();
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

        // Update the distribution over examples
        self.update_distribution_mut();

        // Receive a hypothesis from the base learner
        let h = weak_learner.produce(self.sample, &self.dist);

        let new_edge = utils::edge_of_hypothesis(
            self.sample, &self.dist[..], &h
        );

        let old_edge = utils::edge_of_weighted_hypothesis(
            self.sample, &self.dist[..],
            &self.weights[..], &self.hypotheses[..]
        );

        // let old_confidences = utils::margins_of_weighted_hypothesis(
        //     self.sample, &self.weights[..], &self.hypotheses[..],
        // );
        // let new_confidences = utils::margins_of_hypothesis(
        //     self.sample, &h
        // );
        // let target = self.sample.target();
        // let gap_vec = new_confidences.into_iter()
        //     .zip(old_confidences)
        //     .zip(target.into_iter())
        //     .map(|((n, o), y)| *y * (n - o))
        //     .collect::<Vec<_>>();

        // // Compute the difference between the new hypothesis
        // // and the current combined hypothesis
        // let diff = gap_vec.par_iter()
        //     .zip(&self.dist[..])
        //     .map(|(v, d)| v * d)
        //     .sum::<f64>();

        let diff = new_edge - old_edge;

        // Update the parameters
        if diff <= self.half_tolerance {
            self.terminated = iteration;
            return ControlFlow::Break(iteration);
        }


        let pos = self.hypotheses.iter()
            .position(|f| *f == h)
            .unwrap_or(self.hypotheses.len());

        if pos == self.hypotheses.len() {
            self.hypotheses.push(h);
            self.weights.push(0.0);
        }

        let weights = mem::take(&mut self.weights);
        // Update the weight on hypotheses
        self.weights = self.frank_wolfe.next_iterate(
            iteration, self.sample, &self.dist[..],
            &self.hypotheses[..], pos, weights,
        );

        ControlFlow::Continue(())
    }


    fn postprocess<W>(
        &mut self,
        _weak_learner: &W,
    ) -> Self::Output
        where W: WeakLearner<Hypothesis = F>
    {
        CombinedHypothesis::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}


impl<H> Research for CERLPBoost<'_, H>
    where H: Classifier + Clone,
{
    type Output = CombinedHypothesis<H>;
    fn current_hypothesis(&self) -> Self::Output {
        CombinedHypothesis::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}
