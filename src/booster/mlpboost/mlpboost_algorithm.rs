//! This file defines `MLPBoost` based on the paper
//! [Boosting as Frank-Wolfe](https://arxiv.org/abs/2209.10831).
//! by Mitsuboshi et al.
//! 

use super::lp_model::LPModel;


use crate::{
    Sample,
    Booster,
    WeakLearner,

    Classifier,
    CombinedHypothesis,
    common::{
        utils,
        checker,
        frank_wolfe::{FrankWolfe, FWType},
    },
    research::Research,
};


use std::mem;
use std::cell::RefCell;
use std::ops::ControlFlow;



/// The MLPBoost algorithm, shorthand of Modified LPBoost algorithm,
/// proposed in the following paper:
/// 
/// [Ryotaro Mitsuboshi, Kohei Hatano, and Eiji Takimoto - Boosting as Frank-Wolfe](https://arxiv.org/abs/2209.10831)
/// 
/// MLPBoost is an abstraction of soft-margin boosting algorithms
/// in terms of Frank-Wolfe algorithm.
/// 
/// 
/// # Example
/// The following code shows a small example 
/// for running [`MLPBoost`](MLPBoost).  
/// See also:
/// - [`MLPBoost::nu`]
/// - [`DecisionTree`]
/// - [`DecisionTreeClassifier`]
/// - [`CombinedHypothesis<F>`]
/// 
/// [`MLPBoost::nu`]: MLPBoost::nu
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
///     .expect("Failed to read the training sample")
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
/// // Initialize `MLPBoost` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis whose training error is
/// // less than `0.01` if the traing examples are linearly separable.
/// // Note that the default tolerance parameter is set as `1 / n_sample`,
/// // where `n_sample = sample.shape().0` is 
/// // the number of training examples in `sample`.
/// let mut booster = MLPBoost::init(&sample)
///     .tolerance(0.01)
///     .frank_wolfe(FWType::ShortStep)
///     .nu(0.1 * n_sample);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTreeBuilder::new(&train)
///     .max_depth(2)
///     .criterion(Criterion::Entropy)
///     .build();
/// 
/// // Run `MLPBoost` and obtain the resulting hypothesis `f`.
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
pub struct MLPBoost<'a, F> {
    // Training sample
    sample: &'a Sample,


    // Tolerance parameter
    half_tolerance: f64,


    // Number of examples
    n_sample: usize,


    // Capping parameter
    nu: f64,


    // Regularization parameter.
    eta: f64,


    // Primary (FW) update
    primary: FrankWolfe,

    // Secondary (LPBoost) update
    secondary: Option<RefCell<LPModel>>,


    // Weights on hypotheses
    weights: Vec<f64>,

    // Hypotheses
    hypotheses: Vec<F>,


    terminated: usize,
    max_iter: usize,


    gamma: f64,
}


impl<'a, F> MLPBoost<'a, F> {
    /// Construct a new instance of `MLPBoost`.
    /// 
    /// Time complexity: `O(1)`.
    pub fn init(sample: &'a Sample) -> Self {
        let n_sample = sample.shape().0;
        assert!(n_sample != 0);


        let half_tolerance = 0.005;
        let nu  = 1.0;
        let eta = (n_sample as f64 / nu).ln() / half_tolerance;

        let primary = FrankWolfe::new(eta, nu, FWType::ShortStep);

        Self {
            sample,

            half_tolerance,
            n_sample,
            nu,
            eta,

            primary,
            secondary: None,

            weights: Vec::new(),
            hypotheses: Vec::new(),

            terminated: usize::MAX,
            max_iter: usize::MAX,

            gamma: 1.0,
        }
    }


    /// This method updates the capping parameter.
    /// This parameter must be in `[1, # of training examples]`.
    /// 
    /// Time complexity: `O(1)`.
    pub fn nu(mut self, nu: f64) -> Self {
        assert!(1.0 <= nu && nu <= self.n_sample as f64);
        self.nu = nu;
        self.primary.nu(self.nu);

        self
    }


    /// Set the Frank-Wolfe rule.
    /// See [`FWType`](FWType).
    /// 
    /// Time complexity: `O(1)`.
    pub fn frank_wolfe(mut self, fw_type: FWType) -> Self {
        self.primary.fw_type(fw_type);
        self
    }


    /// Set the tolerance parameter.
    /// 
    /// Time complexity: `O(1)`.
    #[inline(always)]
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.half_tolerance = tolerance / 2.0;
        self
    }


    /// Set the regularization parameter.
    /// 
    /// Time complexity: `O(1)`.
    #[inline(always)]
    fn eta(&mut self) {
        let ln_m = (self.n_sample as f64 / self.nu).ln();
        self.eta = ln_m / self.half_tolerance;
        self.primary.eta(self.eta);
    }


    /// Initialize the LP solver.
    /// 
    /// Time complexity: `O( # of training examples )`.
    fn init_solver(&mut self) {
        // `ub` is the upper-bound of distribution for each example.
        let ub = 1.0 / self.nu;

        let lp_model = RefCell::new(LPModel::init(self.n_sample, ub));

        self.secondary = Some(lp_model);
    }


    /// Initialize all parameters.
    /// The methods `self.tolerance(..)`, `self.eta(..)`, and
    /// `self.init_solver(..)` are accessed only via this method.
    /// 
    /// Time complexity: `O( # of training examples )`.
    fn init_params(&mut self) {
        // Set the regularization parameter.
        self.eta();

        // Initialize the solver.
        self.init_solver();
    }


    /// Returns the maximum iterations 
    /// to obtain the solution with accuracy `self.half_tolerance`.
    /// 
    /// Time complexity: `O(1)`.
    pub fn max_loop(&self) -> usize {
        let ln_m = (self.n_sample as f64 / self.nu).ln();
        (8.0_f64 * ln_m / self.half_tolerance.powi(2)).ceil() as usize
    }


    /// Returns the terminated iteration.
    /// This method returns `0` before the boosting step.
    /// 
    /// Time complexity: `O(1)`.
    pub fn terminated(&self) -> usize {
        self.terminated
    }
}

impl<F> MLPBoost<'_, F>
    where F: Classifier,
{
    fn secondary_update(&self, opt_h: Option<&F>) -> Vec<f64> {
        self.secondary.as_ref()
            .expect("Failed to call `.as_ref()` to `self.secondary`")
            .borrow_mut()
            .update(self.sample, opt_h)
    }


    /// Returns the smoothed objective value 
    /// `-f*` at the current weighting `self.weights`.
    /// 
    /// ```text
    /// - max [ - d^T Aw - sum_i [ di ln( di ) ] ]
    /// s.t. sum_i di = 1, 0 <= di <= 1 / self.nu, for all i <= m.
    ///  ^
    ///  |
    ///  v
    /// min [ d^T Aw + sum_i [ di ln( di ) ] ]
    /// s.t. sum_i di = 1, 0 <= di <= 1 / self.nu, for all i <= m.
    /// ```
    /// 
    /// Time complexity: `O( # of training examples )`.
    fn objval(&self, weights: &[f64]) -> f64 {

        let dist = utils::exp_distribution(
            self.eta, self.nu, self.sample, weights, &self.hypotheses,
        );

        let edge = utils::edge_of_weighted_hypothesis(
            self.sample, &dist[..], weights, &self.hypotheses[..],
        );

        let entropy = utils::entropy_from_uni_distribution(&dist[..]);

        edge + (entropy / self.eta)
    }


    /// Choose the better weights
    /// by comparing the smoothed objective value.
    /// Since MLPBoost maximizes `-f*`,
    /// this method picks the one that yields the better value.
    /// 
    /// Time complexity: `O( # of training examples )`.
    fn better_weight(&mut self, w1: Vec<f64>, w2: Vec<f64>)
    {
        let v1 = self.objval(&w1[..]);
        let v2 = self.objval(&w2[..]);

        self.weights = if v1 >= v2 { w1 } else { w2 };
    }
}


impl<F> Booster<F> for MLPBoost<'_, F>
    where F: Classifier + Clone + PartialEq,
{
    type Output = CombinedHypothesis<F>;


    fn name(&self) -> &str {
        "MLPBoost"
    }


    fn preprocess<W>(
        &mut self,
        _weak_learner: &W,
    )
        where W: WeakLearner<Hypothesis = F>
    {
        self.sample.is_valid_binary_instance();
        self.n_sample = self.sample.shape().0;

        self.init_params();

        self.max_iter = self.max_loop();
        self.terminated = self.max_iter;


        self.hypotheses = Vec::new();
        self.weights = Vec::new();

        // Upper-bound of the optimal `edge`.
        self.gamma = 1.0;
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

        // ------------------------------------------------------

        // Compute the distribution over training instances.
        let dist = utils::exp_distribution(
            self.eta, self.nu,
            self.sample, &self.weights[..], &self.hypotheses[..],
        );


        // Obtain a hypothesis w.r.t. `dist`.
        let h = weak_learner.produce(self.sample, &dist);


        // Compute the edge of newly-attained hypothesis `h`.
        let edge_h = utils::edge_of_hypothesis(self.sample, &dist, &h);


        // Update the estimation of `edge`.
        self.gamma = self.gamma.min(edge_h);



        // For the first iteration,
        // just append the hypothesis and continue.
        if iteration == 1 {
            self.hypotheses.push(h);
            self.weights.push(1.0_f64);

            // **DO NOT FORGET** to update the LP model.
            let _ = self.secondary_update(self.hypotheses.last());

            return ControlFlow::Continue(())
        }


        // Compute the smoothed objective value `-f*`.
        let objval = self.objval(&self.weights[..]);


        // If the difference between `gamma` and `objval` is
        // lower than `self.half_tolerance`,
        // optimality guaranteed with the precision.
        if self.gamma - objval <= self.half_tolerance {
            self.terminated = iteration;
            return ControlFlow::Break(self.terminated);
        }


        // Now, we move to the update of `weights`.
        // We first check whether `h` is obtained in past iterations
        // or not.
        let mut opt_h = None;
        let pos = self.hypotheses.iter()
            .position(|f| *f == h)
            .unwrap_or(self.hypotheses.len());


        // If `h` is a newly-attained hypothesis,
        // append it to `hypotheses`.
        if pos == self.hypotheses.len() {
            self.hypotheses.push(h);
            self.weights.push(0.0);
            opt_h = self.hypotheses.last();
        }


        let weights = mem::take(&mut self.weights);

        let prim = self.primary.next_iterate(
            iteration, self.sample, &dist[..],
            &self.hypotheses[..], pos, weights,
        );

        // Secondary update
        let seco = self.secondary_update(opt_h);

        // Choose the better one
        self.better_weight(prim, seco);

        // DEBUG
        checker::check_capped_simplex_condition(&self.weights[..], 1.0);

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



impl<H> Research for MLPBoost<'_, H>
    where H: Classifier + Clone,
{
    type Output = CombinedHypothesis<H>;
    fn current_hypothesis(&self) -> Self::Output {
        CombinedHypothesis::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}


