//! This file defines `SoftBoost` based on the paper
//! "Boosting Algorithms for Maximizing the Soft Margin"
//! by Warmuth et al.
//! 
#[cfg(not(feature="gurobi"))]
use super::qp_model::QPModel;

#[cfg(feature="gurobi")]
use super::gurobi_qp_model::QPModel;
use crate::{
    Sample,
    Booster,
    WeakLearner,

    Classifier,
    WeightedMajority,
    common::utils,
    common::checker,
    research::Research,
};

use std::cell::RefCell;
use std::ops::ControlFlow;


/// The SoftBoost algorithm proposed in the following paper:  
///
/// [Gunnar Rätsch, Manfred K. Warmuth, and Laren A. Glocer - Boosting Algorithms for Maximizing the Soft Margin](https://papers.nips.cc/paper/2007/hash/cfbce4c1d7c425baf21d6b6f2babe6be-Abstract.html) 
///
/// Given a set `{(x_{1}, y_{1}), (x_{2}, y_{2}), ..., (x_{m}, y_{m})}`
/// of training examples,
/// a capping parameters `ν ∈ [1, m]`, and
/// an accuracy parameter `ε > 0`,
/// `SoftBoost` aims to find an `ε`-approximate solution of
/// the soft-margin optimization problem:
/// ```txt
///  max  ρ - (1/ν) Σ_{i=1}^{m} ξ_{i}
/// ρ,w,ξ
/// s.t. y_{i} Σ_{h ∈ Δ_{H}} w_{h} h(x_{i}) ≥ ρ - ξ_{i},
///                                         for all i ∈ [m],
///      w ∈ Δ_{H},
///      ξ ≥ 0.
/// ```
/// 
/// 
/// # Convergence rate
/// - `SoftBoost` terminates in `O( ln(m/ν) / ε² )` iterations.
/// 
/// # Related information
/// - Every round, `ERLPBoost` solves a convex program
///   by the sequential quadratic minimization technique.
///   So, running time per round is slow 
///   compared to [`LPBoost`](crate::booster::LPBoost).
/// - This code uses Gurobi optimizer,
///   so you need to do the followings:
///     1. Install Gurobi and put its license to your home directory.
///     2. Enable `extended` flag.
///     ```toml
///     miniboosts = { version = "0.3.3", features = ["extended"] }
///     ```
/// - [`SoftBoost`] is the extension 
///   of [`TotalBoost`](crate::booster::TotalBoost).
/// 
/// # Example
/// The following code shows 
/// a small example for running [`SoftBoost`].  
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
/// // Get the number of training examples.
/// let n_sample = sample.shape().0 as f64;
/// 
/// // Set the upper-bound parameter of outliers in `sample`.
/// // Here we assume that the outliers are at most 10% of `sample`.
/// let nu = 0.1 * n_sample;
/// 
/// // Initialize `SoftBoost` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis 
/// // whose soft margin objective value is differs at most `0.01`
/// // from the optimal one.
/// // Further, at the end of this chain,
/// // SoftBoost calls `SoftBoost::nu` to set the capping parameter 
/// // as `0.1 * n_sample`, which means that, 
/// // at most, `0.1 * n_sample` examples are regarded as outliers.
/// let booster = SoftBoost::init(&sample)
///     .tolerance(0.01)
///     .nu(nu);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .criterion(Criterion::Entropy)
///     .build();
/// 
/// // Run `SoftBoost` and obtain the resulting hypothesis `f`.
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
pub struct SoftBoost<'a, F> {
    sample: &'a Sample,

    pub(crate) dist: Vec<f64>,

    // `gamma_hat` corresponds to $\min_{q=1, .., t} P^q (d^{q-1})
    gamma_hat: f64,
    tolerance: f64,
    // an accuracy parameter for the sub-problems
    sub_tolerance: f64,
    nu: f64,

    qp_model: Option<RefCell<QPModel>>,

    hypotheses: Vec<F>,


    max_iter: usize,
    terminated: usize,


    weights: Vec<f64>,
}


impl<'a, F> SoftBoost<'a, F>
    where F: Classifier
{
    /// Initialize the `SoftBoost`.
    pub fn init(sample: &'a Sample) -> Self {
        let n_sample = sample.shape().0;
        assert!(n_sample != 0);

        // Set uni as an uniform weight
        let uni = 1.0 / n_sample as f64;

        let dist = vec![uni; n_sample];


        // Set tolerance, sub_tolerance
        let tolerance = uni;


        SoftBoost {
            sample,

            dist,
            gamma_hat: 1.0,
            tolerance,
            sub_tolerance: 1e-6,
            nu: 1.0,
            qp_model: None,

            hypotheses: Vec::new(),
            weights: Vec::new(),

            max_iter: usize::MAX,
            terminated: usize::MAX,
        }
    }


    /// Set the capping parameter.
    /// 
    /// Time complexity: `O(1)`.
    #[inline(always)]
    pub fn nu(mut self, nu: f64) -> Self {
        let n_sample = self.sample.shape().0 as f64;
        assert!((1.0..=n_sample).contains(&nu));

        self.nu = nu;
        self
    }


    /// Set the tolerance parameter.
    /// 
    /// Time complexity: `O(1)`.
    #[inline(always)]
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }


    fn init_solver(&mut self) {
        let n_sample = self.sample.shape().0;
        checker::check_nu(self.nu, n_sample);

        let upper_bound = 1.0 / self.nu;
        let qp_model = RefCell::new(QPModel::init(n_sample, upper_bound));

        self.qp_model = Some(qp_model);
    }


    /// `max_loop` returns the maximum iteration
    /// of the Adaboost to find a combined hypothesis
    /// that has error at most `tolerance`.
    /// 
    /// Time complexity: `O(1)`.
    pub fn max_loop(&mut self) -> usize {

        let n_sample = self.sample.shape().0 as f64;

        let temp = (n_sample / self.nu).ln();
        let max_iter = 2.0 * temp / self.tolerance.powi(2);

        max_iter.ceil() as usize
    }
}


impl<F> SoftBoost<'_, F>
    where F: Classifier,
{
    /// Set the weight on the hypotheses.
    /// This function is called at the end of the boosting.
    fn set_weights(&self)
        -> std::result::Result<Vec<f64>, ()>
    {
        // Assign weights over the hypotheses
        let weights = self.qp_model.as_ref()
            .expect("Failed to call `.as_ref()` to `self.qp_model`")
            .borrow_mut()
            .weights(self.sample, &self.hypotheses)
            .collect::<Vec<_>>();

        Ok(weights)
    }


    /// Updates `self.dist`
    /// Returns `None` if the stopping criterion satisfied.
    fn update_params_mut(&mut self) -> Option<()> {
        let h = self.hypotheses.last().unwrap();
        self.qp_model.as_ref()
            .expect("Failed to call `.as_ref()` to `self.qp_model`")
            .borrow_mut()
            .update(self.sample, &mut self.dist[..], self.gamma_hat, h)
    }
}


impl<F> Booster<F> for SoftBoost<'_, F>
    where F: Classifier + Clone,
{
    type Output = WeightedMajority<F>;


    fn name(&self) -> &str {
        "SoftBoost"
    }


    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_sample, n_feature) = self.sample.shape();
        let ratio = self.nu * 100f64 / n_sample as f64;
        let nu = utils::format_unit(self.nu);
        let info = Vec::from([
            ("# of examples", format!("{n_sample}")),
            ("# of features", format!("{n_feature}")),
            ("Tolerance", format!("{}", self.tolerance)),
            ("Tolerance (sub-problem)", format!("{}", self.sub_tolerance)),
            ("Max iteration", format!("{}", self.max_iter)),
            ("Capping (outliers)", format!("{nu} ({ratio: >7.3} %)"))
        ]);
        Some(info)
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

        self.sub_tolerance = self.tolerance / 10.0;

        self.max_iter = self.max_loop();
        self.terminated = self.max_iter;
        self.hypotheses = Vec::new();

        self.gamma_hat = 1.0;
        self.init_solver();
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

        // Receive a hypothesis from the base learner
        let h = weak_learner.produce(self.sample, &self.dist);

        let edge = utils::edge_of_hypothesis(self.sample, &self.dist, &h);


        if self.gamma_hat > edge {
            self.gamma_hat = edge;
        }


        // At this point, the stopping criterion is not satisfied.
        // Append a new hypothesis to `self.hypotheses`.
        self.hypotheses.push(h);

        // Update the parameters
        if self.update_params_mut().is_none() {
            self.terminated = iteration;
            return ControlFlow::Break(self.terminated);
        }

        ControlFlow::Continue(())
    }


    fn postprocess<W>(
        &mut self,
        _weak_learner: &W,
    ) -> Self::Output
        where W: WeakLearner<Hypothesis = F>
    {
        // Set the weights on the hypotheses
        // by solving a linear program
        self.weights = self.set_weights()
            .expect("Failed to solve the LP");
        WeightedMajority::from_slices(&self.weights[..], &self.hypotheses[..])
    }
}



impl<H> Research for SoftBoost<'_, H>
    where H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        let weights = self.set_weights()
            .expect("Failed to solve the LP");
        WeightedMajority::from_slices(&weights[..], &self.hypotheses[..])
    }
}


