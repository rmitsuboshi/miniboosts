//! This file defines `ERLPBoost` based on the paper
//! "Entropy Regularized LPBoost"
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

    soft_margin_optimization,

    common::utils,
    common::checker,
    research::Research,
};


use std::cell::RefCell;
use std::ops::ControlFlow;



/// The `ERLPBoost` algorithm proposed in the following paper: 
/// 
/// [Manfred K. Warmuth, Karen A. Glocer, and S. V. N. Vishwanathan - Entropy Regularized LPBoost](https://link.springer.com/chapter/10.1007/978-3-540-87987-9_23)
/// 
/// Given a set `{(x_{1}, y_{1}), (x_{2}, y_{2}), ..., (x_{m}, y_{m})}`
/// of training examples,
/// a capping parameters `ν ∈ [1, m]`, and
/// an accuracy parameter `ε > 0`,
/// `ERLPBoost` aims to find an `ε`-approximate solution of
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
/// # Convergence rate
/// - `ERLPBoost` terminates in `O( ln(m/ν) / ε² )` iterations.
///
/// # Related information
/// - Every round, `ERLPBoost` solves a convex program
///   by the sequential quadratic minimization technique.
///   So, running time per round is slow 
///   compared to [`LPBoost`](crate::booster::LPBoost).
/// 
/// # Example
/// The following code shows a small example 
/// for running [`ERLPBoost`].  
/// 
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
/// 
/// // Get the number of training examples.
/// let n_sample = sample.shape().0 as f64;
/// 
/// // Set the upper-bound parameter of outliers in `sample`.
/// // Here we assume that the outliers are at most 10% of `sample`.
/// let nu = 0.1 * n_sample;
/// 
/// // Initialize `ERLPBoost` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis whose training error is
/// // less than `0.01` if the traing examples are linearly separable.
/// let mut booster = ERLPBoost::init(&sample)
///     .tolerance(0.01)
///     .nu(0.1 * n_sample);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .criterion(Criterion::Entropy)
///     .build();
/// 
/// // Run `ERLPBoost` and obtain the resulting hypothesis `f`.
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
pub struct ERLPBoost<'a, F> {
    // Training sample
    sample: &'a Sample,

    // Distribution over examples
    dist: Vec<f64>,

    // `gamma_hat` corresponds to $\min_{q=1, .., t} P^q (d^{q-1})$
    gamma_hat: f64,

    // `gamma_star` corresponds to $P^{t-1} (d^{t-1})$
    gamma_star: f64,
    // regularization parameter defined in the paper
    eta: f64,

    half_tolerance: f64,

    qp_model: Option<RefCell<QPModel>>,

    hypotheses: Vec<F>,
    weights: Vec<f64>,


    // an accuracy parameter for the sub-problems
    n_sample: usize,
    nu: f64,


    terminated: usize,

    max_iter: usize,
}


impl<'a, F> ERLPBoost<'a, F> {
    /// Constructs a new instance of `ERLPBoost`.
    /// 
    /// Time complexity: `O(1)`.
    pub fn init(sample: &'a Sample) -> Self {
        let n_sample = sample.shape().0;
        assert!(n_sample != 0);

        // Compute $\ln(n_sample)$ in advance
        let ln_n_sample = (n_sample as f64).ln();


        // Set tolerance
        let half_tolerance = 0.005;


        // Set regularization parameter
        let eta = 0.5_f64.max(ln_n_sample / half_tolerance);

        // Set gamma_hat and gamma_star
        let gamma_hat  = 1.0;
        let gamma_star = f64::MIN;


        ERLPBoost {
            sample,

            dist: Vec::new(),
            gamma_hat,
            gamma_star,
            eta,
            half_tolerance,
            qp_model: None,

            hypotheses: Vec::new(),
            weights: Vec::new(),


            n_sample,
            nu: 1.0,

            terminated: usize::MAX,
            max_iter: usize::MAX,
        }
    }


    fn init_solver(&mut self) {
        checker::check_nu(self.nu, self.n_sample);


        let upper_bound = 1.0 / self.nu;
        let qp_model = RefCell::new(QPModel::init(
            self.eta, self.n_sample, upper_bound
        ));

        self.qp_model = Some(qp_model);
    }


    /// Updates the capping parameter.
    /// 
    /// Time complexity: `O(1)`.
    pub fn nu(mut self, nu: f64) -> Self {
        assert!(1.0 <= nu && nu <= self.n_sample as f64);
        self.nu = nu;
        self.regularization_param();

        self
    }


    /// Returns the break iteration.
    /// This method returns `0` before the `.run()` call.
    /// 
    /// Time complexity: `O(1)`.
    #[inline(always)]
    pub fn terminated(&self) -> usize {
        self.terminated
    }


    /// Set the tolerance parameter.
    /// 
    /// Time complexity: `O(1)`.
    #[inline(always)]
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.half_tolerance = tolerance / 2.0;
        self.regularization_param();
        self
    }


    /// Setter method of `self.eta`
    /// 
    /// Time complexity: `O(1)`.
    #[inline(always)]
    fn regularization_param(&mut self) {
        let ln_n_sample = (self.n_sample as f64 / self.nu).ln();


        self.eta = 0.5_f64.max(ln_n_sample / self.half_tolerance);
    }


    /// `max_loop` returns the maximum iteration
    /// of the Adaboost to find a combined hypothesis
    /// that has error at most `tolerance`.
    /// 
    /// Time complexity: `O(1)`.
    fn max_loop(&mut self) -> usize {
        let n_sample = self.n_sample as f64;

        let mut max_iter = 4.0 / self.half_tolerance;


        let ln_n_sample = (n_sample / self.nu).ln();
        let temp = 8.0 * ln_n_sample / self.half_tolerance.powi(2);


        max_iter = max_iter.max(temp);

        max_iter.ceil() as usize
    }
}


impl<F> ERLPBoost<'_, F>
    where F: Classifier
{
    /// Update `self.gamma_hat`.
    /// `self.gamma_hat` holds the minimum value of the objective value.
    /// 
    /// Time complexity: `O(m)`, where `m` is the number of training examples.
    #[inline]
    fn update_gamma_hat_mut(&mut self, h: &F)
    {
        let edge = utils::edge_of_hypothesis(self.sample, &self.dist[..], h);
        let entropy = utils::entropy_from_uni_distribution(&self.dist[..]);

        let obj_val = edge + (entropy / self.eta);

        self.gamma_hat = self.gamma_hat.min(obj_val);
    }


    /// Update `self.gamma_star`.
    /// `self.gamma_star` holds the current optimal value.
    /// 
    /// Time complexity: `O(t)`, where `t` is the number of hypotheses
    /// attained by the current iteration.
    fn update_gamma_star_mut(&mut self)
    {
        let max_edge = self.hypotheses.iter()
            .map(|h|
                utils::edge_of_hypothesis(self.sample, &self.dist, h)
            )
            .reduce(f64::max)
            .expect("Failed to compute the max-edge");
        let entropy = utils::entropy_from_uni_distribution(&self.dist);
        self.gamma_star = max_edge + (entropy / self.eta);
    }


    /// Updates `self.dist`
    /// This method repeatedly minimizes the quadratic approximation of 
    /// ERLPB. objective around current distribution `self.dist`.
    /// Then update `self.dist` as the optimal solution of 
    /// the approximate problem. 
    /// This method continues minimizing the quadratic objective 
    /// while the decrease of the optimal value is 
    /// greater than `self.sub_tolerance`.
    fn update_distribution_mut(&mut self, clf: &F)
    {
        self.qp_model.as_ref()
            .expect("Failed to call `.as_ref()` to `self.qp_model`")
            .borrow_mut()
            .update(self.sample, clf);

        self.dist = self.qp_model.as_ref()
            .expect("Failed to call `.as_ref()` to `self.qp_model`")
            .borrow()
            .distribution();
    }
}


impl<F> Booster<F> for ERLPBoost<'_, F>
    where F: Classifier + Clone,
{
    type Output = WeightedMajority<F>;


    fn name(&self) -> &str {
        "ERLPBoost"
    }


    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_sample, n_feature) = self.sample.shape();
        let ratio = self.nu * 100f64 / n_sample as f64;
        let nu = utils::format_unit(self.nu);
        let info = Vec::from([
            ("# of examples", format!("{n_sample}")),
            ("# of features", format!("{n_feature}")),
            ("Tolerance", format!("{}", 2f64 * self.half_tolerance)),
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

        self.max_iter = self.max_loop();
        self.terminated = self.max_iter;

        self.hypotheses = Vec::new();

        self.gamma_hat = 1.0;
        self.gamma_star = -1.0;


        assert!((0.0..1.0).contains(&self.half_tolerance));
        self.regularization_param();
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
        let h = weak_learner.produce(self.sample, &self.dist[..]);


        // update `self.gamma_hat`
        self.update_gamma_hat_mut(&h);


        // Check the stopping criterion
        let diff = self.gamma_hat - self.gamma_star;
        if diff <= self.half_tolerance {
            self.terminated = iteration;
            let (_, weights) = soft_margin_optimization(self.nu, &self.sample, &self.hypotheses[..]);
            self.weights = weights;
            return ControlFlow::Break(iteration);
        }

        // At this point, the stopping criterion is not satisfied.

        // Update the parameters
        self.update_distribution_mut(&h);


        // Append a new hypothesis to `clfs`.
        self.hypotheses.push(h);


        // update `self.gamma_star`.
        self.update_gamma_star_mut();

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

impl<H> Research for ERLPBoost<'_, H>
    where H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        let weights = self.qp_model.as_ref()
            .expect("Failed to call `.as_ref()` to `self.qp_model`")
            .borrow_mut()
            .weight()
            .collect::<Vec<_>>();

        WeightedMajority::from_slices(&weights[..], &self.hypotheses[..])
    }
}


