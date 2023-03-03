//! This file defines `MLPBoost` based on the paper
//! [Boosting as Frank-Wolfe](https://arxiv.org/abs/2209.10831).
//! by Mitsuboshi et al.
//! 

use super::{
    lp_model::LPModel,
    dist::dist_at,
    options::*,
    utils::*,
};

// use crate::research::{
//     Logger,
//     soft_margin_objective,
// };


use crate::{
    Sample,
    Booster,
    WeakLearner,

    State,
    Classifier,
    CombinedHypothesis
};


use std::cell::RefCell;



/// MLPBoost. This code is based on this paler: 
/// [Boosting as Frank-Wolfe](https://arxiv.org/abs/2209.10831)
/// by Ryotaro Mitsuboshi, Kohei Hatano, and Eiji Takimoto.
/// 
/// # Example
/// The following code shows a small example 
/// for running [`MLPBoost`](MLPBoost).  
/// See also:
/// - [`MLPBoost::nu`]
/// - [`MLPBoost::primary`]
/// - [`MLPBoost::stop_condition`]
/// - [`StopCondition`]
/// - [`Primary`]
/// - [`DTree`]
/// - [`DTreeClassifier`]
/// - [`CombinedHypothesis<F>`]
/// - [`DTree::max_depth`]
/// - [`DTree::criterion`]
/// - [`DataFrame`]
/// - [`Series`]
/// - [`DataFrame::shape`]
/// - [`CsvReader`]
/// 
/// [`MLPBoost::nu`]: MLPBoost::nu
/// [`MLPBoost::primary`]: MLPBoost::primary
/// [`MLPBoost::stop_condition`]: MLPBoost::stop_condition
/// [`StopCondition`]: StopCondition
/// [`Primary`]: Primary
/// [`DTree`]: crate::weak_learner::DTree
/// [`DTreeClassifier`]: crate::weak_learner::DTreeClassifier
/// [`CombinedHypothesis<F>`]: crate::hypothesis::CombinedHypothesis
/// [`DTree::max_depth`]: crate::weak_learner::DTree::max_depth
/// [`DTree::criterion`]: crate::weak_learner::DTree::criterion
/// [`DataFrame`]: polars::prelude::DataFrame
/// [`Series`]: polars::prelude::Series
/// [`DataFrame::shape`]: polars::prelude::DataFrame::shape
/// [`CsvReader`]: polars::prelude::CsvReader
/// 
/// 
/// ```no_run
/// use polars::prelude::*;
/// use miniboosts::prelude::*;
/// 
/// // Read the training data from the CSV file.
/// let mut data = CsvReader::from_path(path_to_csv_file)
///     .unwrap()
///     .has_header(true)
///     .finish()
///     .unwrap();
/// 
/// // Split the column corresponding to labels.
/// let target = data.drop_in_place(class_column_name).unwrap();
/// 
/// // Get the number of training examples.
/// let n_sample = data.shape().0 as f64;
/// 
/// // Initialize `MLPBoost` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis 
/// // whose soft margin objective value is differs at most `0.01`
/// // from the optimal one.
/// // Further, at the end of this chain,
/// // MLPBoost calls `MLPBoost::nu` to set the capping parameter 
/// // as `0.1 * n_sample`, which means that, 
/// // at most, `0.1 * n_sample` examples are regarded as outliers.
/// let booster = MLPBoost::init(&data, &target)
///     .tolerance(0.01)
///     .primary(Primary::ShortStep)
///     .stop_condition(StopCondition::ObjVal)
///     .nu(0.1 * n_sample);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTree::init(&data, &target)
///     .max_depth(2)
///     .criterion(Criterion::Edge);
/// 
/// // Run `MLPBoost` and obtain the resulting hypothesis `f`.
/// let f: CombinedHypothesis<DTreeClassifier> = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions: Vec<i64> = f.predict_all(&data);
/// 
/// // Calculate the training loss.
/// let training_loss = target.i64()
///     .unwrap()
///     .into_iter()
///     .zip(predictions)
///     .map(|(true_label, prediction) {
///         let true_label = true_label.unwrap();
///         if true_label == prediction { 0.0 } else { 1.0 }
///     })
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
    primary: Primary,

    // Secondary (heuristic) update
    secondary: Secondary,

    // Stopping condition
    condition: StopCondition,


    // GRBModel.
    lp_model: Option<RefCell<LPModel>>,


    // Weights on hypotheses
    weights: Vec<f64>,

    // Hypotheses
    classifiers: Vec<F>,


    terminated: usize,
    max_iter: usize,


    gamma: f64,
}


impl<'a, F> MLPBoost<'a, F> {
    /// Initialize the `MLPBoost`.
    pub fn init(sample: &'a Sample) -> Self {
        let n_sample = sample.shape().0;
        assert!(n_sample != 0);


        let uni = 0.5 / n_sample as f64;
        let eta = 2.0 * (n_sample as f64).ln() / uni;
        let nu  = 1.0;

        MLPBoost {
            sample,

            half_tolerance: uni,
            n_sample,
            nu,
            eta,
            lp_model: None,


            primary: Primary::ShortStep,
            secondary: Secondary::LPB,
            condition: StopCondition::ObjVal,

            weights: Vec::new(),
            classifiers: Vec::new(),

            terminated: usize::MAX,
            max_iter: usize::MAX,

            gamma: 1.0,
        }
    }


    /// This method updates the capping parameter.
    /// This parameter must be in `[1, sample_size]`.
    pub fn nu(mut self, nu: f64) -> Self {
        assert!(1.0 <= nu && nu <= self.n_sample as f64);
        self.nu = nu;

        self
    }


    /// Update the Primary rule.
    pub fn primary(mut self, rule: Primary) -> Self {
        self.primary = rule;
        self
    }


    /// Update the Secondary rule.
    pub fn secondary(mut self, rule: Secondary) -> Self {
        self.secondary = rule;
        self
    }


    /// Update the stopping condition.
    pub fn stop_condition(mut self, cond: StopCondition) -> Self {
        self.condition = cond;
        self
    }


    /// Set the tolerance parameter.
    #[inline(always)]
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.half_tolerance = tolerance / 2.0;
        self
    }


    /// Set the regularization parameter.
    #[inline(always)]
    fn eta(&mut self) {
        let ln_m = (self.n_sample as f64 / self.nu).ln();
        self.eta = ln_m / self.half_tolerance;
    }


    /// Initialize the LP solver.
    fn init_solver(&mut self) {
        let upper_bound = 1.0 / self.nu;

        assert!((0.0..=1.0).contains(&upper_bound));

        let lp_model = RefCell::new(LPModel::init(self.n_sample, upper_bound));

        self.lp_model = Some(lp_model);
    }


    /// Initialize all parameters.
    /// The methods `self.tolerance(..)`, `self.eta(..)`, and
    /// `self.init_solver(..)` are accessed only via this method.
    fn init_params(&mut self) {
        // Set the regularization parameter.
        self.eta();

        // Initialize the solver.
        self.init_solver();
    }


    /// Returns the maximum iterations 
    /// to obtain the solution with accuracy `self.half_tolerance`.
    pub fn max_loop(&self) -> usize {
        let ln_m = (self.n_sample as f64 / self.nu).ln();
        (8.0_f64 * ln_m / self.half_tolerance.powi(2)).ceil() as usize
    }


    /// Returns the terminated iteration.
    /// This method returns `0` before the boosting step.
    pub fn terminated(&self) -> usize {
        self.terminated
    }
}

impl<F> MLPBoost<'_, F>
    where F: Classifier,
{
    fn secondary_update(&self, opt_h: Option<&F>) -> Vec<f64> {
        match self.secondary {
            Secondary::LPB => {
                self.lp_model.as_ref()
                    .unwrap()
                    .borrow_mut()
                    .update(self.sample, opt_h)
            }
        }
    }


    /// Returns the objective value 
    /// `- \tilde{f}^\star (-Aw)` at the current weighting `w = weights`.
    fn objval(&self, weights: &[f64]) -> f64 {
        let dist = dist_at(
            self.eta,
            self.nu,
            self.sample,
            &self.classifiers[..],
            weights
        );


        let margin = edge_of(
            self.sample, &dist[..], &self.classifiers[..], weights
        );


        let entropy = dist.iter()
            .copied()
            .map(|d| if d == 0.0 { 0.0 } else { d * d.ln() })
            .sum::<f64>();

        margin + (entropy + (self.n_sample as f64).ln()) / self.eta
    }


    /// Choose the better weights by some criterion.
    fn better_weight(
        &mut self,
        dist: &[f64],
        prim: Vec<f64>,
        seco: Vec<f64>,
    )
    {
        let prim_val;
        let seco_val;

        match self.condition {
            StopCondition::Edge => {
                prim_val = edge_of(
                    self.sample, dist, &self.classifiers[..], &prim[..]
                );
                seco_val = edge_of(
                    self.sample, dist, &self.classifiers[..], &seco[..]
                );
            },

            StopCondition::ObjVal => {
                prim_val = self.objval(&prim[..]);
                seco_val = self.objval(&seco[..]);
            },
        }
        self.weights = if prim_val >= seco_val { prim } else { seco };
    }
}


impl<F> Booster<F> for MLPBoost<'_, F>
    where F: Classifier + Clone + PartialEq,
{
    fn preprocess<W>(
        &mut self,
        _weak_learner: &W,
    )
        where W: WeakLearner<Hypothesis = F>
    {
        self.n_sample = self.sample.shape().0;

        self.init_params();

        self.max_iter = self.max_loop();
        self.terminated = self.max_iter;


        self.classifiers = Vec::new();
        self.weights = Vec::new();

        // Upper-bound of the optimal `edge`.
        self.gamma = 1.0;
    }


    fn boost<W>(
        &mut self,
        weak_learner: &W,
        iteration: usize,
    ) -> State
        where W: WeakLearner<Hypothesis = F>,
    {

        if self.max_iter < iteration {
            return State::Terminate;
        }

        // ------------------------------------------------------

        // Compute the distribution over training instances.
        let dist = dist_at(
            self.eta,
            self.nu,
            self.sample,
            &self.classifiers[..],
            &self.weights[..]
        );


        // Obtain a hypothesis w.r.t. `dist`.
        let h = weak_learner.produce(self.sample, &dist);


        // Compute the edge of newly-attained hypothesis `h`.
        let edge_h = edge_of_h(self.sample, &dist[..], &h);


        // Update the estimation of `edge`.
        self.gamma = self.gamma.min(edge_h);



        // For the first iteration,
        // just append the hypothesis and continue.
        if iteration == 1 {
            self.classifiers.push(h);
            self.weights.push(1.0_f64);

            // **DO NOT FORGET** to update the LP model.
            let _ = self.secondary_update(self.classifiers.last());

            return State::Continue;
        }


        // Compute the objective value.
        let objval = self.objval(&self.weights[..]);


        // If the difference between `gamma` and `objval` is
        // lower than `self.half_tolerance`,
        // optimality guaranteed with the precision.
        if self.gamma - objval <= self.half_tolerance {
            self.terminated = iteration;
            return State::Terminate;
        }


        // Now, we move to the update of `weights`.
        // We first check whether `h` is obtained in past iterations
        // or not.
        let mut opt_h = None;
        let pos = self.classifiers.iter()
            .position(|f| *f == h)
            .unwrap_or(self.classifiers.len());


        // If `h` is a newly-attained hypothesis,
        // append it to `classifiers`.
        if pos == self.classifiers.len() {
            self.classifiers.push(h);
            self.weights.push(0.0);
            opt_h = self.classifiers.last();
        }


        // Primary update
        let prim = self.primary.update(
            self.eta,
            self.nu,
            self.sample,
            &dist[..],
            pos,
            &self.classifiers[..],
            self.weights.clone(),
            iteration
        );

        // Secondary update
        let seco = self.secondary_update(opt_h);


        // Choose the better one
        self.better_weight(&dist[..], prim, seco);

        State::Continue
    }


    fn postprocess<W>(
        &mut self,
        _weak_learner: &W,
    ) -> CombinedHypothesis<F>
        where W: WeakLearner<Hypothesis = F>
    {
        let clfs = self.weights.clone()
            .into_iter()
            .zip(self.classifiers.clone())
            .filter(|(w, _)| *w > 0.0)
            .collect::<Vec<_>>();


        CombinedHypothesis::from(clfs)
    }
}


// impl<F> Logger for MLPBoost<'_, F>
//     where F: Classifier
// {
//     /// MLPBoost optimizes the soft margin objective
//     fn objective_value(&self)
//         -> f64
//     {
//         soft_margin_objective(
//             self.data, self.target,
//             &self.weights[..], &self.classifiers[..], self.nu
//         )
//     }
// 
// 
//     fn prediction(&self, data: &DataFrame, i: usize) -> f64 {
//         self.weights.iter()
//             .zip(&self.classifiers[..])
//             .map(|(w, h)| w * h.confidence(data, i))
//             .sum::<f64>()
//     }
// 
// 
//     fn logging<L>(
//         &self,
//         loss_function: &L,
//         test_data: &DataFrame,
//         test_target: &Series,
//     ) -> (f64, f64, f64)
//         where L: Fn(f64, f64) -> f64
//     {
//         let objval = self.objective_value();
//         let train = self.loss(loss_function, self.data, self.target);
//         let test = self.loss(loss_function, test_data, test_target);
// 
//         (objval, train, test)
//     }
// }
