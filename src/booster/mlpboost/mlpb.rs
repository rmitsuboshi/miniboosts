//! This file defines `MLPBoost` based on the paper
//! "Boosting algorithms for Maximizing the Soft Margin"
//! by Warmuth et al.
//! 
use polars::prelude::*;
// use rayon::prelude::*;

use super::{
    lp_model::LPModel,
    dist::dist_at,
    options::*,
    utils::*,
};

use crate::{
    Booster,
    BaseLearner,

    State,
    Classifier,
    CombinedClassifier
};


use std::cell::RefCell;



/// MLPBoost struct. See [this paper](https://arxiv.org/abs/2209.10831).
pub struct MLPBoost<C> {
    // Tolerance parameter
    tolerance: f64,


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
    classifiers: Vec<C>,


    terminated: usize,
    max_iter: usize,


    gamma: f64,
}


impl<C> MLPBoost<C> {
    /// Initialize the `MLPBoost`.
    pub fn init(data: &DataFrame, _target: &Series) -> Self {
        let n_sample = data.shape().0;
        assert!(n_sample != 0);


        let uni = 1.0 / n_sample as f64;
        let eta = 2.0 * (n_sample as f64).ln() / uni;
        let nu  = 1.0;

        MLPBoost {
            tolerance: uni,
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
        self.tolerance = tolerance / 2.0;
        self
    }


    /// Set the regularization parameter.
    #[inline(always)]
    fn eta(&mut self) {
        let ln_m = (self.n_sample as f64 / self.nu).ln();
        self.eta = ln_m / self.tolerance;
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
        // Set the tolerance parameter.
        assert!((0.0..0.5).contains(&self.tolerance));
        // Set the regularization parameter.
        // Note that this method must called after
        // `self.tolerance(..)`.
        self.eta();

        // Initialize the solver.
        self.init_solver();
    }


    /// Returns the maximum iterations 
    /// to obtain the solution with accuracy `tolerance`.
    pub fn max_loop(&self) -> usize {
        let ln_m = (self.n_sample as f64 / self.nu).ln();
        (8.0_f64 * ln_m / self.tolerance.powi(2)).ceil() as usize
    }


    /// Returns the terminated iteration.
    /// This method returns `0` before the boosting step.
    pub fn terminated(&self) -> usize {
        self.terminated
    }
}

impl<C> MLPBoost<C>
    where C: Classifier,
{
    fn secondary_update(
        &self,
        data: &DataFrame,
        target: &Series,
        opt_h: Option<&C>
    ) -> Vec<f64>
    {
        match self.secondary {
            Secondary::LPB => {
                self.lp_model.as_ref()
                    .unwrap()
                    .borrow_mut()
                    .update(data, target, opt_h)
            }
        }
    }


    /// Returns the objective value 
    /// `- \tilde{f}^\star (-Aw)` at the current weighting `w = weights`.
    fn objval(
        &self,
        data: &DataFrame,
        target: &Series,
        weights: &[f64],
    ) -> f64
    {
        let dist = dist_at(
            self.eta,
            self.nu,
            data,
            target,
            &self.classifiers[..],
            weights
        );


        let margin = edge_of(
            data, target, &dist[..], &self.classifiers[..], weights
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
        data: &DataFrame,
        target: &Series,
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
                    data, target, dist, &self.classifiers[..], &prim[..]
                );
                seco_val = edge_of(
                    data, target, dist, &self.classifiers[..], &seco[..]
                );
            },

            StopCondition::ObjVal => {
                prim_val = self.objval(
                    data, target, &prim[..]
                );
                seco_val = self.objval(
                    data, target, &seco[..]
                );
            },
        }
        self.weights = if prim_val >= seco_val { prim } else { seco };
    }
}


impl<C> Booster<C> for MLPBoost<C>
    where C: Classifier + Clone + PartialEq,
{
    fn preprocess<B>(
        &mut self,
        _base_learner: &B,
        data: &DataFrame,
        _target: &Series,
    )
        where B: BaseLearner<Clf = C>
    {
        self.n_sample = data.shape().0;

        self.init_params();

        self.max_iter = self.max_loop();
        self.terminated = self.max_iter;


        // // Obtain a hypothesis by passing the uniform distribution.
        // let h = base_learner.produce(
        //     data, target, &vec![1.0 / self.size as f64; self.size]
        // );


        // // Defines the vector of hypotheses obtained from `base_learner`.
        // let mut classifiers = vec![h];
        // // Defines the vector of weights on `classifiers`.
        // let mut weights = vec![1.0];

        self.classifiers = Vec::new();
        self.weights = Vec::new();

        // Upper-bound of the optimal `edge`.
        self.gamma = 1.0;
    }


    fn boost<B>(
        &mut self,
        base_learner: &B,
        data: &DataFrame,
        target: &Series,
        iteration: usize,
    ) -> State
        where B: BaseLearner<Clf = C>,
    {

        if self.max_iter < iteration {
            return State::Terminate;
        }

        // ------------------------------------------------------

        // Compute the distribution over training instances.
        let dist = dist_at(
            self.eta,
            self.nu,
            data,
            target,
            &self.classifiers[..],
            &self.weights[..]
        );


        // Obtain a hypothesis w.r.t. `dist`.
        let h = base_learner.produce(data, target, &dist);


        // Compute the edge of newly-attained hypothesis `h`.
        let edge_h = edge_of_h(data, target, &dist[..], &h);


        // Update the estimation of `edge`.
        self.gamma = self.gamma.min(edge_h);



        // For the first iteration,
        // just append the hypothesis and continue.
        if iteration == 1 {
            self.classifiers.push(h);
            self.weights.push(1.0_f64);

            return State::Continue;
        }


        // Compute the objective value.
        let objval = self.objval(data, target, &self.weights[..]);


        // If the difference between `gamma` and `objval` is
        // lower than `self.tolerance`,
        // optimality guaranteed with the precision.
        if self.gamma - objval <= self.tolerance {
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
            data,
            target,
            &dist[..],
            pos,
            &self.classifiers[..],
            self.weights.clone(),
            iteration
        );

        // Secondary update
        let seco = self.secondary_update(data, target, opt_h);


        // Choose the better one
        self.better_weight(data, target, &dist[..], prim, seco);

        State::Continue
    }


    fn postprocess<B>(
        &mut self,
        _base_learner: &B,
        _data: &DataFrame,
        _target: &Series,
    ) -> CombinedClassifier<C>
        where B: BaseLearner<Clf = C>
    {
        let clfs = self.weights.clone()
            .into_iter()
            .zip(self.classifiers.clone())
            .filter(|(w, _)| *w > 0.0)
            .collect::<Vec<_>>();


        CombinedClassifier::from(clfs)
    }
}


