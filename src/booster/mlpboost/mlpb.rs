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

use crate::{Classifier, CombinedClassifier};
use crate::BaseLearner;
use crate::Booster;


use std::cell::RefCell;



/// MLPBoost struct. See [this paper](https://arxiv.org/abs/2209.10831).
pub struct MLPBoost {
    // Tolerance parameter
    tolerance: f64,


    // Number of examples
    size: usize,


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


    terminated: usize,
}


impl MLPBoost {
    /// Initialize the `MLPBoost`.
    pub fn init(data: &DataFrame, _target: &Series) -> Self {
        let (size, _) = data.shape();
        assert!(size != 0);


        let uni = 1.0 / size as f64;
        let eta = 2.0 * (size as f64).ln() / uni;
        let nu  = 1.0;

        MLPBoost {
            tolerance: uni,
            size,
            nu,
            eta,
            lp_model: None,


            primary: Primary::ShortStep,
            secondary: Secondary::LPB,
            condition: StopCondition::ObjVal,

            terminated: 0_usize,
        }
    }


    /// This method updates the capping parameter.
    /// This parameter must be in `[1, sample_size]`.
    pub fn nu(mut self, nu: f64) -> Self {
        assert!(1.0 <= nu && nu <= self.size as f64);
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
        let ln_m = (self.size as f64 / self.nu).ln();
        self.eta = ln_m / self.tolerance;
    }


    /// Initialize the LP solver.
    fn init_solver(&mut self) {
        let upper_bound = 1.0 / self.nu;

        assert!((0.0..=1.0).contains(&upper_bound));

        let lp_model = RefCell::new(LPModel::init(self.size, upper_bound));

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
        let ln_m = (self.size as f64 / self.nu).ln();
        (8.0_f64 * ln_m / self.tolerance.powi(2)).ceil() as usize
    }


    /// Returns the terminated iteration.
    /// This method returns `0` before the boosting step.
    pub fn terminated(&self) -> usize {
        self.terminated
    }


    fn secondary_update<C>(&self,
                           data: &DataFrame,
                           target: &Series,
                           opt_h: Option<&C>)
        -> Vec<f64>
        where C: Classifier,
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
    fn objval<C>(&self,
                 data: &DataFrame,
                 target: &Series,
                 classifiers: &[C],
                 weights: &[f64])
        -> f64
        where C: Classifier
    {
        let dist = dist_at(
            self.eta, self.nu, data, target, classifiers, weights
        );


        let margin = edge_of(data, target, &dist[..], classifiers, weights);


        let entropy = dist.iter()
            .copied()
            .map(|d| if d == 0.0 { 0.0 } else { d * d.ln() })
            .sum::<f64>();

        margin + (entropy + (self.size as f64).ln()) / self.eta
    }


    /// Choose the better weights by some criterion.
    fn better_weight<C>(&self,
                        data: &DataFrame,
                        target: &Series,
                        dist: &[f64],
                        prim: Vec<f64>,
                        seco: Vec<f64>,
                        classifiers: &[C])
        -> Vec<f64>
        where C: Classifier
    {
        let prim_val;
        let seco_val;

        match self.condition {
            StopCondition::Edge => {
                prim_val = edge_of(data, target, dist, classifiers, &prim[..]);
                seco_val = edge_of(data, target, dist, classifiers, &seco[..]);
            },

            StopCondition::ObjVal => {
                prim_val = self.objval(data, target, classifiers, &prim[..]);
                seco_val = self.objval(data, target, classifiers, &seco[..]);
            },
        }
        if prim_val >= seco_val {
            prim
        } else {
            seco
        }
    }
}


impl<C> Booster<C> for MLPBoost
    where C: Classifier + PartialEq,
{
    fn run<B>(&mut self,
              base_learner: &B,
              data: &DataFrame,
              target: &Series,
    ) -> CombinedClassifier<C>
        where B: BaseLearner<Clf = C>,
    {
        // ------------------------------------------------------
        // Pre-processing

        // Initialize the parameters.
        self.init_params();


        // Compute the maximum iteration and set `self.terminated`.
        let max_iter = self.max_loop();
        self.terminated = max_iter;


        // Obtain a hypothesis by passing the uniform distribution.
        let h = base_learner.produce(
            data, target, &vec![1.0 / self.size as f64; self.size]
        );


        // Defines the vector of hypotheses obtained from `base_learner`.
        let mut classifiers = vec![h];
        // Defines the vector of weights on `classifiers`.
        let mut weights = vec![1.0];


        // Upper-bound of the optimal `edge`.
        let mut gamma: f64 = 1.0;

        // ------------------------------------------------------

        for step in 1..max_iter {


            // Compute the distribution over training instances.
            let dist = dist_at(self.eta,
                               self.nu,
                               data,
                               target,
                               &classifiers[..],
                               &weights[..]);


            // Obtain a hypothesis w.r.t. `dist`.
            let h = base_learner.produce(data, target, &dist);


            // Compute the edge of newly-attained hypothesis `h`.
            let edge_h = edge_of_h(data, target, &dist[..], &h);


            // Update the estimation of `edge`.
            gamma = gamma.min(edge_h);


            // Compute the objective value.
            let objval = self.objval(
                data, target, &classifiers[..], &weights[..]
            );


            if step % 10 == 0 {
                println!("Step {step}: diff: {}", gamma - objval);
            }


            // If the difference between `gamma` and `objval` is
            // lower than `self.tolerance`,
            // optimality guaranteed with the precision.
            if gamma - objval <= self.tolerance {
                println!("Break loop at: {step}");
                self.terminated = step;
                break;
            }


            // Now, we move to the update of `weights`.
            // We first check whether `h` is obtained in past iterations
            // or not.
            let mut opt_h = None;
            let pos = classifiers.iter()
                .position(|f| *f == h)
                .unwrap_or(classifiers.len());


            // If `h` is a newly-attained hypothesis,
            // append it to `classifiers`.
            if pos == classifiers.len() {
                classifiers.push(h);
                weights.push(0.0);
                opt_h = classifiers.last();
            }


            // Primary update
            let prim = self.primary.update(self.eta,
                                           self.nu,
                                           data,
                                           target,
                                           &dist[..],
                                           pos,
                                           &classifiers[..],
                                           weights,
                                           step);

            // Secondary update
            let seco = self.secondary_update(data, target, opt_h);


            // Choose the better one
            weights = self.better_weight(
                data, target, &dist[..], prim, seco, &classifiers[..]
            );
        }


        // Construct the combined classifier.
        let clfs = weights.into_iter()
            .zip(classifiers)
            .filter(|(w, _)| *w > 0.0)
            .collect::<Vec<_>>();


        CombinedClassifier::from(clfs)
    }
}


