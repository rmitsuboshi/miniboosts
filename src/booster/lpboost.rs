//! This file defines `LPBoost` based on the paper
//! "Boosting algorithms for Maximizing the Soft Margin"
//! by Warmuth et al.
//! 
use crate::data_type::{Data, Label, Sample};
use crate::booster::core::Booster;
use crate::base_learner::core::Classifier;
use crate::base_learner::core::BaseLearner;
use grb::prelude::*;



/// Struct `LPBoost` has 3 main parameters.
/// 
/// - `dist` is the distribution over training examples,
/// - `weights` is the weights over `classifiers`
///   that the LPBoost obtained up to iteration `t`.
/// - `classifiers` is the classifier that the LPBoost obtained.
/// The length of `weights` and `classifiers` must be same.
pub struct LPBoost<D, L> {
    pub(crate) dist:        Vec<f64>,
    pub(crate) weights:     Vec<f64>,
    pub(crate) classifiers: Vec<Box<dyn Classifier<D, L>>>,

    // These are the parameters used in the `update_param(..)`
    pub(crate) gamma_hat: f64,

    // Tolerance parameter
    eps: f64,

    // Variables for the Gurobi optimizer
    model:   Model,
    vars:    Vec<Var>,
    gamma:   Var,
    constrs: Vec<Constr>
}


impl<D, L> LPBoost<D, L> {
    /// Initialize the `LPBoost<D, L>`.
    pub fn init(sample: &Sample<D, L>) -> LPBoost<D, L> {
        let m = sample.len();
        assert!(m != 0);

        // Set GRBEnv
        let mut env = Env::new("").unwrap();
        env.set(param::OutputFlag, 0).unwrap();


        // Set GRBModel
        let mut model = Model::with_env("", env).unwrap();


        // Set GRBVars
        let vars = (0..m).map(|i| {
                let name = format!("w{}", i);
                add_ctsvar!(model, name: &name, bounds: 0.0..).unwrap()
            }).collect::<Vec<_>>();

        let gamma = add_ctsvar!(model, name: &"gamma", bounds: ..)
            .unwrap();


        // Set a constraint
        let constr = model.add_constr(
            &"sum_is_1", c!(vars.iter().grb_sum() == 1.0)
        ).unwrap();

        let constrs = vec![constr];


        // Set objective function
        model.set_objective(gamma, Minimize).unwrap();


        // Update the model
        model.update().unwrap();


        let uni = 1.0 / m as f64;
        LPBoost {
            dist:        vec![uni; m],
            weights:     Vec::new(),
            classifiers: Vec::new(),
            gamma_hat:   1.0,
            eps:         uni,
            model,
            vars,
            gamma,
            constrs
        }
    }


    /// This method updates the capping parameter.
    /// Once the capping parameter changed,
    /// we need to update the `model` of the Gurobi.
    pub fn capping(mut self, capping_param: f64) -> Self {
        assert!(
            1.0 <= capping_param
            &&
            capping_param <= self.vars.len() as f64
        );

        let ub = 1.0 / capping_param;
        let m = self.vars.len();

        // Initialize GRBModel
        let mut env = Env::new("").unwrap();
        env.set(param::OutputFlag, 0).unwrap();
        let mut model = Model::with_env("", env).unwrap();

        // Initialize GRBVars
        self.gamma = add_ctsvar!(model, name: &"gamma", bounds: ..)
            .unwrap();
        self.vars = (0..m).into_iter()
            .map(|i| {
                let name = format!("w{}", i);
                add_ctsvar!(model, name: &name, bounds: 0.0..ub)
                    .unwrap()
            }).collect::<Vec<Var>>();
        self.model = model;


        // Set GRBConstraint
        let constr = self.model.add_constr(
            &"sum_is_1", c!(self.vars.iter().grb_sum() == 1.0)
        ).unwrap();
        self.constrs[0] = constr;


        // Set objective
        self.model.set_objective(self.gamma, Minimize).unwrap();
        self.model.update().unwrap();


        self
    }


    #[inline(always)]
    fn precision(&mut self, eps: f64) {
        self.eps = eps;
    }

}


impl<D> Booster<D, f64> for LPBoost<D, f64> {

    /// `update_params` updates `self.distribution` and determine the weight on hypothesis
    /// that the algorithm obtained at current iteration.
    fn update_params(&mut self,
                     h: Box<dyn Classifier<D, f64>>,
                     sample: &Sample<D, f64>)
        -> Option<()>
    {


        // update `self.gamma_hat`
        let edge = self.dist.iter()
            .zip(sample.iter())
            .fold(0.0_f64, |mut acc, (d, example)| {
                acc += d * example.label * h.predict(&example.data);
                acc
            });
        if self.gamma_hat > edge {
            self.gamma_hat = edge;
        }



        // Add a new constraint
        let expr = sample.iter()
            .zip(self.vars.iter())
            .map(|(ex, v)| *v * ex.label * h.predict(&ex.data))
            .grb_sum();
        let constr = self.model
            .add_constr(&"", c!(expr <= self.gamma))
            .unwrap();
        self.model.update().unwrap();



        // Solve a linear program to update the distribution over the examples.
        self.model.optimize().unwrap();


        // Check the status. If not `Status::Optimal`, terminate immediately.
        // This will never happen
        // since the domain is a bounded & closed convex set,
        let status = self.model.status().unwrap();
        if status != Status::Optimal {
            panic!("Status is not optimal. something wrong.");
        }


        // At this point,
        // the status of the optimization problem is `Status::Optimal`
        // Therefore, we append a new hypothesis to `self.classifiers`
        self.classifiers.push(h);
        self.constrs.push(constr);


        // Check the stopping criterion.
        let gamma_star = self.model
            .get_obj_attr(attr::X, &self.gamma)
            .unwrap();
        if gamma_star >= self.gamma_hat - self.eps {
            self.weights = self.constrs[1..].iter()
                .map(|constr| {
                    self.model.get_obj_attr(attr::Pi, constr)
                        .unwrap()
                        .abs()
                }).collect::<Vec<f64>>();

            return None;
        }


        // Update the distribution over the training examples.
        self.dist = self.vars.iter()
            .map(|var| self.model.get_obj_attr(attr::X, var).unwrap())
            .collect::<Vec<f64>>();

        Some(())
    }


    fn run(&mut self,
           base_learner: &dyn BaseLearner<D, f64>,
           sample: &Sample<D, f64>,
           eps: f64)
    {
        if self.eps != eps {
            self.precision(eps);
        }


        // Since the LPBoost does not have non-trivial iteration,
        // we run this until the stopping criterion is satisfied.
        loop {
            let h = base_learner.best_hypothesis(sample, &self.dist);
            if let None = self.update_params(h, sample) {
                println!("Break loop at: {}", self.classifiers.len());
                break;
            }
        }
    }


    fn predict(&self, data: &Data<D>) -> Label<f64> {
        assert_eq!(self.weights.len(), self.classifiers.len());

        let mut confidence = 0.0;
        for (w, h) in self.weights.iter().zip(self.classifiers.iter()) {
            confidence += w * h.predict(data);
        }


        confidence.signum()
    }
}
