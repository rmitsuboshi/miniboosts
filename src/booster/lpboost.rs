//! This file defines `LPBoost` based on the paper
//! "Boosting algorithms for Maximizing the Soft Margin"
//! by Warmuth et al.
//! 
use polars::prelude::*;
// use rayon::prelude::*;

use crate::{Classifier, CombinedClassifier};
use crate::BaseLearner;
use crate::Booster;

use grb::prelude::*;



/// Defines `LPBoost`.
pub struct LPBoost {
    dist: Vec<f64>,

    // min-max edge of the new hypothesis
    gamma_hat: f64,

    // Tolerance parameter
    tolerance: f64,

    // Variables for the Gurobi optimizer
    env: Env,
    model: Model,
    gamma: Var,
    vars: Vec<Var>,
    constrs: Vec<Constr>
}


impl LPBoost {
    /// Initialize the `LPBoost`.
    pub fn init(df: &DataFrame) -> Self {
        let (m, _) = df.shape();
        assert!(m != 0);

        // Set GRBEnv
        let mut env = Env::new("").unwrap();
        env.set(param::OutputFlag, 0).unwrap();


        // Set GRBModel
        let mut model = Model::with_env("", &env).unwrap();


        // Set GRBVars
        let vars = (0..m).map(|i| {
                let name = format!("w[{i}]");
                add_ctsvar!(model, name: &name, bounds: 0.0..1.0)
                    .unwrap()
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
            dist:      vec![uni; m],
            gamma_hat: 1.0,
            tolerance: uni,
            env,
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
        // let mut env = Env::new("").unwrap();
        // env.set(param::OutputFlag, 0).unwrap();
        let mut model = Model::with_env("", &self.env).unwrap();

        // Initialize GRBVars
        self.gamma = add_ctsvar!(model, name: &"gamma", bounds: ..)
            .unwrap();
        self.vars = (0..m).into_iter()
            .map(|i| {
                let name = format!("d[{i}]");
                add_ctsvar!(model, name: &name, bounds: 0.0..ub)
                    .unwrap()
            }).collect::<Vec<Var>>();
        self.model = model;


        // Set GRBConstraint
        let constr = self.model.add_constr(
            &"sum_is_1", c!(self.vars.iter().grb_sum() == 1.0)
        ).unwrap();
        self.constrs = vec![constr];


        // Set objective
        self.model.set_objective(self.gamma, Minimize).unwrap();
        self.model.update().unwrap();


        self
    }


    #[inline(always)]
    fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
    }


    /// Returns a optimal value of the optimization problem LPBoost solves
    pub fn opt_val(&self) -> f64 {
        self.gamma_hat
    }


    /// `update_params` updates `self.distribution` and `self.gamma_hat`
    /// by solving a linear program
    #[inline(always)]
    fn update_params(&mut self, margins: &[f64])
        -> f64
    {
        let edge = margins.into_iter()
            .copied()
            .zip(self.dist.iter().copied())
            .map(|(yh, d)| yh * d)
            .sum::<f64>();


        // update `self.gamma_hat`
        if self.gamma_hat > edge {
            self.gamma_hat = edge;
        }


        // Add a new constraint
        let expr = margins.into_iter()
            .copied()
            .zip(self.vars.iter())
            .map(|(yh, &v)| v * yh)
            .grb_sum();


        let t = self.constrs.len();
        let name = format!("{t}-th hypothesis");
        let constr = self.model
            .add_constr(&name, c!(expr <= self.gamma))
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
        self.constrs.push(constr);


        // Check the stopping criterion.
        let gamma_star = self.model
            .get_obj_attr(attr::X, &self.gamma)
            .unwrap();

        gamma_star
    }
}


impl<C> Booster<C> for LPBoost
    where C: Classifier,
{
    fn run<B>(&mut self,
              base_learner: &B,
              data: &DataFrame,
              target: &Series,
              tolerance: f64)
        -> CombinedClassifier<C>
        where B: BaseLearner<Clf = C>,
    {
        if self.tolerance != tolerance {
            self.set_tolerance(tolerance);
        }

        let mut classifiers = Vec::new();

        // Since the LPBoost does not have non-trivial iteration,
        // we run this until the stopping criterion is satisfied.
        loop {
            let h = base_learner.produce(data, target, &self.dist);

            // Each element in `margins` is the product of
            // the predicted vector and the correct vector
            let margins = target.i64()
                .expect("The target class is not a dtype of i64")
                .into_iter()
                .enumerate()
                .map(|(i, y)|
                    (y.unwrap() * h.predict(data, i)) as f64
                )
                .collect::<Vec<f64>>();


            let gamma_star = self.update_params(&margins[..]);


            classifiers.push(h);

            if gamma_star >= self.gamma_hat - self.tolerance {
                println!("Break loop at: {t}", t = classifiers.len());
                break;
            }

            // Update the distribution over the training examples.
            self.dist = self.vars.iter()
                .map(|var| self.model.get_obj_attr(attr::X, var).unwrap())
                .collect::<Vec<f64>>();
        }

        println!("gstar: {}", self.gamma_hat);


        let clfs = self.constrs[1..].iter()
            .zip(classifiers.into_iter())
            .filter_map(|(constr, classifiers)| {
                let weight = self.model.get_obj_attr(attr::Pi, constr)
                    .unwrap()
                    .abs();
                if weight != 0.0 {
                    Some((weight, classifiers))
                } else {
                    None
                }
            })
            .collect::<Vec<(f64, C)>>();


        CombinedClassifier::from(clfs)
    }
}


