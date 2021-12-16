/// This file defines `LPBoost` based on the paper
/// "Boosting algorithms for Maximizing the Soft Margin"
/// by Warmuth et al.
/// 
use crate::data_type::{Data, Label, Sample};
use crate::booster::core::Booster;
use crate::base_learner::core::Classifier;
use crate::base_learner::core::BaseLearner;
use grb::prelude::*;



/// Struct `LPBoost` has 3 parameters.
/// `dist` is the distribution over training examples,
/// `weights` is the weights over `classifiers` that the LPBoost obtained up to iteration `t`.
/// `classifiers` is the classifier that the LPBoost obtained.
/// The length of `weights` and `classifiers` must be same.
pub struct LPBoost<D, L> {
    pub dist: Vec<f64>,
    pub weights: Vec<f64>,
    pub classifiers: Vec<Box<dyn Classifier<D, L>>>,
    pub gamma_hat: f64,

    eps: f64,
    grb_model: Model,
    grb_vars: Vec<Var>,
    grb_gamma: Var,
    grb_constrs: Vec<Constr>
}


impl<D, L> LPBoost<D, L> {
    pub fn init(sample: &Sample<D, L>) -> LPBoost<D, L> {
        let m = sample.len();
        assert!(m != 0);

        let mut env = Env::new("").unwrap();

        env.set(param::OutputFlag, 0).unwrap();
        let mut grb_model = Model::with_env("", env).unwrap();

        let mut grb_vars = Vec::with_capacity(m);
        for i in 0..m {
            let name = format!("grb_vars[{}]", i);
            let var = add_ctsvar!(grb_model, name: &name, bounds: 0.0..).unwrap();
            grb_vars.push(var);
        }
        let grb_gamma = add_ctsvar!(grb_model, name: &"gamma", bounds: ..).unwrap();

        let constr = grb_model.add_constr(
            &"sum_is_1", c!(grb_vars.iter().grb_sum() == 1.0)
        ).unwrap();

        let grb_constrs = vec![constr];


        grb_model.set_objective(grb_gamma, Minimize).unwrap();

        grb_model.update().unwrap();


        let uni = 1.0 / m as f64;
        LPBoost {
            dist: vec![uni; m], weights: Vec::new(), classifiers: Vec::new(), gamma_hat: 1.0, eps: 1.0,
            grb_model, grb_vars, grb_gamma, grb_constrs
        }
    }


    /// This method updates the capping parameter.
    pub fn capping(mut self, capping_param: f64) -> Self {
        assert!(1.0 <= capping_param && capping_param <= self.grb_vars.len() as f64);
        let ub = 1.0 / capping_param;
        let m = self.grb_vars.len();

        let mut env = Env::new("").unwrap();

        env.set(param::OutputFlag, 0).unwrap();
        let mut grb_model = Model::with_env("", env).unwrap();

        self.grb_vars = (0..m).into_iter()
            .map(|i| add_ctsvar!(grb_model, name: &format!("grb_vars[{}]", i), bounds: 0.0..ub).unwrap())
            .collect::<Vec<Var>>();
        self.grb_model = grb_model;

        let constr = self.grb_model.add_constr(
            &"sum_is_1", c!(self.grb_vars.iter().grb_sum() == 1.0)
        ).unwrap();

        self.grb_constrs[0] = constr;

        self.grb_model.set_objective(self.grb_gamma, Minimize).unwrap();
        self.grb_model.update().unwrap();

        self
    }

    pub fn precision(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }


    fn make_expr(&self, sample: &Sample<D, f64>, h: &Box<dyn Classifier<D, f64>>) -> Expr {
        sample.iter()
            .zip(self.grb_vars.iter())
            .map(|(example, var)| *var * example.label * h.predict(&example.data))
            .grb_sum()
    }
}


impl<D> Booster<D, f64> for LPBoost<D, f64> {

    /// `update_params` updates `self.distribution` and determine the weight on hypothesis
    /// that the algorithm obtained at current iteration.
    fn update_params(&mut self, h: Box<dyn Classifier<D, f64>>, sample: &Sample<D, f64>) -> Option<()> {


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
        let expr = self.make_expr(sample, &h);
        let constr = self.grb_model.add_constr(&"", c!(expr <= self.grb_gamma)).unwrap();
        self.grb_model.update().unwrap();



        // Solve a linear program to update the distribution over the examples.
        self.grb_model.optimize().unwrap();


        // Check the status. If not `Status::Optimal`, then terminate immediately.
        // This will never happen since the domain is a bounded & closed convex set,
        let status = self.grb_model.status().unwrap();
        if status != Status::Optimal {
            println!("Status is not optimal. something wrong.");
            return None;
        }

        // At this point, the status of the optimization problem is `Status::Optimal`
        // therefore, we append a new hypothesis to `self.classifiers`
        self.classifiers.push(h);
        self.grb_constrs.push(constr);


        // Check the stopping criterion.
        let gamma_star = self.grb_model.get_obj_attr(attr::X, &self.grb_gamma).unwrap();
        if self.gamma_hat >= gamma_star - self.eps {
            self.weights = self.grb_constrs[1..].iter()
                .map(|constr| self.grb_model.get_obj_attr(attr::Pi, constr).unwrap().abs())
                .collect::<Vec<f64>>();

            return None;
        }


        // Update the distribution over the training examples.
        self.dist = self.grb_vars.iter()
            .map(|var| self.grb_model.get_obj_attr(attr::X, var).unwrap())
            .collect::<Vec<f64>>();

        Some(())
    }


    fn run(&mut self, base_learner: Box<dyn BaseLearner<D, f64>>, sample: &Sample<D, f64>, eps: f64) {
        if self.eps != eps {
            self.eps = eps;
        }


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
