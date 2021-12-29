/// This file defines `SoftBoost` based on the paper
/// "Boosting Algorithms for Maximizing the Soft Margin"
/// by Warmuth et al.
/// 
use crate::data_type::{Data, Label, Sample};
use crate::booster::core::Booster;
use crate::base_learner::core::Classifier;
use crate::base_learner::core::BaseLearner;
use grb::prelude::*;



/// Struct `SoftBoost` has 3 main parameters.
///     - `dist` is the distribution over training examples,
///     - `weights` is the weights over `classifiers` that the SoftBoost obtained up to iteration `t`.
///     - `classifiers` is the classifiers that the SoftBoost obtained.
/// The length of `weights` and `classifiers` must be same.
pub struct SoftBoost<D, L> {
    pub dist: Vec<f64>,
    pub weights: Vec<f64>,
    pub classifiers: Vec<Box<dyn Classifier<D, L>>>,


    gamma_hat: f64,  // `gamma_hat` corresponds to $\min_{q=1, .., t} P^q (d^{q-1})$
    eps: f64,
    sub_eps: f64, // an accuracy parameter for the sub-problems
    capping_param: f64,
    grb_env: Env,
}


impl<D, L> SoftBoost<D, L> {
    pub fn init(sample: &Sample<D, L>) -> SoftBoost<D, L> {
        let m = sample.len();
        assert!(m != 0);

        let mut env = Env::new("").unwrap();

        env.set(param::OutputFlag, 0).unwrap();

        // Set uni as an uniform weight
        let uni = 1.0 / m as f64;


        // Set eps, sub_eps
        let eps = uni;
        let sub_eps = uni / 10.0;


        // Set gamma_hat
        let gamma_hat = 1.0;


        SoftBoost {
            dist: vec![uni; m], weights: Vec::new(), classifiers: Vec::new(),
            gamma_hat, eps, sub_eps, capping_param: 1.0, grb_env: env
        }
    }


    /// This method updates the capping parameter.
    pub fn capping(mut self, capping_param: f64) -> Self {
        assert!(1.0 <= capping_param && capping_param <= self.dist.len() as f64);
        self.capping_param = capping_param;

        self
    }


    fn precision(&mut self, eps: f64) {
        self.eps = eps;
        self.sub_eps = eps / 10.0;
    }


    /// `max_loop` returns the maximum iteration of the Adaboost to find a combined hypothesis
    /// that has error at most `eps`.
    pub fn max_loop(&mut self, eps: f64) -> u64 {
        if self.eps != eps {
            self.precision(eps);
        }

        let m = self.dist.len() as f64;

        let max_iter = 2.0 * (m / self.capping_param).ln() / (self.eps * self.eps);

        max_iter.ceil() as u64
    }
}


impl<D> SoftBoost<D, f64> {
    fn set_weights(&mut self, sample: &Sample<D, f64>) -> Result<(), grb::Error> {
        let mut model = Model::with_env("", &self.grb_env)?;

        let m = self.dist.len();
        let t = self.classifiers.len();

        // Initialize GRBVars
        let ws = vec![
            add_ctsvar!(model, name: &"", bounds: 0.0..1.0)?; t
        ];
        let xi = vec![
            add_ctsvar!(model, name: &"", bounds: 0.0..)?; m
        ];
        let rho = add_ctsvar!(model, name: &"rho", bounds: ..)?;


        // Set constraints
        for (ex, &x) in sample.iter().zip(xi.iter()) {
            let expr = ws.iter()
                .zip(self.classifiers.iter())
                .map(|(&w, h)| ex.label * h.predict(&ex.data) * w)
                .grb_sum();

            model.add_constr(&"", c!(expr >= rho - x))?;
        }

        model.add_constr(
            &"sum_is_1", c!(ws.iter().grb_sum() == 1.0)
        )?;
        model.update()?;


        // Set the objective function
        let objective = rho - (1.0 / self.capping_param) * xi.iter().grb_sum();
        model.set_objective(objective, Maximize)?;
        model.update()?;


        model.optimize()?;


        let status = model.status()?;

        if status != Status::Optimal {
            panic!("Failed to finding an optimal solution");
        }


        // Assign weights over the hypotheses
        self.weights = vec![0.0; t];
        for (w, grb_w) in self.weights.iter_mut().zip(ws.into_iter()) {
            *w = model.get_obj_attr(attr::X, &grb_w)?;
        }

        Ok(())
    }
}


impl<D> Booster<D, f64> for SoftBoost<D, f64> {

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


        // At this point, the stopping criterion is not satisfied.
        // Append a new hypothesis to `self.classifiers`.
        self.classifiers.push(h);
        loop {
            // Initialize GRBModel
            let mut model = Model::with_env("", &self.grb_env).unwrap();


            // Set variables that are used in the optimization problem
            let ub = 1.0 / self.capping_param;

            let vars = self.dist.iter()
                .map(|&d| add_ctsvar!(
                    model, name: &"", bounds: -d..ub-d
                ).unwrap())
                .collect::<Vec<Var>>();

            model.update().unwrap();

            // Set constraints
            for h in self.classifiers.iter() {
                let expr = sample.iter()
                    .zip(self.dist.iter())
                    .zip(vars.iter())
                    .map(|((ex, d), v)| ex.label * h.predict(&ex.data) * (*d + *v))
                    .grb_sum();

                model.add_constr(&"", c!(expr <= self.gamma_hat - self.eps)).unwrap();
            }
            model.add_constr(&"sum_is_1", c!(vars.iter().grb_sum() == 0.0)).unwrap();
            model.update().unwrap();


            // Set objective function
            let m = sample.len() as f64;
            let objective = self.dist.iter()
                .zip(vars.iter())
                .map(|(&d, &v)| {
                    let temp = (m * d).ln() + 1.0;
                    temp * v + (v * v) * (1.0 / (2.0 * d))
                })
                .grb_sum();

            model.set_objective(objective, Minimize).unwrap();
            model.update().unwrap();


            // Optimize
            model.optimize().unwrap();


            // Check the status
            let status = model.status().unwrap();
            // If the status is `Status::Infeasible`,
            // it implies that the `eps`-optimality of the previous solution
            if status == Status::Infeasible {
                return None;
            }


            // At this point, the status is not `Status::Infeasible`.
            // Therefore, if the status is not `Status::Optimal`, then something wrong.
            if status != Status::Optimal {
                println!("Status is {:?}. something wrong.", status);
                return None;
            }



            // Check the stopping criterion
            let mut l2 = 0.0;
            for (v, d) in vars.iter().zip(self.dist.iter_mut()) {
                let val = model.get_obj_attr(attr::X, &v).unwrap();
                *d += val;
                l2 += val * val;
            }
            let l2 = l2.sqrt();

            if l2 < self.sub_eps {
                break;
            }
        }


        if self.dist.iter().any(|&d| d == 0.0) {
            return None;
        }


        Some(())
    }


    // fn run(&mut self, base_learner: Box<dyn BaseLearner<D, f64>>, sample: &Sample<D, f64>, eps: f64) {
    fn run(&mut self, base_learner: &dyn BaseLearner<D, f64>, sample: &Sample<D, f64>, eps: f64) {
        let max_iter = self.max_loop(eps);

        for t in 1..=max_iter {
            // Receive a hypothesis from the base learner
            let h = base_learner.best_hypothesis(sample, &self.dist);

            // Update the parameters
            if let None = self.update_params(h, sample) {
                println!("Break loop at: {}", t);
                break;
            }
        }

        // Set the weights on the hypotheses
        // by solving a linear program
        self.set_weights(&sample).unwrap();
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
