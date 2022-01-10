//! This file defines `ERLPBoost` based on the paper
//! "Entropy Regularized LPBoost"
//! by Warmuth et al.
//! 
use crate::data_type::{Data, Label, Sample};
use crate::booster::core::Booster;
use crate::base_learner::core::Classifier;
use crate::base_learner::core::BaseLearner;
use grb::prelude::*;



/// Struct `ERLPBoost` has 3 main parameters.
///     - `dist` is the distribution over training examples,
///     - `weights` is the weights over `classifiers`
///       that the ERLPBoost obtained up to iteration `t`.
///     - `classifiers` is the classifier that the ERLPBoost obtained.
/// The length of `weights` and `classifiers` must be same.
pub struct ERLPBoost<D, L> {
    pub(crate) dist:        Vec<f64>,
    pub(crate) weights:     Vec<f64>,
    pub(crate) classifiers: Vec<Box<dyn Classifier<D, L>>>,

    // `gamma_hat` corresponds to $\min_{q=1, .., t} P^q (d^{q-1})$
    pub(crate) gamma_hat:   f64,

    // `gamma_star` corresponds to $P^{t-1} (d^{t-1})
    gamma_star:      f64,
    // `eta` is the regularization parameter defined in the paper
    eta:             f64,

    eps:             f64,
    sub_eps:         f64, // an accuracy parameter for the sub-problems
    capping_param:   f64,
    grb_env:         Env,
}


impl<D, L> ERLPBoost<D, L> {
    /// Initialize the `ERLPBoost<D, L>`.
    pub fn init(sample: &Sample<D, L>) -> ERLPBoost<D, L> {
        let m = sample.len();
        assert!(m != 0);

        let mut env = Env::new("").unwrap();

        env.set(param::OutputFlag, 0).unwrap();

        // Set uni as an uniform weight
        let uni = 1.0 / m as f64;

        // Compute $\ln(m)$ in advance
        let ln_m = (m as f64).ln();


        // Set eps, sub_eps
        let eps     = uni /  2.0;
        let sub_eps = eps / 10.0;


        // Set regularization parameter
        let mut eta = 1.0 / 2.0;
        let temp    = 2.0 * ln_m / eps;

        if eta < temp {
            eta = temp;
        }

        // Set gamma_hat and gamma_star
        let gamma_hat  = 1.0 + (ln_m / eta);
        let gamma_star = f64::MIN;


        ERLPBoost {
            dist:        vec![uni; m],
            weights:     Vec::new(),
            classifiers: Vec::new(),
            gamma_hat,
            gamma_star,
            eps,
            sub_eps,
            eta,
            capping_param: 1.0,
            grb_env:       env
        }
    }


    /// This method updates the capping parameter.
    pub fn capping(mut self, capping_param: f64) -> Self {
        assert!(
            1.0 <= capping_param
            &&
            capping_param <= self.dist.len() as f64
        );
        self.capping_param = capping_param;
        self.regularization_param();

        self
    }


    /// Setter method of `self.eps`
    fn precision(&mut self, eps: f64) {
        self.eps = eps / 2.0;
        self.sub_eps = eps / 10.0;
        self.regularization_param();
    }


    /// Setter method of `self.eta`
    fn regularization_param(&mut self) {
        let ln_m = (self.dist.len() as f64 / self.capping_param).ln();
        self.eta = 1.0 / 2.0;
        let temp = 2.0 * ln_m / self.eps;

        if self.eta < temp {
            self.eta = temp;
        }

        self.gamma_hat = 1.0 + (ln_m / self.eta);
    }



    /// `max_loop` returns the maximum iteration
    /// of the Adaboost to find a combined hypothesis
    /// that has error at most `eps`.
    pub fn max_loop(&mut self, eps: f64) -> u64 {
        if self.eps != eps {
            self.precision(eps);
        }

        let m = self.dist.len();

        let mut max_iter = 8.0 / self.eps;

        let temp = (m as f64 / self.capping_param).ln();
        let temp = 32.0 * temp / (self.eps * self.eps);

        if max_iter < temp {
            max_iter = temp;
        }

        max_iter.ceil() as u64
    }
}


impl<D> ERLPBoost<D, f64> {
    fn set_weights(&mut self, sample: &Sample<D, f64>)
        -> Result<(), grb::Error>
    {
        let mut model = Model::with_env("", &self.grb_env)?;

        let m = sample.len();
        let t = self.classifiers.len();

        // Initialize GRBVars
        let ws = (0..t).map(|i| {
                let name = format!("w{}", i);
                add_ctsvar!(model, name: &name, bounds: 0.0..1.0).unwrap()
            }).collect::<Vec<_>>();
        let xi = (0..m).map(|i| {
                let name = format!("w{}", i);
                add_ctsvar!(model, name: &name, bounds: 0.0..).unwrap()
            }).collect::<Vec<_>>();
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
        let temp = 1.0 / self.capping_param;
        let objective = rho - temp * xi.iter().grb_sum();
        model.set_objective(objective, Maximize)?;
        model.update()?;


        model.optimize()?;


        let status = model.status()?;

        if status != Status::Optimal {
            println!("Status: {:?}", status);
            panic!("Failed to finding an optimal solution");
        }


        // Assign weights over the hypotheses
        self.weights = ws.into_iter()
            .map(|w| model.get_obj_attr(attr::X, &w).unwrap())
            .collect::<Vec<_>>();

        Ok(())
    }
}


impl<D> Booster<D, f64> for ERLPBoost<D, f64> {

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


        // Check the stopping criterion
        let delta = self.gamma_hat - self.gamma_star;
        if delta <= self.eps / 2.0 {
            return None;
        }

        // At this point, the stopping criterion is not satisfied.
        // Append a new hypothesis to `self.classifiers`.
        self.classifiers.push(h);
        loop {
            // Initialize GRBModel
            let mut model = Model::with_env("", &self.grb_env).unwrap();
            let gamma = add_ctsvar!(
                model, name: &"gamma", bounds: ..
            ).unwrap();


            // Set variables that are used in the optimization problem
            let ub = 1.0 / self.capping_param;

            let vars = self.dist.iter()
                .enumerate()
                .map(|(i, &d)| {
                    let name = format!("v{}", i);
                    add_ctsvar!(model, name: &name, bounds: -d..ub-d)
                        .unwrap()
                }).collect::<Vec<Var>>();

            model.update().unwrap();

            // Set constraints
            for h in self.classifiers.iter() {
                let expr = sample.iter()
                    .zip(self.dist.iter())
                    .zip(vars.iter())
                    .map(|((ex, &d), &v)| {
                        ex.label * h.predict(&ex.data) * (d + v)
                    })
                    .grb_sum();

                model.add_constr(&"", c!(expr <= gamma)).unwrap();
            }
            model.add_constr(&"sum_is_1", c!(vars.iter().grb_sum() == 0.0))
                .unwrap();
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

            let objective = gamma + objective * (1.0 / self.eta);
            model.set_objective(objective, Minimize).unwrap();

            model.update().unwrap();


            model.optimize().unwrap();


            // Check the status.
            // If not `Status::Optimal`, terminate immediately.
            // This will never happen
            // since the domain is a bounded & closed convex set,
            let status = model.status().unwrap();
            if status != Status::Optimal {
                println!("Status is {:?}. something wrong.", status);
                return None;
            }


            // At this point, there exists an optimal solution in `vars`
            // Check the stopping criterion 
            let mut l2 = 0.0;
            for (v, d) in vars.iter().zip(self.dist.iter_mut()) {
                let val = model.get_obj_attr(attr::X, &v).unwrap();
                *d += val;
                l2 += val * val;
            }
            let l2 = l2.sqrt();

            if l2 < self.sub_eps {
                self.gamma_star = model.get_attr(attr::ObjVal).unwrap();
                break;
            }
        }


        Some(())
    }


    fn run(&mut self,
           base_learner: &dyn BaseLearner<D, f64>,
           sample: &Sample<D, f64>,
           eps: f64)
    {
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
