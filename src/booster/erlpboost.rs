//! This file defines `ERLPBoost` based on the paper
//! "Entropy Regularized LPBoost"
//! by Warmuth et al.
//! 
use crate::Sample;
use crate::{Classifier, CombinedClassifier};
use crate::BaseLearner;
use crate::Booster;
use grb::prelude::*;



/// Struct `ERLPBoost` has 3 main parameters.
/// - `dist` is the distribution over training examples,
pub struct ERLPBoost {
    pub(crate) dist: Vec<f64>,

    // `gamma_hat` corresponds to $\min_{q=1, .., t} P^q (d^{q-1})$
    gamma_hat: f64,

    // `gamma_star` corresponds to $P^{t-1} (d^{t-1})$
    gamma_star: f64,
    // regularization parameter defined in the paper
    eta:           f64,

    tolerance:     f64,
    // an accuracy parameter for the sub-problems
    sub_tolerance: f64,
    capping_param: f64,
    env:           Env,
}


impl ERLPBoost {
    /// Initialize the `ERLPBoost`.
    pub fn init(sample: &Sample) -> ERLPBoost {
        let m = sample.len();
        assert!(m != 0);

        let mut env = Env::new("").unwrap();

        env.set(param::OutputFlag, 0).unwrap();

        // Set uni as an uniform weight
        let uni = 1.0 / m as f64;

        // Compute $\ln(m)$ in advance
        let ln_m = (m as f64).ln();


        // Set tolerance, sub_tolerance
        let tolerance     = uni / 2.0;
        let sub_tolerance = tolerance / 10.0;


        // Set regularization parameter
        let mut eta = 1.0 / 2.0;
        let temp    = 2.0 * ln_m / tolerance;

        if eta < temp {
            eta = temp;
        }

        // Set gamma_hat and gamma_star
        let gamma_hat  = 1.0 + (ln_m / eta);
        let gamma_star = f64::MIN;


        ERLPBoost {
            dist:          vec![uni; m],
            gamma_hat,
            gamma_star,
            eta,
            tolerance,
            sub_tolerance,
            capping_param: 1.0,
            env:           env
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


    /// Setter method of `self.tolerance`
    #[inline(always)]
    fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance / 2.0;
        self.sub_tolerance = tolerance / 10.0;
        self.regularization_param();
    }


    /// Setter method of `self.eta`
    #[inline(always)]
    fn regularization_param(&mut self) {
        let ln_m = (self.dist.len() as f64 / self.capping_param).ln();
        self.eta = 1.0 / 2.0;
        let temp = 2.0 * ln_m / self.tolerance;

        if self.eta < temp {
            self.eta = temp;
        }

        self.gamma_hat = 1.0 + (ln_m / self.eta);
    }


    /// Returns a optimal value of the optimization problem LPBoost solves
    pub fn opt_val(&self) -> f64 {
        self.gamma_hat
    }



    /// `max_loop` returns the maximum iteration
    /// of the Adaboost to find a combined hypothesis
    /// that has error at most `tolerance`.
    pub fn max_loop(&mut self, tolerance: f64) -> u64 {
        if self.tolerance * 2.0 != tolerance {
            self.set_tolerance(tolerance);
        }

        let m = self.dist.len();

        let mut max_iter = 8.0 / self.tolerance;

        let temp = (m as f64 / self.capping_param).ln();
        let temp = 32.0 * temp / (self.tolerance * self.tolerance);

        if max_iter < temp {
            max_iter = temp;
        }

        max_iter.ceil() as u64
    }
}


impl ERLPBoost {
    /// Compute the weight on hypotheses
    fn set_weights<C>(&mut self, sample: &Sample, clfs: &[C])
        -> Result<Vec<f64>, grb::Error>
        where C: Classifier
    {
        let mut model = Model::with_env("", &self.env)?;

        let m = sample.len();
        let t = clfs.len();

        // Initialize GRBVars
        let wt_vec = (0..t).map(|i| {
                let name = format!("w{}", i);
                add_ctsvar!(model, name: &name, bounds: 0.0..1.0).unwrap()
            }).collect::<Vec<_>>();
        let xi_vec = (0..m).map(|i| {
                let name = format!("xi{}", i);
                add_ctsvar!(model, name: &name, bounds: 0.0..).unwrap()
            }).collect::<Vec<_>>();
        let rho = add_ctsvar!(model, name: &"rho", bounds: ..)?;


        // Set constraints
        for (ex, &xi) in sample.iter().zip(xi_vec.iter()) {
            let expr = wt_vec.iter()
                .zip(clfs.iter())
                .map(|(&w, h)| ex.label * h.predict(&ex.data) * w)
                .grb_sum();

            model.add_constr(&"", c!(expr >= rho - xi))?;
        }

        model.add_constr(
            &"sum_is_1", c!(wt_vec.iter().grb_sum() == 1.0)
        )?;
        model.update()?;


        // Set the objective function
        let temp = 1.0 / self.capping_param;
        let objective = rho - temp * xi_vec.iter().grb_sum();
        model.set_objective(objective, Maximize)?;
        model.update()?;


        model.optimize()?;


        let status = model.status()?;

        if status != Status::Optimal {
            let message = format!(
                "Failed to finding an optimal solution. Status: {:?}",
                status
            );
            panic!("{message}");
        }


        // Assign weights over the hypotheses
        let weights = wt_vec.into_iter()
            .map(|w| model.get_obj_attr(attr::X, &w).unwrap())
            .collect::<Vec<_>>();

        Ok(weights)
    }


    /// Updates `self.distribution`
    fn update_params_mut<C>(&mut self, clfs: &[C], sample:  &Sample)
        where C: Classifier
    {


        loop {
            // Initialize GRBModel
            let mut model = Model::with_env("", &self.env).unwrap();
            let gamma = add_ctsvar!(
                model, name: &"gamma", bounds: ..
            ).unwrap();


            // Set variables that are used in the optimization problem
            let upper_bound = 1.0 / self.capping_param;

            let vars = self.dist.iter()
                .enumerate()
                .map(|(i, &d)| {
                    let name = format!("v{}", i);
                    // define the i'th lb & ub
                    let lb = -d;
                    let ub = upper_bound - d;
                    add_ctsvar!(model, name: &name, bounds: lb..ub).unwrap()
                }).collect::<Vec<Var>>();

            model.update().unwrap();

            // Set constraints
            for h in clfs.iter() {
                let expr = sample.iter()
                    .zip(self.dist.iter())
                    .zip(vars.iter())
                    .map(|((ex, &d), &v)| {
                        ex.label * h.predict(&ex.data) * (d + v)
                    })
                    .grb_sum();

                model.add_constr(&"", c!(expr <= gamma)).unwrap();
            }
            model.add_constr(
                &"zero_sum",
                c!(vars.iter().grb_sum() == 0.0_f64)
            ).unwrap();
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
                let message = format!(
                    "Status is {:?}. something wrong.", status
                );
                panic!("{message}");
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

            if l2 < self.sub_tolerance {
                self.gamma_star = model.get_attr(attr::ObjVal).unwrap();
                break;
            }
        }
    }
}


impl<C> Booster<C> for ERLPBoost
    where C: Classifier + Eq + PartialEq
{
    fn run<B>(&mut self, base_learner: &B, sample: &Sample, tolerance: f64)
        -> CombinedClassifier<C>
        where B: BaseLearner<Clf = C>
    {
        let max_iter = self.max_loop(tolerance);

        let mut clfs = Vec::new();

        for t in 1..=max_iter {
            // Receive a hypothesis from the base learner
            let h = base_learner.best_hypothesis(sample, &self.dist);


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
            if delta <= self.tolerance / 2.0 {
                println!("Break loop at: {t}");
                break;
            }

            // At this point, the stopping criterion is not satisfied.
            // Append a new hypothesis to `clfs`.
            clfs.push(h);

            // Update the parameters
            self.update_params_mut(&clfs, sample);
        }

        // Set the weights on the hypotheses
        // by solving a linear program
        let weighted_classifier = match self.set_weights(&sample, &clfs) {
            Err(e) => {
                panic!("{e}");
            },
            Ok(weights) => {
                weights.into_iter()
                    .zip(clfs.into_iter())
                    .filter(|(w, _)| *w != 0.0)
                    .collect::<Vec<_>>()
            }
        };

        CombinedClassifier { weighted_classifier }
    }
}


