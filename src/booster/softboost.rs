//! This file defines `SoftBoost` based on the paper
//! "Boosting Algorithms for Maximizing the Soft Margin"
//! by Warmuth et al.
//! 
use crate::{Data, Sample};
use crate::{Classifier, CombinedClassifier};
use crate::BaseLearner;
use crate::Booster;
use grb::prelude::*;



/// Struct `SoftBoost` has 3 main parameters.
/// 
/// - `dist` is the distribution over training examples,
pub struct SoftBoost {
    pub(crate) dist: Vec<f64>,

    // `gamma_hat` corresponds to $\min_{q=1, .., t} P^q (d^{q-1})
    gamma_hat:       f64,
    tolerance:       f64,
    // an accuracy parameter for the sub-problems
    sub_tolerance:   f64,
    capping_param:   f64,

    env: Env,
}


impl SoftBoost {
    /// Initialize the `SoftBoost`.
    pub fn init<T: Data>(sample: &Sample<T>) -> SoftBoost {
        let m = sample.len();
        assert!(m != 0);

        let mut env = Env::new("").unwrap();

        env.set(param::OutputFlag, 0).unwrap();

        // Set uni as an uniform weight
        let uni = 1.0 / m as f64;


        // Set tolerance, sub_tolerance
        let tolerance = uni;
        let sub_tolerance = uni / 10.0;


        // Set gamma_hat
        let gamma_hat = 1.0;


        SoftBoost {
            dist:          vec![uni; m],
            gamma_hat,
            tolerance,
            sub_tolerance,
            capping_param: 1.0,
            env: env
        }
    }


    /// This method updates the capping parameter.
    #[inline(always)]
    pub fn capping(mut self, capping_param: f64) -> Self {
        assert!(
            1.0 <= capping_param
            &&
            capping_param <= self.dist.len() as f64
        );
        self.capping_param = capping_param;

        self
    }


    /// Set the tolerance parameter.
    #[inline(always)]
    fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
        self.sub_tolerance = tolerance / 10.0;
    }


    /// `max_loop` returns the maximum iteration
    /// of the Adaboost to find a combined hypothesis
    /// that has error at most `tolerance`.
    pub fn max_loop(&mut self, tolerance: f64) -> u64 {
        if self.tolerance != tolerance {
            self.set_tolerance(tolerance);
        }

        let m = self.dist.len() as f64;

        let temp = (m / self.capping_param).ln();
        let max_iter = 2.0 * temp / (self.tolerance * self.tolerance);

        max_iter.ceil() as u64
    }


    /// Returns a optimal value of the optimization problem LPBoost solves
    pub fn opt_val(&self) -> f64 {
        self.gamma_hat
    }
}


impl SoftBoost {
    /// Set the weight on the classifiers.
    /// This function is called at the end of the boosting.
    fn set_weights<C, D>(&mut self, sample: &Sample<D>, clfs: &[C])
        -> Result<Vec<f64>, grb::Error>
        where C: Classifier<D>,
              D: Data,
    {
        let mut model = Model::with_env("", &self.env)?;

        let m = self.dist.len();
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
        for ((dat, lab), &x) in sample.iter().zip(xi_vec.iter()) {
            let expr = wt_vec.iter()
                .zip(clfs.iter())
                .map(|(&w, h)| *lab * h.predict(dat) * w)
                .grb_sum();

            model.add_constr(&"", c!(expr >= rho - x))?;
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
            panic!(
                "Failed to finding an optimal solution. Status: {:?}",
                status
            );
        }


        // Assign weights over the hypotheses
        let weights = wt_vec.into_iter()
            .map(|w| model.get_obj_attr(attr::X, &w).unwrap())
            .collect::<Vec<_>>();

        Ok(weights)
    }


    /// Updates `self.distribution`
    /// Returns `None` if the stopping criterion satisfied.
    fn update_params_mut<C, D>(&mut self,
                               sample: &Sample<D>,
                               clfs:   &[C])
        -> Option<()>
        where C: Classifier<D>,
              D: Data
    {
        loop {
            // Initialize GRBModel
            let mut model = Model::with_env("", &self.env).unwrap();


            // Set variables that are used in the optimization problem
            let cap = 1.0 / self.capping_param;

            let vars = self.dist.iter()
                .map(|&d| {
                    let lb = - d;
                    let ub = cap - d;
                    add_ctsvar!(model, name: &"", bounds: lb..ub)
                        .unwrap()
                })
                .collect::<Vec<Var>>();
            model.update().unwrap();


            // Set constraints
            for h in clfs.iter() {
                let expr = sample.iter()
                    .zip(self.dist.iter())
                    .zip(vars.iter())
                    .map(|(((dat, lab), &d), &v)| {
                        *lab * h.predict(dat) * (d + v)
                    }).grb_sum();

                model.add_constr(
                    &"", c!(expr <= self.gamma_hat - self.tolerance)
                ).unwrap();
            }
            model.add_constr(
                &"zero_sum", c!(vars.iter().grb_sum() == 0.0)
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

            model.set_objective(objective, Minimize).unwrap();
            model.update().unwrap();


            // Optimize
            model.optimize().unwrap();


            // Check the status
            let status = model.status().unwrap();
            // If the status is `Status::Infeasible`,
            // it implies that the `tolerance`-optimalitys
            // of the previous solution
            if status == Status::Infeasible
                || status == Status::InfOrUnbd {
                return None;
            }


            // At this point, the status is not `Status::Infeasible`.
            // If the status is not `Status::Optimal`, something wrong.
            if status != Status::Optimal
                && status != Status::SubOptimal
            {
                panic!("Status is {:?}. something wrong.", status);
            }



            // Check the stopping criterion
            let mut l2 = 0.0;
            for (v, d) in vars.iter().zip(self.dist.iter_mut()) {
                let val = model.get_obj_attr(attr::X, &v).unwrap();
                *d += val;
                l2 += val * val;
            }
            let l2 = l2.sqrt();

            if l2 < self.sub_tolerance {
                break;
            }
        }


        if self.dist.iter().any(|&d| d == 0.0) {
            return None;
        }


        Some(())
    }
}


impl<D, C> Booster<D, C> for SoftBoost
    where C: Classifier<D>,
          D: Data<Output = f64>,
{


    fn run<B>(&mut self,
              base_learner: &B,
              sample:       &Sample<D>,
              tolerance:    f64)
        -> CombinedClassifier<D, C>
        where B: BaseLearner<D, Clf = C>,
    {
        let max_iter = self.max_loop(tolerance);

        let mut clfs = Vec::new();
        for t in 1..=max_iter {
            // Receive a hypothesis from the base learner
            let h = base_learner.best_hypothesis(sample, &self.dist);

            // update `self.gamma_hat`
            let edge = self.dist.iter()
                .zip(sample.iter())
                .fold(0.0_f64, |mut acc, (&d, (dat, lab))| {
                    acc += d * *lab * h.predict(dat);
                    acc
                });


            if self.gamma_hat > edge {
                self.gamma_hat = edge;
            }


            // At this point, the stopping criterion is not satisfied.
            // Append a new hypothesis to `self.classifiers`.
            clfs.push(h);

            // Update the parameters
            if let None = self.update_params_mut(sample, &clfs) {
                println!("Break loop at: {t}");
                break;
            }
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
                    .collect::<Vec<(f64, C)>>()
            }
        };

        CombinedClassifier::from(weighted_classifier)
    }
}


