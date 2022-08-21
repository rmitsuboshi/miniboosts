//! This file defines `SoftBoost` based on the paper
//! "Boosting Algorithms for Maximizing the Soft Margin"
//! by Warmuth et al.
//! 
use polars::prelude::*;
// use rayon::prelude::*;


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
    gamma_hat: f64,
    tolerance: f64,
    // an accuracy parameter for the sub-problems
    sub_tolerance: f64,
    capping_param: f64,

    env: Env,
}


impl SoftBoost {
    /// Initialize the `SoftBoost`.
    pub fn init(df: &DataFrame) -> Self {
        let (m, _) = df.shape();
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
            env
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
    fn set_weights<C>(&mut self,
                      data: &DataFrame,
                      target: &Series,
                      classifiers: &[C])
        -> std::result::Result<Vec<f64>, grb::Error>
        where C: Classifier,
    {
        let mut model = Model::with_env("", &self.env)?;

        let m = self.dist.len();
        let t = classifiers.len();

        // Initialize GRBVars
        let wt_vec = (0..t).map(|i| {
                let name = format!("w[{i}]");
                add_ctsvar!(model, name: &name, bounds: 0_f64..).unwrap()
            }).collect::<Vec<_>>();
        let xi_vec = (0..m).map(|i| {
                let name = format!("xi[{i}]");
                add_ctsvar!(model, name: &name, bounds: 0.0_f64..).unwrap()
            }).collect::<Vec<_>>();
        let rho = add_ctsvar!(model, name: "rho", bounds: ..)?;


        // Set constraints
        let iter = target.i64()
            .expect("The target class is not a dtype i64")
            .into_iter()
            .zip(xi_vec.iter())
            .enumerate();

        for (i, (y, &xi)) in iter {
            let y = y.unwrap() as f64;
            let expr = wt_vec.iter()
                .zip(classifiers)
                .map(|(&w, h)| w * h.predict(data, i) as f64)
                .grb_sum();
            let name = format!("sample[{i}]");
            model.add_constr(&name, c!(y * expr >= rho - xi))?;
        }

        model.add_constr(
            "sum_is_1", c!(wt_vec.iter().grb_sum() == 1.0)
        )?;
        model.update()?;


        // Set the objective function
        let param = 1.0 / self.capping_param;
        let objective = rho - param * xi_vec.iter().grb_sum();
        model.set_objective(objective, Maximize)?;
        model.update()?;


        model.optimize()?;


        let status = model.status()?;

        if status != Status::Optimal {
            panic!("Cannot solve the primal problem. Status: {status:?}");
        }


        // Assign weights over the hypotheses
        let weights = wt_vec.into_iter()
            .map(|w| model.get_obj_attr(attr::X, &w).unwrap())
            .collect::<Vec<_>>();

        Ok(weights)
    }


    /// Updates `self.distribution`
    /// Returns `None` if the stopping criterion satisfied.
    fn update_params_mut<C>(&mut self,
                            data: &DataFrame,
                            target: &Series,
                            classifiers: &[C])
        -> Option<()>
        where C: Classifier,
    {
        loop {
            // Initialize GRBModel
            let mut model = Model::with_env("", &self.env).unwrap();


            // Set variables that are used in the optimization problem
            let cap = 1.0 / self.capping_param;

            let vars = self.dist.iter()
                .copied()
                .enumerate()
                .map(|(i, d)| {
                    let lb = - d;
                    let ub = cap - d;
                    let name = format!("delta[{i}]");
                    add_ctsvar!(model, name: &name, bounds: lb..ub)
                        .unwrap()
                })
                .collect::<Vec<Var>>();
            model.update().unwrap();


            // Set constraints
            classifiers.iter()
                .enumerate()
                .for_each(|(j, h)| {
                    let iter = target.i64()
                        .expect("The target class is not a dtype i64");
                    let expr = vars.iter()
                        .zip(self.dist.iter().copied())
                        .zip(iter)
                        .enumerate()
                        .map(|(i, ((v, d), y))| {
                            let y = y.unwrap() as f64;
                            let p = h.predict(data, i) as f64;
                            y * p * (d + *v)
                        })
                        .grb_sum();

                    let name = format!("h[{j}]");
                    model.add_constr(
                        &name, c!(expr <= self.gamma_hat - self.tolerance)
                    ).unwrap();
                });


            model.add_constr(
                "zero_sum", c!(vars.iter().grb_sum() == 0.0)
            ).unwrap();
            model.update().unwrap();


            // Set objective function
            let m = self.dist.len() as f64;
            let objective = self.dist.iter()
                .zip(vars.iter())
                .map(|(&d, &v)| {
                    let lin_coef = (m * d).ln() + 1.0;
                    lin_coef * v + (v * v) * (1.0 / (2.0 * d))
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
            if status != Status::Optimal {
                panic!("Status is {status:?}. something wrong.");
            }



            // Check the stopping criterion
            let mut l2 = 0.0;
            for (v, d) in vars.iter().zip(self.dist.iter_mut()) {
                let val = model.get_obj_attr(attr::X, v).unwrap();
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


impl<C> Booster<C> for SoftBoost
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
        let max_iter = self.max_loop(tolerance);

        let mut classifiers = Vec::new();
        for t in 1..=max_iter {
            // Receive a hypothesis from the base learner
            let h = base_learner.produce(data, target, &self.dist);

            // update `self.gamma_hat`
            // let edge = self.dist.iter()
            //     .zip(sample.iter())
            //     .fold(0.0_f64, |acc, (&d, (dat, lab))| {
            //         let l: f64 = lab.clone().into();
            //         let p: f64 = h.predict(dat).into();
            //         acc + d * l * p
            //     });
            let edge = target.i64()
                .expect("The target class is not a dtype i64")
                .into_iter()
                .zip(self.dist.iter().copied())
                .enumerate()
                .map(|(i, (y, d))|
                    d * y.unwrap() as f64 * h.predict(data, i) as f64
                )
                .sum::<f64>();


            if self.gamma_hat > edge {
                self.gamma_hat = edge;
            }


            // At this point, the stopping criterion is not satisfied.
            // Append a new hypothesis to `self.classifiers`.
            classifiers.push(h);

            // Update the parameters
            if self.update_params_mut(data, target, &classifiers).is_none() {
                println!("Break loop at: {t}");
                break;
            }
        }

        // Set the weights on the hypotheses
        // by solving a linear program
        let clfs = match self.set_weights(data, target, &classifiers) {
            Err(e) => {
                panic!("{e}");
            },
            Ok(weights) => {
                weights.into_iter()
                    .zip(classifiers.into_iter())
                    .filter(|(w, _)| *w != 0.0)
                    .collect::<Vec<(f64, C)>>()
            }
        };

        CombinedClassifier::from(clfs)
    }
}


