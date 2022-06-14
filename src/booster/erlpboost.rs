//! This file defines `ERLPBoost` based on the paper
//! "Entropy Regularized LPBoost"
//! by Warmuth et al.
//! 
use crate::{Data, Sample};
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
    pub fn init<D, L>(sample: &Sample<D, L>) -> ERLPBoost {
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
        let eta = (1.0_f64 / 2.0_f64).max(2.0_f64 * ln_m / tolerance);

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
    }


    /// Setter method of `self.eta`
    #[inline(always)]
    fn regularization_param(&mut self) {
        let ln_m = (self.dist.len() as f64 / self.capping_param).ln();
        let temp = 2.0 * ln_m / self.tolerance;


        self.eta = 0.5_f64.max(temp);
    }



    /// Set `gamma_hat` and `gamma_star`.
    #[inline]
    fn set_gamma(&mut self) {
        let ln_m = (self.dist.len() as f64).ln();

        self.gamma_hat  = 1.0 + (ln_m / self.eta);
        self.gamma_star = f64::MIN;
    }


    /// Set all parameters in ERLPBoost.
    #[inline]
    fn init_params(&mut self, tolerance: f64) {
        self.set_tolerance(tolerance);

        self.regularization_param();

        self.set_gamma();
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
            self.init_params(tolerance);
        }

        let m = self.dist.len() as f64;

        let mut max_iter = 8.0 / self.tolerance;


        let temp = (m / self.capping_param).ln();
        let temp = 32.0 * temp / self.tolerance.powi(2);


        max_iter = max_iter.max(temp);

        max_iter.ceil() as u64
    }
}


impl ERLPBoost {
    /// Update `self.gamma_hat`.
    /// `self.gamma_hat` holds the minimum value of the objective value.
    #[inline]
    fn update_gamma_hat_mut<D, L, C>(&mut self,
                                     clf: &C,
                                     sample: &Sample<D, L>)
        where C: Classifier<D, L>,
              D: Data,
              L: Clone + Into<f64>,
    {
        let edge = self.dist.iter()
            .zip(sample.iter())
            .map(|(d, (x, y))| {
                let l: f64 = y.clone().into();
                let p: f64 = clf.predict(x).into();
                *d * l * p
            })
            .sum::<f64>();


        let m = sample.len() as f64;
        let entropy = self.dist.iter()
            .map(|&d| d * (m * d).ln())
            .sum::<f64>();


        let obj_val = edge + (entropy / self.eta);

        self.gamma_hat = self.gamma_hat.min(obj_val);
    }


    /// Update `self.gamma_star`.
    /// `self.gamma_star` holds the current optimal value.
    #[inline]
    fn update_gamma_star_mut<D, L, C>(&mut self,
                                      clfs: &[C],
                                      sample: &Sample<D, L>)
        where C: Classifier<D, L>,
              D: Data,
              L: Clone + Into<f64>,
    {
        let edge_of = |f: &C| {
            self.dist.iter()
                .zip(sample.iter())
                .map(|(d, (x, y))| {
                    let l: f64 = y.clone().into();
                    let p: f64 = f.predict(x).into();
                    *d * l * p
                })
                .sum::<f64>()
        };


        let max_edge = clfs.iter()
            .map(|f| edge_of(f))
            .reduce(f64::max)
            .unwrap();


        let m = sample.len() as f64;
        let entropy = self.dist.iter()
            .map(|&d| d * (m * d).ln())
            .sum::<f64>();


        self.gamma_star = max_edge + (entropy / self.eta);
    }


    /// Compute the weight on hypotheses
    fn set_weights<C, D, L>(&mut self, sample: &Sample<D, L>, clfs: &[C])
        -> Result<Vec<f64>, grb::Error>
        where C: Classifier<D, L>,
              D: Data,
              L: Clone + Into<f64>,
    {
        let mut model = Model::with_env("", &self.env)?;

        let m = sample.len();
        let t = clfs.len();

        // Initialize GRBVars
        let wt_vec = (0..t).map(|i| {
                let name = format!("w[{i}]");
                add_ctsvar!(model, name: &name, bounds: 0.0..1.0).unwrap()
            }).collect::<Vec<_>>();
        let xi_vec = (0..m).map(|i| {
                let name = format!("xi[{i}]");
                add_ctsvar!(model, name: &name, bounds: 0.0..).unwrap()
            }).collect::<Vec<_>>();
        let rho = add_ctsvar!(model, name: &"rho", bounds: ..)?;


        // Set constraints
        for ((dat, lab), &xi) in sample.iter().zip(xi_vec.iter()) {
            let y: f64 = lab.clone().into();

            let expr = wt_vec.iter()
                .zip(clfs.iter())
                .map(|(&w, h)| {
                    let p: f64 = h.predict(dat).into();
                    w * p
                })
                .grb_sum();

            model.add_constr(&"", c!(y * expr >= rho - xi))?;
        }

        model.add_constr(
            &"sum_is_1", c!(wt_vec.iter().grb_sum() == 1.0)
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
    fn update_params_mut<C, D, L>(&mut self,
                                  clfs: &[C],
                                  sample: &Sample<D, L>)
        where C: Classifier<D, L>,
              D: Data,
              L: Clone + Into<f64>,
    {
        loop {
            // Initialize GRBModel
            let mut model = Model::with_env("", &self.env).unwrap();
            let gamma = add_ctsvar!(model, name: &"gamma", bounds: ..)
                .unwrap();


            // Set variables that are used in the optimization problem
            let upper_bound = 1.0 / self.capping_param;

            let vars = self.dist.iter()
                .enumerate()
                .map(|(i, &d)| {
                    let name = format!("v[{i}]");
                    // define the i'th lb & ub
                    let lb = -d;
                    let ub = upper_bound - d;
                    add_ctsvar!(model, name: &name, bounds: lb..ub)
                        .unwrap()
                }).collect::<Vec<Var>>();

            model.update().unwrap();

            // Set constraints
            for h in clfs.iter() {
                let expr = sample.iter()
                    .zip(self.dist.iter())
                    .zip(vars.iter())
                    .map(|(((dat, lab), &d), &v)| {
                        let l: f64 = lab.clone().into();
                        let p: f64 = h.predict(dat).into();
                        l * p * (d + v)
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
                    let linear = temp * v;
                    let quad = (1.0 / (2.0 * d)) * (v * v);

                    linear + quad
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
                panic!("Status ({status:?}) is not optimal.");
            }


            // At this point, there exists an optimal solution in `vars`
            // Check the stopping criterion 
            let mut l2 = 0.0;
            for (v, d) in vars.iter().zip(self.dist.iter_mut()) {
                let val = model.get_obj_attr(attr::X, &v).unwrap();
                *d += val;
                l2 += val.powi(2);
            }
            let l2 = l2.sqrt();

            if l2 < self.sub_tolerance {
                // self.gamma_star = model.get_attr(attr::ObjVal).unwrap();
                self.update_gamma_star_mut(clfs, sample);
                break;
            }
        }
    }
}


impl<D, L, C> Booster<D, L, C> for ERLPBoost
    where C: Classifier<D, L>,
          D: Data<Output = f64>,
          L: Clone + Into<f64>,
{
    fn run<B>(&mut self,
              base_learner: &B,
              sample:       &Sample<D, L>,
              tolerance:    f64)
        -> CombinedClassifier<D, L, C>
        where B: BaseLearner<D, L, Clf = C>,
    {
        // Initialize all parameters
        self.init_params(tolerance);


        // Get max iteration.
        let max_iter = self.max_loop(tolerance);


        // This vector holds the classifiers
        // obtained from the `base_learner`.
        let mut clfs = Vec::new();

        for t in 1..=max_iter {
            // Receive a hypothesis from the base learner
            let h = base_learner.produce(sample, &self.dist);


            // update `self.gamma_hat`
            self.update_gamma_hat_mut(&h, &sample);


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

        CombinedClassifier::from(weighted_classifier)
    }
}


