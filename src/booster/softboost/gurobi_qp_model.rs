use grb::prelude::*;


use crate::Sample;
use crate::common::utils;
use crate::hypothesis::Classifier;

const QP_TOLERANCE: f64 = 1e-9;
const NUMERIC_TOLERANCE: f64 = 1e-200;

/// A linear programming model for edge minimization. 
pub(super) struct QPModel {
    pub(self) env: Env,
    pub(self) nu_inv: f64,
    pub(self) margins: Vec<Vec<f64>>,
}


impl QPModel {
    /// Initialize the LP model.
    /// arguments.
    /// - `size`: Number of variables (Number of examples).
    /// - `upper_bound`: Capping parameter. `[1, size]`.
    pub(super) fn init(size: usize, upper_bound: f64)
        -> Self
    {
        let mut env = Env::empty()
            .expect("Failed to construct a new `Env` for SoftBoost");
        env.set(param::OutputFlag, 0)
            .expect("Failed to set `param::OutputFlag` to `0`");
        env.set(param::NumericFocus, 3)
            .expect("Failed to set `NumericFocus` parameter to `3`");
        let env = env.start()
            .expect("Failed to construct a new `Env` for SoftBoost");

        Self {
            env,
            nu_inv: upper_bound,
            margins: Vec::new(),
        }
    }


    /// Solve the edge minimization problem 
    /// over the hypotheses `h1, ..., ht` 
    /// and outputs the optimal value.
    pub(super) fn update<F>(
        &mut self,
        sample: &Sample,
        dist: &mut [f64],
        ghat: f64,
        clf: &F,
    )
        where F: Classifier
    {
        let mut model = Model::with_env("SoftBoost", &self.env)
            .expect(
                "Failed to construct a new model for `SoftBoost` \
                or `TotalBoost`"
            );

        let mut model = Model::with_env("ERLPBoost", env)
            .expect("Failed to construct a new model for `ERLPBoost`");

        let dist = (0..size).map(|i| {
                let name = format!("d[{i}]");
                add_ctsvar!(model, name: &name, bounds: 0_f64..self.nu_inv)
            }).collect::<Result<Vec<_>, _>>()
            .expect("Failed to add new variables `d[..]`");


        // Set a constraint
        model.add_constr("sum_is_1", c!(dist.iter().grb_sum() == 1.0))
            .expect("Failed to set the constraint `sum( d[..] ) = 1.0`");


        // Update the model
        model.update()
            .expect("Faild to update the model after the initialization");


        // Set variables that are used in the optimization problem
        let margins = utils::margins_of_hypothesis(sample, clf);
        self.margins.push(margins);

        for (t, margins) in self.margins.iter() {
            let name = format!("{t}-th hypothesis");
            let edge = sample.target()
                .iter()
                .enumerate()
                .map(|(i, y)| y * clf.confidence(sample, i))
                .zip(self.dist.iter().copied())
                .map(|(yh, d)| d * yh)
                .grb_sum();
            model.add_constr(&name, c!(edge <= ghat));
        }
        model.update()
            .expect("Failed to update the model after adding a new constraint");


        let mut old_objval = 1e9;

        loop {
            // Set objective function
            let objective = dist.iter()
                .copied()
                .zip(self.dist.iter())
                .map(|(d, &grb_d)| {
                    let l_term = d.ln() * grb_d;
                    let q_term = (0.5_f64 / d) * (grb_d * grb_d);

                    l_term + q_term
                })
                .grb_sum();
            self.model.set_objective(objective, Minimize)
                .expect("Failed to set the objective function");


            self.model.optimize()
                .expect("Failed to optimize the problem");


            let status = self.model.status()
                .expect("Failed to get the model status");
            match status {
                Status::InfOrUnbd | Status::Infeasible => { return None; },
                Status::Numeric => { break; }
                Status::Optimal => {}
                _ => {
                    panic!("Status is {status:?}. something wrong.");
                }
            }


            // At this point, there exists an optimal solution in `vars`
            // Check the stopping criterion 
            let objval = self.model.get_attr(attr::ObjVal)
                .expect("Failed to attain the optimal value");


            let mut any_zero = false;
            dist.iter_mut()
                .zip(&self.dist[..])
                .for_each(|(d, grb_d)| {
                    let g = self.model.get_obj_attr(attr::X, grb_d)
                        .expect("Failed to get the optimal solution");
                    *d = g;
                });
            if self.has_zero(dist) { return None; }

            if old_objval - objval < QP_TOLERANCE {
                break;
            }

            old_objval = objval;
        }
        Some(())
    }


    pub(self) fn has_zero(&self, dist: &[f64]) -> bool {
        dist.into_iter()
            .copied()
            .any(|di| di < NUMERIC_TOLERANCE)
    }

    /// Returns the distribution over examples.
    pub(super) fn distribution(&self)
        -> Vec<f64>
    {
        self.dist.iter()
            .map(|d| self.model.get_obj_attr(attr::X, d))
            .collect::<Result<Vec<_>, _>>()
            .expect("Failed to get the optimal solutions `d[..]`")
    }


    /// Returns the weights over the hypotheses.
    pub(super) fn weights<F>(
        &mut self,
        sample: &Sample,
        hypotheses: &[F],
    ) -> impl Iterator<Item=f64> + '_
        where F: Classifier
    {
        let mut model = Model::with_env("Soft margin optimization", &self.env)?;

        let n_sample = sample.shape().0;
        let n_hypotheses = self.hypotheses.len();

        // Initialize GRBVars
        let wt_vec = (0..n_hypotheses).map(|i| {
                let name = format!("w[{i}]");
                add_ctsvar!(model, name: &name, bounds: 0_f64..)
            }).collect::<Result<Vec<_>, _>>()?;
        let xi_vec = (0..n_sample).map(|i| {
                let name = format!("xi[{i}]");
                add_ctsvar!(model, name: &name, bounds: 0_f64..)
            }).collect::<Result<Vec<_>, _>>()?;
        let rho = add_ctsvar!(model, name: "rho", bounds: ..)?;


        // Set constraints
        let target = sample.target();
        let iter = target.iter()
            .zip(xi_vec.iter())
            .enumerate();

        for (i, (&y, &xi)) in iter {
            let expr = wt_vec.iter()
                .zip(&hypotheses[..])
                .map(|(&w, h)| w * h.confidence(sample, i))
                .grb_sum();
            let name = format!("sample[{i}]");
            model.add_constr(&name, c!(y * expr >= rho - xi))?;
        }

        model.add_constr(
            "sum_is_1", c!(wt_vec.iter().grb_sum() == 1.0)
        )?;
        model.update()?;


        // Set the objective function
        let param = self.nu_inv;
        let objective = rho - param * xi_vec.iter().grb_sum();
        model.set_objective(objective, Maximize)?;
        model.update()?;


        model.optimize()?;


        let status = model.status()?;

        if status != Status::Optimal {
            panic!("Cannot solve the primal problem. Status: {status:?}");
        }
        wt_vec[..].iter()
            .map(|c|
                self.model.get_obj_attr(attr::X, c)
                    .map(f64::abs)
                    .unwrap()
            )
    }
}


