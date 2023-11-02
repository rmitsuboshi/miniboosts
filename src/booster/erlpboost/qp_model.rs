use grb::prelude::*;


use crate::Sample;
use crate::hypothesis::Classifier;

const QP_TOLERANCE: f64 = 1e-9;

/// A linear programming model for edge minimization. 
pub(super) struct QPModel {
    pub(self) eta: f64,
    pub(self) model: Model,
    pub(self) gamma: Var,
    pub(self) dist: Vec<Var>,
    pub(self) constrs: Vec<Constr>,
}


impl QPModel {
    /// Initialize the LP model.
    /// arguments.
    /// - `size`: Number of variables (Number of examples).
    /// - `upper_bound`: Capping parameter. `[1, size]`.
    pub(super) fn init(eta: f64, size: usize, upper_bound: f64)
        -> Self
    {
        let mut env = Env::new("")
            .expect("Failed to construct a new `Env` for ERLPBoost");
        env.set(param::OutputFlag, 0)
            .expect("Failed to set `param::OutputFlag` to `0`");
        env.set(param::NumericFocus, 3)
            .expect("Failed to set `NumericFocus` parameter to `3`");

        let mut model = Model::with_env("ERLPBoost", env)
            .expect("Failed to construct a new model for `ERLPBoost`");


        // Set GRBVars
        let gamma = add_ctsvar!(model, name: "gamma", bounds: ..)
            .expect("Failed to add a new variable `gamma`");

        let dist = (0..size).map(|i| {
                let name = format!("d[{i}]");
                add_ctsvar!(model, name: &name, bounds: 0_f64..upper_bound)
            }).collect::<Result<Vec<_>, _>>()
            .expect("Failed to add new variables `d[..]`");


        // Set a constraint
        model.add_constr("sum_is_1", c!(dist.iter().grb_sum() == 1.0))
            .expect("Failed to set the constraint `sum( d[..] ) = 1.0`");


        // Update the model
        model.update()
            .expect("Faild to update the model after the initialization");


        Self {
            eta,
            model,
            gamma,
            dist,
            constrs: Vec::new(),
        }
    }


    /// Solve the edge minimization problem 
    /// over the hypotheses `h1, ..., ht` 
    /// and outputs the optimal value.
    pub(super) fn update<F>(
        &mut self,
        sample: &Sample,
        dist: &mut [f64],
        clf: &F,
    )
        where F: Classifier
    {
        // If we got a new hypothesis,
        // 1. append a constraint, and
        // 2. optimize the model.
        let edge = sample.target()
            .iter()
            .enumerate()
            .map(|(i, y)| y * clf.confidence(sample, i))
            .zip(self.dist.iter().copied())
            .map(|(yh, d)| d * yh)
            .grb_sum();


        let name = format!("{t}-th hypothesis", t = self.constrs.len());


        self.constrs.push(
            self.model.add_constr(&name, c!(edge <= self.gamma))
                .expect("Failed to add a new constraint `edge <= gamma`")
        );
        self.model.update()
            .expect("Failed to update the model after adding a new constraint");


        let mut old_objval = 1e9;

        loop {
            // Set objective function
            let regularizer = dist.iter()
                .copied()
                .zip(self.dist.iter())
                .map(|(d, &grb_d)| {
                    let l_term = d.ln() * grb_d;
                    let q_term = (0.5_f64 / d) * (grb_d * grb_d);

                    l_term + q_term
                })
                .grb_sum();
            let objective = self.gamma
                + ((1.0_f64 / self.eta) * regularizer);
            self.model.set_objective(objective, Minimize)
                .expect("Failed to set the objective function");


            self.model.optimize()
                .expect("Failed to optimize the problem");


            let status = self.model.status()
                .expect("Failed to get the model status");
            if status != Status::Optimal && status != Status::SubOptimal {
                break;
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
                    any_zero |= g == 0.0;
                    *d = g;
                });


            if any_zero || old_objval - objval < QP_TOLERANCE {
                break;
            }

            old_objval = objval;
        }
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
    pub(super) fn weight(&mut self) -> impl Iterator<Item=f64> + '_
    {
        let objective = self.gamma;
        self.model.set_objective(objective, Minimize)
            .expect("Failed to set the LP objective `gamma`");

        self.model.update()
            .expect(
                "Failed to update the model after setting the LP objective"
            );

        self.model.optimize()
            .expect("Failed to solve the LP");

        let status = self.model.status()
            .expect("Failed to get the model status");

        if status != Status::Optimal {
            panic!("Cannot solve the primal problem. Status: {status:?}");
        }


        self.constrs[0..].iter()
            .map(|c|
                self.model.get_obj_attr(attr::Pi, c)
                    .map(f64::abs)
                    .unwrap()
            )
    }
}


