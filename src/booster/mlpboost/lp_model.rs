use grb::prelude::*;


use crate::{Sample, Classifier};
use crate::common::utils;

pub(super) struct LPModel {
    model: Model,
    gamma: Var,
    dist: Vec<Var>,
    constrs: Vec<Constr>,
}


impl LPModel {
    pub(super) fn init(n_sample: usize, upper_bound: f64) -> Self {
        let mut env = Env::new("")
            .expect("Failed to construct a new `Env` for MLPBoost");
        env.set(param::OutputFlag, 0)
            .expect("Failed to set `param::OutputFlag` to `0`");

        let mut model = Model::with_env("MLPBoost", env)
            .expect("Failed to construct a new model for `MLPBoost`");


        // Set GRBVars
        let gamma = add_ctsvar!(model, name: "gamma", bounds: ..)
            .expect("Failed to add a new variable `gamma`");

        let dist = (0..n_sample).map(|i| {
                let name = format!("d[{i}]");
                add_ctsvar!(model, name: &name, bounds: 0_f64..upper_bound)
            }).collect::<Result<Vec<_>, _>>()
            .expect("Failed to add new variables `d[..]`");


        // Set a constraint
        model.add_constr("sum_is_1", c!(dist.iter().grb_sum() == 1.0))
            .expect("Failed to set the constraint `sum( d[..] ) = 1.0`");


        // Set objective function
        model.set_objective(gamma, Minimize)
            .expect("Failed to set the LP objective `gamma`");

        // Update the model
        model.update()
            .expect("Failed to update the model after setting the objective");

        // `constrs` keesp all the constraints for the past hypotheses.
        let constrs = Vec::new();

        Self { model, gamma, dist, constrs, }
    }


    /// Solve the edge minimization problem over the hypotheses
    /// `h1, ..., ht`.
    /// The argument `h` is the new hypothesis `ht`.
    pub(super) fn update<C>(
        &mut self,
        sample: &Sample,
        opt_h: Option<&C>
    ) -> Vec<f64>
        where C: Classifier
    {
        // If we got a new hypothesis,
        // 1. append the corresponding constraint, and
        // 2. optimize the model.
        if let Some(h) = opt_h {
            let edge = utils::margins_of_hypothesis(sample, h)
                .into_iter()
                .zip(self.dist.iter().copied())
                .map(|(yh, d)| d * yh)
                .grb_sum();


            let name = format!("{t}-th hypothesis", t = self.constrs.len());
            self.constrs.push(
                self.model.add_constr(&name, c!(edge <= self.gamma))
                    .expect("Failed to add a new constraint `edge <= gamma`")
            );
            self.model.update()
                .expect(
                    "Failed to update the model after adding a new constraint"
                );

            self.model.optimize()
                .expect("Failed to optimize the problem");

            let status = self.model.status()
                .expect("Failed to get the model status");
            if status != Status::Optimal {
                panic!("Status is {status:?}. Something wrong.");
            }
        }
        self.constrs.iter()
            .map(|c| self.model.get_obj_attr(attr::Pi, c).map(f64::abs))
            .collect::<Result<Vec<_>, _>>()
            .expect("Failed to get the dual solution `w[..]`")
    }
}


