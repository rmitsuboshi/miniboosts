use grb::prelude::*;


use crate::Sample;
use crate::hypothesis::Classifier;

/// A linear programming model for edge minimization. 
pub(super) struct LPModel {
    pub(self) model: Model,
    pub(self) gamma: Var,
    pub(self) dist: Vec<Var>,
    pub(self) constrs: Vec<Constr>,
}


impl LPModel {
    /// Initialize the LP model.
    /// arguments.
    /// - `size`: Number of variables (Number of examples).
    /// - `upper_bound`: Capping parameter. `[1, size]`.
    pub(super) fn init(size: usize, upper_bound: f64) -> Self {
        let mut env = Env::new("")
            .expect("Failed to construct a new `Env` for LPBoost");
        env.set(param::OutputFlag, 0)
            .expect("Failed to set `param::OutputFlag` to `0`");

        let mut model = Model::with_env("LPBoost", env)
            .expect("Failed to construct a new model for `MLPBoost`");


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


        // Set objective function
        model.set_objective(gamma, Minimize)
            .expect("Failed to set the LP objective `gamma`");


        // Update the model
        model.update()
            .expect("Failed to update the model after setting the objective");


        Self {
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
        clf: &F
    ) -> f64
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


        self.model.optimize()
            .expect("Failed to optimize the problem");


        let status = self.model.status()
            .expect("Failed to get the model status");
        if status != Status::Optimal {
            panic!("Status is {status:?}. Something wrong.");
        }


        self.model.get_obj_attr(attr::X, &self.gamma)
            .expect("Failed to get the dual solution `gamma`")
    }

    /// Returns the distribution over examples.
    pub(super) fn distribution(&self)
        -> Vec<f64>
    {
        self.dist.iter()
            .map(|d| self.model.get_obj_attr(attr::X, d))
            .collect::<Result<Vec<_>, _>>()
            .expect("Failed to get the solution `d[..]`")
    }


    /// Returns the weights over the hypotheses.
    pub(super) fn weight(&self) -> impl Iterator<Item=f64> + '_
    {
        self.constrs[0..].iter()
            .map(|c| self.model.get_obj_attr(attr::Pi, c).map(f64::abs).unwrap())
    }
}


