//! This file defines `SoftBoost` based on the paper
//! "Boosting Algorithms for Maximizing the Soft Margin"
//! by Warmuth et al.
//! 
use crate::{
    Sample,
    Booster,
    WeakLearner,

    State,
    Classifier,
    CombinedHypothesis,
    research::Research,
};


use grb::prelude::*;



/// `SoftBoost`.
/// This algorithm is based on this paper: 
/// [Boosting Algorithms for Maximizing the Soft Margin](https://papers.nips.cc/paper/2007/hash/cfbce4c1d7c425baf21d6b6f2babe6be-Abstract.html) 
/// by Gunnar Rätsch, Manfred K. Warmuth, and Laren A. Glocer.
/// 
/// # Example
/// The following code shows a small example 
/// for running [`SoftBoost`](SoftBoost).  
/// See also:
/// - [`SoftBoost::nu`]
/// - [`DTree`]
/// - [`DTreeClassifier`]
/// - [`CombinedHypothesis<F>`]
/// - [`DTree::max_depth`]
/// - [`DTree::criterion`]
/// - [`DataFrame`]
/// - [`Series`]
/// - [`DataFrame::shape`]
/// - [`CsvReader`]
/// 
/// [`SoftBoost::nu`]: SoftBoost::nu
/// [`DTree`]: crate::weak_learner::DTree
/// [`DTreeClassifier`]: crate::weak_learner::DTreeClassifier
/// [`CombinedHypothesis<F>`]: crate::hypothesis::CombinedHypothesis
/// [`DTree::max_depth`]: crate::weak_learner::DTree::max_depth
/// [`DTree::criterion`]: crate::weak_learner::DTree::criterion
/// [`DataFrame`]: polars::prelude::DataFrame
/// [`Series`]: polars::prelude::Series
/// [`DataFrame::shape`]: polars::prelude::DataFrame::shape
/// [`CsvReader`]: polars::prelude::CsvReader
/// 
/// 
/// ```no_run
/// use polars::prelude::*;
/// use miniboosts::prelude::*;
/// 
/// // Read the training data from the CSV file.
/// let mut data = CsvReader::from_path(path_to_csv_file)
///     .unwrap()
///     .has_header(true)
///     .finish()
///     .unwrap();
/// 
/// // Split the column corresponding to labels.
/// let target = data.drop_in_place(class_column_name).unwrap();
/// 
/// // Get the number of training examples.
/// let n_sample = data.shape().0 as f64;
/// 
/// // Initialize `SoftBoost` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis 
/// // whose soft margin objective value is differs at most `0.01`
/// // from the optimal one.
/// // Further, at the end of this chain,
/// // SoftBoost calls `SoftBoost::nu` to set the capping parameter 
/// // as `0.1 * n_sample`, which means that, 
/// // at most, `0.1 * n_sample` examples are regarded as outliers.
/// let booster = SoftBoost::init(&data, &target)
///     .tolerance(0.01)
///     .nu(0.1 * n_sample);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTree::init(&data, &target)
///     .max_depth(2)
///     .criterion(Criterion::Edge);
/// 
/// // Run `SoftBoost` and obtain the resulting hypothesis `f`.
/// let f: CombinedHypothesis<DTreeClassifier> = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions: Vec<i64> = f.predict_all(&data);
/// 
/// // Calculate the training loss.
/// let training_loss = target.i64()
///     .unwrap()
///     .into_iter()
///     .zip(predictions)
///     .map(|(true_label, prediction) {
///         let true_label = true_label.unwrap();
///         if true_label == prediction { 0.0 } else { 1.0 }
///     })
///     .sum::<f64>()
///     / n_sample;
///
///
/// println!("Training Loss is: {training_loss}");
/// ```
pub struct SoftBoost<'a, F> {
    sample: &'a Sample,

    pub(crate) dist: Vec<f64>,

    // `gamma_hat` corresponds to $\min_{q=1, .., t} P^q (d^{q-1})
    gamma_hat: f64,
    tolerance: f64,
    // an accuracy parameter for the sub-problems
    sub_tolerance: f64,
    nu: f64,

    env: Env,


    classifiers: Vec<F>,


    max_iter: usize,
    terminated: usize,


    weights: Vec<f64>,
}


impl<'a, F> SoftBoost<'a, F>
    where F: Classifier
{
    /// Initialize the `SoftBoost`.
    pub fn init(sample: &'a Sample) -> Self {
        let n_sample = sample.shape().0;
        assert!(n_sample != 0);

        let mut env = Env::new("").unwrap();

        env.set(param::OutputFlag, 0).unwrap();

        // Set uni as an uniform weight
        let uni = 1.0 / n_sample as f64;

        let dist = vec![uni; n_sample];


        // Set tolerance, sub_tolerance
        let tolerance = uni;


        // Set gamma_hat
        let gamma_hat = 1.0;


        SoftBoost {
            sample,

            dist,
            gamma_hat,
            tolerance,
            sub_tolerance: 1e-6,
            nu: 1.0,
            env,

            classifiers: Vec::new(),
            weights: Vec::new(),

            max_iter: usize::MAX,
            terminated: usize::MAX,
        }
    }


    /// Set the capping parameter.
    #[inline(always)]
    pub fn nu(mut self, nu: f64) -> Self {
        let n_sample = self.sample.shape().0 as f64;
        assert!((1.0..=n_sample).contains(&nu));

        self.nu = nu;
        self
    }


    /// Set the tolerance parameter.
    #[inline(always)]
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }


    /// `max_loop` returns the maximum iteration
    /// of the Adaboost to find a combined hypothesis
    /// that has error at most `tolerance`.
    pub fn max_loop(&mut self) -> usize {

        let n_sample = self.sample.shape().0 as f64;

        let temp = (n_sample / self.nu).ln();
        let max_iter = 2.0 * temp / self.tolerance.powi(2);

        max_iter.ceil() as usize
    }


    /// Returns a optimal value of the optimization problem LPBoost solves
    pub fn opt_val(&self) -> f64 {
        self.gamma_hat
    }
}


impl<F> SoftBoost<'_, F>
    where F: Classifier,
{
    /// Set the weight on the classifiers.
    /// This function is called at the end of the boosting.
    fn set_weights(&self)
        -> std::result::Result<Vec<f64>, grb::Error>
    {
        let mut model = Model::with_env("", &self.env)?;

        let n_sample = self.sample.shape().0;
        let n_hypotheses = self.classifiers.len();

        // Initialize GRBVars
        let wt_vec = (0..n_hypotheses).map(|i| {
                let name = format!("w[{i}]");
                add_ctsvar!(model, name: &name, bounds: 0_f64..).unwrap()
            }).collect::<Vec<_>>();
        let xi_vec = (0..n_sample).map(|i| {
                let name = format!("xi[{i}]");
                add_ctsvar!(model, name: &name, bounds: 0.0_f64..).unwrap()
            }).collect::<Vec<_>>();
        let rho = add_ctsvar!(model, name: "rho", bounds: ..)?;


        // Set constraints
        let target = self.sample.target();
        let iter = target.into_iter()
            .zip(xi_vec.iter())
            .enumerate();

        for (i, (&y, &xi)) in iter {
            let expr = wt_vec.iter()
                .zip(&self.classifiers[..])
                .map(|(&w, h)| w * h.confidence(self.sample, i))
                .grb_sum();
            let name = format!("sample[{i}]");
            model.add_constr(&name, c!(y * expr >= rho - xi))?;
        }

        model.add_constr(
            "sum_is_1", c!(wt_vec.iter().grb_sum() == 1.0)
        )?;
        model.update()?;


        // Set the objective function
        let param = 1.0 / self.nu;
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


    /// Updates `self.dist`
    /// Returns `None` if the stopping criterion satisfied.
    fn update_params_mut(&mut self) -> Option<()> {
        loop {
            // Initialize GRBModel
            let mut model = Model::with_env("", &self.env).unwrap();


            // Set variables that are used in the optimization problem
            let cap = 1.0 / self.nu;

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
            self.classifiers.iter()
                .enumerate()
                .for_each(|(j, h)| {
                    let expr = vars.iter()
                        .zip(self.dist.iter().copied())
                        .zip(self.sample.target().into_iter())
                        .enumerate()
                        .map(|(i, ((v, d), y))| {
                            let p = h.confidence(self.sample, i);
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
            let n_sample = self.sample.shape().0 as f64;
            let objective = self.dist.iter()
                .zip(vars.iter())
                .map(|(&d, &v)| {
                    let lin_coef = (n_sample * d).ln() + 1.0;
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
            // it implies that a `tolerance`-optimality
            // of the previous solution
            if status == Status::Infeasible || status == Status::InfOrUnbd {
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


        // Current solution is an `tolerance`-approximate solution
        // if `self.dist` contains `0.0`,
        if self.dist.iter().any(|&d| d == 0.0) {
            return None;
        }


        Some(())
    }
}


impl<F> Booster<F> for SoftBoost<'_, F>
    where F: Classifier + Clone,
{
    fn preprocess<W>(
        &mut self,
        _weak_learner: &W,
    )
        where W: WeakLearner<Hypothesis = F>
    {
        let n_sample = self.sample.shape().0;

        let uni = 1.0 / n_sample as f64;

        self.dist = vec![uni; n_sample];

        self.sub_tolerance = self.tolerance / 10.0;

        self.max_iter = self.max_loop();
        self.terminated = self.max_iter;
        self.classifiers = Vec::new();

        self.gamma_hat = 1.0;
    }


    fn boost<W>(
        &mut self,
        weak_learner: &W,
        iteration: usize,
    ) -> State
        where W: WeakLearner<Hypothesis = F>,
    {
        if self.max_iter < iteration {
            return State::Terminate;
        }

        // Receive a hypothesis from the base learner
        let h = weak_learner.produce(self.sample, &self.dist);

        // update `self.gamma_hat`
        let edge = self.sample.target()
            .into_iter()
            .zip(self.dist.iter().copied())
            .enumerate()
            .map(|(i, (y, d))|
                d * y * h.confidence(self.sample, i)
            )
            .sum::<f64>();


        if self.gamma_hat > edge {
            self.gamma_hat = edge;
        }


        // At this point, the stopping criterion is not satisfied.
        // Append a new hypothesis to `self.classifiers`.
        self.classifiers.push(h);

        // Update the parameters
        if self.update_params_mut().is_none() {
            self.terminated = iteration;
            return State::Terminate;
        }

        State::Continue
    }


    fn postprocess<W>(
        &mut self,
        _weak_learner: &W,
    ) -> CombinedHypothesis<F>
        where W: WeakLearner<Hypothesis = F>
    {
        // Set the weights on the hypotheses
        // by solving a linear program
        self.weights = self.set_weights().unwrap();
        let clfs = self.weights.iter()
            .copied()
            .zip(self.classifiers.clone())
            .filter(|(w, _)| *w != 0.0)
            .collect::<Vec<(f64, F)>>();

        CombinedHypothesis::from(clfs)
    }
}



impl<H> Research<H> for SoftBoost<'_, H>
    where H: Classifier + Clone,
{
    fn current_hypothesis(&self) -> CombinedHypothesis<H> {
        let weights = self.set_weights().unwrap();

        let f = weights.iter()
            .copied()
            .zip(self.classifiers.iter().cloned())
            .filter(|(w, _)| *w > 0.0)
            .collect::<Vec<(f64, H)>>();

        CombinedHypothesis::from(f)
    }
}


