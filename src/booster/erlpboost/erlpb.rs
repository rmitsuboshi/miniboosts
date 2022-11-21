//! This file defines `ERLPBoost` based on the paper
//! "Entropy Regularized LPBoost"
//! by Warmuth et al.
//! 
use polars::prelude::*;
// use rayon::prelude::*;


use crate::{
    Booster,
    BaseLearner,

    State,
    Classifier,
    CombinedClassifier,
};
use super::qp_model::QPModel;


use std::cell::RefCell;



/// ERLPBoost struct. 
/// See [this paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.141.1759&rep=rep1&type=pdf).
pub struct ERLPBoost<C> {
    dist: Vec<f64>,

    // `gamma_hat` corresponds to $\min_{q=1, .., t} P^q (d^{q-1})$
    gamma_hat: f64,

    // `gamma_star` corresponds to $P^{t-1} (d^{t-1})$
    gamma_star: f64,
    // regularization parameter defined in the paper
    eta: f64,

    tolerance: f64,

    qp_model: Option<RefCell<QPModel>>,

    classifiers: Vec<C>,


    // an accuracy parameter for the sub-problems
    n_sample: usize,
    nu: f64,


    terminated: usize,

    max_iter: usize,
}


impl<C> ERLPBoost<C> {
    /// Initialize the `ERLPBoost`.
    /// Use `data` for argument.
    /// This method does not care 
    /// whether the label is included in `data` or not.
    pub fn init(data: &DataFrame, _target: &Series) -> Self {
        let n_sample = data.shape().0;
        assert!(n_sample != 0);


        // Set uni as an uniform weight
        let uni = 1.0 / n_sample as f64;

        // Compute $\ln(n_sample)$ in advance
        let ln_n_sample = (n_sample as f64).ln();


        // Set tolerance
        let tolerance = uni / 2.0;


        // Set regularization parameter
        let eta = 0.5_f64.max(2.0_f64 * ln_n_sample / tolerance);

        // Set gamma_hat and gamma_star
        let gamma_hat  = 1.0;
        let gamma_star = f64::MIN;


        ERLPBoost {
            dist: vec![uni; n_sample],
            gamma_hat,
            gamma_star,
            eta,
            tolerance,
            qp_model: None,

            classifiers: Vec::new(),

            n_sample,
            nu: 1.0,

            terminated: usize::MAX,
            max_iter: usize::MAX,
        }
    }


    fn init_solver(&mut self) {
        let upper_bound = 1.0 / self.nu;

        assert!((0.0..=1.0).contains(&upper_bound));


        let qp_model = RefCell::new(QPModel::init(
            self.eta, self.n_sample, upper_bound
        ));

        self.qp_model = Some(qp_model);
    }


    /// Updates the capping parameter.
    pub fn nu(mut self, nu: f64) -> Self {
        assert!(1.0 <= nu && nu <= self.n_sample as f64);
        self.nu = nu;
        self.regularization_param();

        self
    }


    /// Returns the break iteration.
    /// This method returns `0` before the `.run()` call.
    #[inline(always)]
    pub fn terminated(&self) -> usize {
        self.terminated
    }


    /// Set the tolerance parameter.
    #[inline(always)]
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance / 2.0;
        self
    }


    /// Setter method of `self.eta`
    #[inline(always)]
    fn regularization_param(&mut self) {
        let ln_n_sample = (self.n_sample as f64 / self.nu).ln();


        self.eta = 0.5_f64.max(ln_n_sample / self.tolerance);
    }


    /// `max_loop` returns the maximum iteration
    /// of the Adaboost to find a combined hypothesis
    /// that has error at most `tolerance`.
    fn max_loop(&mut self) -> usize {
        let n_sample = self.n_sample as f64;

        let mut max_iter = 4.0 / self.tolerance;


        let ln_n_sample = (n_sample / self.nu).ln();
        let temp = 8.0 * ln_n_sample / self.tolerance.powi(2);


        max_iter = max_iter.max(temp);

        max_iter.ceil() as usize
    }
}


impl<C> ERLPBoost<C>
    where C: Classifier
{
    /// Update `self.gamma_hat`.
    /// `self.gamma_hat` holds the minimum value of the objective value.
    #[inline]
    fn update_gamma_hat_mut(
        &mut self,
        h: &C,
        data: &DataFrame,
        target: &Series
    )
    {
        let edge = target.i64()
            .expect("The target class is not a dtype i64")
            .into_iter()
            .zip(self.dist.iter().copied())
            .enumerate()
            .map(|(i, (y, d))| d * y.unwrap() as f64 * h.confidence(data, i))
            .sum::<f64>();


        let m = self.dist.len() as f64;
        let entropy = self.dist.iter()
            .copied()
            .map(|d| if d == 0.0 { 0.0 } else { d * d.ln() })
            .sum::<f64>() + m.ln();


        let obj_val = edge + (entropy / self.eta);

        self.gamma_hat = self.gamma_hat.min(obj_val);
    }


    /// Update `self.gamma_star`.
    /// `self.gamma_star` holds the current optimal value.
    fn update_gamma_star_mut(&mut self, data: &DataFrame, target: &Series)
    {
        let max_edge = self.classifiers.iter()
            .map(|h|
                target.i64()
                    .expect("The target class is not a dtype i64")
                    .into_iter()
                    .zip(self.dist.iter().copied())
                    .enumerate()
                    .map(|(i, (y, d))|
                        d * y.unwrap() as f64 * h.confidence(data, i)
                    )
                    .sum::<f64>()
            )
            .reduce(f64::max)
            .unwrap();


        let entropy = self.dist.iter()
            .copied()
            .map(|d| d * d.ln())
            .sum::<f64>();


        let m = self.dist.len() as f64;
        self.gamma_star = max_edge + (entropy + m.ln()) / self.eta;
    }


    /// Updates `self.dist`
    /// This method repeatedly minimizes the quadratic approximation of 
    /// ERLPB. objective around current distribution `self.dist`.
    /// Then update `self.dist` as the optimal solution of 
    /// the approximate problem. 
    /// This method continues minimizing the quadratic objective 
    /// while the decrease of the optimal value is 
    /// greater than `self.sub_tolerance`.
    fn update_distribution_mut(
        &mut self,
        data: &DataFrame,
        target: &Series,
        clf: &C
    )
    {
        self.qp_model.as_ref()
            .unwrap()
            .borrow_mut()
            .update(data, target, &mut self.dist[..], clf);

        self.dist = self.qp_model.as_ref()
            .unwrap()
            .borrow()
            .distribution();
    }
}


impl<C> Booster<C> for ERLPBoost<C>
    where C: Classifier + Clone,
{
    fn preprocess<B>(
        &mut self,
        _base_learner: &B,
        data: &DataFrame,
        _target: &Series,
    )
        where B: BaseLearner<Clf = C>
    {
        let n_sample = data.shape().0;
        let uni = 1.0 / n_sample as f64;

        self.dist = vec![uni; n_sample];

        self.max_iter = self.max_loop();
        self.terminated = self.max_iter;

        self.classifiers = Vec::new();

        self.gamma_hat = 1.0;
        self.gamma_star = -1.0;


        assert!((0.0..1.0).contains(&self.tolerance));
        self.regularization_param();
        self.init_solver();
    }


    fn boost<B>(
        &mut self,
        base_learner: &B,
        data: &DataFrame,
        target: &Series,
        iteration: usize,
    ) -> State
        where B: BaseLearner<Clf = C>,
    {
        if self.max_iter < iteration {
            return State::Terminate;
        }

        // Receive a hypothesis from the base learner
        let h = base_learner.produce(data, target, &self.dist[..]);


        // update `self.gamma_hat`
        self.update_gamma_hat_mut(&h, data, target);


        // Check the stopping criterion
        let diff = self.gamma_hat - self.gamma_star;
        if diff <= self.tolerance {
            self.terminated = iteration;
            return State::Terminate;
        }

        // At this point, the stopping criterion is not satisfied.

        // Update the parameters
        self.update_distribution_mut(data, target, &h);


        // Append a new hypothesis to `clfs`.
        self.classifiers.push(h);


        // update `self.gamma_star`.
        self.update_gamma_star_mut(data, target);

        State::Continue
    }


    fn postprocess<B>(
        &mut self,
        _base_learner: &B,
        _data: &DataFrame,
        _target: &Series,
    ) -> CombinedClassifier<C>
        where B: BaseLearner<Clf = C>
    {
        let clfs = self.qp_model.as_ref()
            .unwrap()
            .borrow_mut()
            .weight()
            .zip(self.classifiers.clone())
            .filter(|(w, _)| *w != 0.0)
            .collect::<Vec<(f64, C)>>();


        CombinedClassifier::from(clfs)
    }
}


