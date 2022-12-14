//! This file defines `SquareLev.R` based on the paper
//! ``Boosting Methods for Regression''
//! by Nigel Duffy and David Helmbold.


use polars::prelude::*;
use rayon::prelude::*;

use crate::{
    Booster,
    WeakLearner,

    State,
    Regressor,
    CombinedHypothesis,
};


/// SquareLev.R algorithm.
pub struct SquareLevR<R> {
    /// Number of examples
    n_sample: usize,


    /// Tolerance parameter
    rho: f64,


    /// Distribution vector on examples
    dist: Vec<f64>,


    /// Residual vector
    residuals: Vec<f64>,


    /// Weights on hypotheses
    weights: Vec<f64>,


    /// Hypotheses
    regressors: Vec<R>,


    /// Max iteration
    max_iter: usize,


    /// Terminated iteration
    terminated: usize,
}


impl<R> SquareLevR<R> {
    /// Initialize `SquareLev.R`
    pub fn init(data: &DataFrame, target: &Series) -> Self {
        let n_sample = data.shape().0;

        let residuals = target.f64()
            .expect("The target class is not a dtype f64")
            .into_iter()
            .map(|y| y.unwrap())
            .collect::<Vec<f64>>();

        assert_ne!(n_sample, 0);


        Self {
            n_sample,
            rho: 1e-2,
            dist: Vec::new(),
            residuals,
            weights: Vec::new(),
            regressors: Vec::new(),

            max_iter: usize::MAX,
            terminated: usize::MAX,
        }
    }


    /// Set the parameter `kappa`.
    #[inline(always)]
    pub fn tolerance(mut self, rho: f64) -> Self {
        self.rho = rho;
        self
    }


    fn stop_now(
        &self,
        r_bar: f64, // Mean of `res`
        it: usize,  // Current iteration
    ) -> bool
    {
        let diff = self.residuals.par_iter()
            .copied()
            .map(|ri| (ri - r_bar).powi(2))
            .sum::<f64>();


        // DEBUG
        if it % 10 == 0 {
            println!("loss (iter: {it:>3}): {}", diff / self.n_sample as f64);
        }

        !(diff >= self.rho * self.n_sample as f64 && it < self.max_iter)
    }
}



impl<R: Regressor> SquareLevR<R> {
    fn update_residuals(
        &mut self,
        data: &DataFrame, // Training instances
        alpha: f64,       // Weight on f
        f: &R,            // A newly attained hypothesis
    )
    {
        self.residuals.iter_mut()
            .enumerate()
            .for_each(|(i, ri)| {
                *ri -= alpha * f.predict(data, i);
            });
    }


    fn weight_on_new_regressor(
        &self,
        data: &DataFrame,
        r_bar: f64,
        f: &R,
    ) -> f64
    {
        let f = f.predict_all(data);
        let f_bar = f.iter()
            .sum::<f64>()
            / self.n_sample as f64;


        let mut r_norm = 0.0;
        let mut f_norm = 0.0;
        let mut res_dot_f = 0.0;


        self.residuals.iter()
            .zip(f)
            .for_each(|(&ri, fi)| {
                let r_diff = ri - r_bar;
                let f_diff = fi - f_bar;

                r_norm += r_diff.powi(2);
                f_norm += f_diff.powi(2);

                res_dot_f += r_diff * f_diff;
            });

        r_norm = r_norm.sqrt();
        f_norm = f_norm.sqrt();

        let epsilon = res_dot_f / (r_norm * f_norm);
        assert!(epsilon.is_finite());

        let alpha = epsilon * r_norm / f_norm;

        alpha
    }
}


impl<R> Booster<R> for SquareLevR<R>
    where R: Regressor + Clone + std::fmt::Debug
{
    fn preprocess<W>(
        &mut self,
        _weak_learner: &W,
        data: &DataFrame,
        target: &Series,
    )
        where W: WeakLearner<Hypothesis = R>
    {
        self.n_sample = data.shape().0;

        let uni = 1.0 / self.n_sample as f64;

        self.dist = vec![uni; self.n_sample];
        self.weights = Vec::new();
        self.regressors = Vec::new();

        self.residuals = target.f64()
            .expect("The target class is not a dtype f64")
            .into_iter()
            .map(|y| y.unwrap())
            .collect::<Vec<f64>>();

        self.terminated = self.max_iter;
    }


    fn boost<W>(
        &mut self,
        weak_learner: &W,
        data: &DataFrame,
        _target: &Series,
        iteration: usize,
    ) -> State
        where W: WeakLearner<Hypothesis = R>
    {
        // Check stopping conditions
        let res_mean = self.residuals.par_iter()
            .sum::<f64>()
            / self.n_sample as f64;


        if self.stop_now(res_mean, iteration) {
            self.terminated = iteration;
            return State::Terminate;
        }

        // Modify the labels
        let y_tilde = self.residuals.iter()
            .map(|r| r - res_mean)
            .collect::<Series>();


        // Obtain a new hypothesis
        let f = weak_learner.produce(data, &y_tilde, &self.dist[..]);


        // Obtain the weight on the new hypothesis `f`.
        let alpha = self.weight_on_new_regressor(
            data,
            res_mean,
            &f
        );
        assert!(alpha.is_finite());


        self.update_residuals(data, alpha, &f);
        self.weights.push(alpha);
        self.regressors.push(f);

        State::Continue
    }


    fn postprocess<W>(
        &mut self,
        _weak_learner: &W,
        _data: &DataFrame,
        _target: &Series,
    ) -> CombinedHypothesis<R>
        where W: WeakLearner<Hypothesis = R>
    {
        let regs = self.weights.clone()
            .into_iter()
            .zip(self.regressors.clone())
            .collect::<Vec<(f64, R)>>();

        CombinedHypothesis::from(regs)
    }
}


