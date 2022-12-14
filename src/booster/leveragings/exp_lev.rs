//! This file defines `ExpLev` based on the paper
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


pub struct ExpLev<R> {
    /// Number of examples
    n_sample: usize,


    /// The soft-max parameter `s` in the paper.
    softmax_param: f64,


    /// Tolerance parameter
    eta: f64,


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


    /// Maximal edge
    epsilon_max: f64,
}


impl<R> ExpLev<R> {
    /// Initialize `ExpLev`
    pub fn init(data: &DataFrame, target: &Series) -> Self {
        let n_sample = data.shape().0;

        let residuals = target.f64()
            .expect("The target class is not a dtype f64")
            .into_iter()
            .map(|y| y.unwrap())
            .collect::<Vec<f64>>();

        let eta = 1e-2;
        let softmax_param = (n_sample as f64).ln() / eta;

        assert_ne!(n_sample, 0);


        Self {
            n_sample,
            softmax_param,
            eta,
            dist: Vec::new(),
            residuals,
            weights: Vec::new(),
            regressors: Vec::new(),

            max_iter: usize::MAX,
            terminated: usize::MAX,

            epsilon_max: 1.0_f64,
        }
    }


    /// Set the tolerance parameter `eta`.
    #[inline(always)]
    pub fn tolerance(mut self, eta: f64) -> Self {
        self.eta = eta;
        self
    }


    fn update_distribution(&mut self) {
        let dist = self.residuals.par_iter()
            .map(|r| {
                let sr = self.eta * r;
                if r > 0 {
                    self.eta.ln() + sr * (1 - (-2.0 * sr).exp()).ln()
                } else if r < 0 {
                    self.eta.ln() - sr * (1 - (2.0 * sr).exp()).ln()
                } else {
                    f64::MIN
                }
            })
            .collect::<Vec<f64>>();

        let mut indices = (0..self.n_sample).into_iter()
            .collect::<Vec<usize>>();

        // Sort the indices in the non-increasing order.
        indices.sort_by(|&i, &j| dist[i].partial_cmp(&dist[j]).unwrap());


        let log_denom = indices.into_iter()
            .fold(0.0, |(acc, i)| {
                let d = dist[i];
                let a = acc.max(d);
                let b = acc.min(d);

                a + (1.0 + (b - a).exp()).ln()
            });


        self.dist = dist.into_par_iter()
            .map(|d| (d - log_denom).exp())
            .collect::<Vec<_>>();
    }


    fn stop_now(&self, residual_max: f64) -> bool {
        residual_max <= self.eta
    }
}


impl<R: Regressor> ExpLev<R> {
    fn weight_on_new_regressor(
        &self,
        data: &DataFrame,
        y_tilde: Series,
        f: &R
    ) -> f64
    {
        let edge = y_tilde.f64()
            .expect("The target class is not a dtype of f64")
            .iter()
            .zip(&self.dist[..])
            .enumerate()
            .map(|(i, (y, d))| {
                let p = f.predict(data, i);
                let y = y.unwrap();

                d * y * p
            })
            .sum::<f64>();

        let eps_hat = edge.min(self.epsilon_max);


        // Compute the L1-norm of the gradient vector of the potential.
        let nabla_p = self.residuals.par_iter()
            .map(|r| {
                let sr = self.eta * r;
                if r > 0 {
                    self.eta.ln() + sr * (1 - (-2.0 * sr).exp()).ln()
                } else if r < 0 {
                    self.eta.ln() - sr * (1 - (2.0 * sr).exp()).ln()
                } else {
                    f64::MIN
                }
            })
            .collect::<Vec<f64>>();

        let mut indices = (0..self.n_sample).into_iter()
            .collect::<Vec<usize>>();

        // Sort the indices in the non-increasing order.
        indices.sort_by(|&i, &j| nabla_p[i].partial_cmp(&nabla_p[j]).unwrap());


        let log_l1norm = indices.into_iter()
            .fold(0.0, |(acc, i)| {
                let d = nabla_p[i];
                let a = acc.max(d);
                let b = acc.min(d);

                a + (1.0 + (b - a).exp()).ln()
            });


    }
}


impl<R> Booster<R> for ExpLev<R>
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

        self.softmax_param = (self.n_sample as f64).ln() / self.eta;

        let uni = 1.0 / self.n_sample as f64;

        self.dist = vec![uni; self.n_sample];
        self.weights = Vec::new();
        self.regressors = Vec::new();

        self.residuals = target.f64()
            .expect("The target class is not a dtype f64")
            .into_iter()
            .map(|y| y.unwrap())
            .collect::<Vec<f64>>();

        self.max_iter = usize::MAX;
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
        let res_max = self.residuals.par_iter().max();


        if self.stop_now(res_max) {
            self.terminated = iteration;
            return State::Terminate;
        }

        // Modify the labels
        let y_tilde = self.residuals.iter()
            .map(|r| if r >= 0 { 1.0 } else { -1.0 })
            .collect::<Series>();


        // Obtain a new hypothesis
        let f = weak_learner.produce(data, &y_tilde, &self.dist[..]);


        // Obtain the weight on the new hypothesis `f`.
        let alpha = self.weight_on_new_regressor(
            data,
            y_tilde,
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


