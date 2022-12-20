//! Provides the `AdaBoost*` by Rätsch & Warmuth, 2005.
use polars::prelude::*;
use rayon::prelude::*;


use crate::{
    Booster,
    WeakLearner,

    State,
    Classifier,
    CombinedHypothesis,
};


use crate::research::Logger;



/// Struct `AdaBoostV` has 4 parameters.
/// 
/// - `tolerance` is the gap parameter,
/// - `rho` is a guess of the optimal margin,
/// - `gamma` is the minimum edge over the past edges,
/// - `dist` is the distribution over training examples,
pub struct AdaBoostV<F> {
    tolerance: f64,
    rho: f64,
    gamma: f64,
    dist: Vec<f64>,

    weighted_classifiers: Vec<(f64, F)>,

    max_iter: usize,

    terminated: usize,
}


impl<F> AdaBoostV<F> {
    /// Initialize the `AdaBoostV<D, L>`.
    pub fn init(data: &DataFrame, _target: &Series) -> Self {
        let n_sample = data.shape().0;
        assert!(n_sample != 0);


        let uni = 1.0 / n_sample as f64;
        let dist = vec![uni; n_sample];

        AdaBoostV {
            tolerance: 0.0,
            rho:       1.0,
            gamma:     1.0,
            dist,

            weighted_classifiers: Vec::new(),

            max_iter: usize::MAX,
            terminated: usize::MAX,
        }
    }



    /// Set the tolerance parameter.
    #[inline]
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;

        self
    }


    /// `max_loop` returns the maximum iteration
    /// of the `AdaBoostV` to find a combined hypothesis
    /// that has error at most `tolerance`.
    /// After the `self.max_loop()` iterations,
    /// `AdaBoost` guarantees zero training error in terms of zero-one loss
    /// if the training examples are linearly separable.
    #[inline]
    pub fn max_loop(&self) -> usize {
        let m = self.dist.len();

        (2.0 * (m as f64).ln() / self.tolerance.powi(2)) as usize
    }


    /// Returns a weight on the new hypothesis.
    /// `update_params` also updates `self.dist`
    #[inline]
    fn update_params(&mut self, margins: Vec<f64>, edge: f64)
        -> f64
    {


        // Update edge & margin estimation parameters
        self.gamma = edge.min(self.gamma);
        self.rho = self.gamma - self.tolerance;


        let weight = {
            let e = ((1.0 + edge) / (1.0 - edge)).ln() / 2.0;
            let m = ((1.0 + self.rho) / (1.0 - self.rho)).ln() / 2.0;

            e - m
        };


        // To prevent overflow, take the logarithm.
        self.dist.par_iter_mut()
            .zip(margins)
            .for_each(|(d, yh)| *d = d.ln() - weight * yh);


        let m = self.dist.len();
        let mut indices = (0..m).collect::<Vec<usize>>();
        indices.sort_unstable_by(|&i, &j| {
            self.dist[i].partial_cmp(&self.dist[j]).unwrap()
        });


        let mut normalizer = self.dist[indices[0]];
        for i in indices.into_iter().skip(1) {
            let mut a = normalizer;
            let mut b = self.dist[i];
            if a < b {
                std::mem::swap(&mut a, &mut b);
            }

            normalizer = a + (1.0 + (b - a).exp()).ln();
        }


        self.dist.par_iter_mut()
            .for_each(|d| *d = (*d - normalizer).exp());

        weight
    }
}


impl<F> Booster<F> for AdaBoostV<F>
    where F: Classifier + Clone,
{
    fn preprocess<W>(
        &mut self,
        _weak_learner: &W,
        data: &DataFrame,
        _target: &Series,
    )
        where W: WeakLearner<Hypothesis = F>
    {
        // Initialize parameters
        let n_sample = data.shape().0;
        self.dist = vec![1.0 / n_sample as f64; n_sample];


        self.weighted_classifiers = Vec::new();


        self.max_iter = self.max_loop();
    }


    fn boost<W>(
        &mut self,
        weak_learner: &W,
        data: &DataFrame,
        target: &Series,
        iteration: usize,
    ) -> State
        where W: WeakLearner<Hypothesis = F>,
    {
        if self.max_iter < iteration {
            return State::Terminate;
        }

        // Get a new hypothesis
        let h = weak_learner.produce(data, target, &self.dist);


        // Each element in `predictions` is the product of
        // the predicted vector and the correct vector
        let margins = target.i64()
            .expect("The target class is not an dtype i64")
            .into_iter()
            .enumerate()
            .map(|(i, y)| (y.unwrap() as f64 * h.confidence(data, i)))
            .collect::<Vec<f64>>();


        let edge = margins.iter()
            .zip(&self.dist[..])
            .map(|(&yh, &d)| yh * d)
            .sum::<f64>();


        // If `h` predicted all the examples in `data` correctly,
        // use it as the combined classifier.
        if edge.abs() >= 1.0 {
            self.terminated = iteration;
            self.weighted_classifiers = vec![(edge.signum(), h)];
            return State::Terminate;
        }


        // Compute the weight on the new hypothesis
        let weight = self.update_params(margins, edge);
        self.weighted_classifiers.push((weight, h));

        State::Continue
    }


    fn postprocess<W>(
        &mut self,
        _weak_learner: &W,
        _data: &DataFrame,
        _target: &Series,
    ) -> CombinedHypothesis<F>
        where W: WeakLearner<Hypothesis = F>
    {
        CombinedHypothesis::from(self.weighted_classifiers.clone())
    }
}



impl<F> Logger for AdaBoostV<F>
    where F: Classifier
{
    /// AdaBoost optimizes the exp loss
    fn objective_value(&self, data: &DataFrame, target: &Series)
        -> f64
    {
        let n_sample = data.shape().0 as f64;

        target.i64()
            .expect("The target class is not a dtype i64")
            .into_iter()
            .map(|y| y.unwrap() as f64)
            .enumerate()
            .map(|(i, y)| (- y * self.prediction(data, i)).exp())
            .sum::<f64>()
            / n_sample
    }


    fn prediction(&self, data: &DataFrame, i: usize) -> f64 {
        self.weighted_classifiers.iter()
            .map(|(w, h)| w * h.confidence(data, i))
            .sum::<f64>()
    }
}
