//! Provides `AdaBoost` by Freund & Schapire, 1995.
use polars::prelude::*;
use rayon::prelude::*;


use crate::{
    Booster,
    WeakLearner,
    State,
    Classifier,
    CombinedHypothesis
};


/// Defines `AdaBoost`.
pub struct AdaBoost<F> {
    dist: Vec<f64>,
    tolerance: f64,

    weighted_classifiers: Vec<(f64, F)>,

    max_iter: usize,

    terminated: usize,
}


impl<F> AdaBoost<F> {
    /// Initialize the `AdaBoost`.
    /// This method just sets the parameter `AdaBoost` holds.
    pub fn init(data: &DataFrame, _target: &Series) -> Self {
        assert!(!data.is_empty());
        let (n_sample, _) = data.shape();

        let uni = 1.0 / n_sample as f64;
        AdaBoost {
            dist: vec![uni; n_sample],
            tolerance: 1.0 / (n_sample as f64 + 1.0),

            weighted_classifiers: Vec::new(),

            max_iter: usize::MAX,

            terminated: usize::MAX,
        }
    }


    /// `max_loop` returns the maximum iteration
    /// of the `AdaBoost` to find a combined hypothesis
    /// that has error at most `eps`.
    /// After the `self.max_loop()` iterations,
    /// `AdaBoost` guarantees zero training error in terms of zero-one loss
    /// if the training examples are linearly separable.
    pub fn max_loop(&self) -> usize {
        let n_sample = self.dist.len();

        ((n_sample as f64).ln() / self.tolerance.powi(2)) as usize
    }


    /// Set the tolerance parameter.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }


    /// Returns a weight on the new hypothesis.
    /// `update_params` also updates `self.dist`
    #[inline]
    fn update_params(
        &mut self,
        margins: Vec<f64>,
        edge: f64
    ) -> f64
    {
        let n_sample = self.dist.len();


        // Compute the weight on new hypothesis.
        // This is the returned value of this function.
        let weight = ((1.0 + edge) / (1.0 - edge)).ln() / 2.0;


        // To prevent overflow, take the logarithm.
        self.dist.par_iter_mut()
            .zip(margins)
            .for_each(|(d, p)| *d = d.ln() - weight * p);


        // Sort indices by ascending order
        let mut indices = (0..n_sample).into_par_iter()
            .collect::<Vec<usize>>();
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



        // Update self.dist
        self.dist.par_iter_mut()
            .for_each(|d| *d = (*d - normalizer).exp());


        weight
    }
}


impl<F> Booster<F> for AdaBoost<F>
    where F: Classifier + Clone,
{
    fn preprocess<W>(
        &mut self,
        _weak_learner: &W,
        data: &DataFrame,
        _target: &Series,
    )
        where W: WeakLearner<Clf = F>
    {
        // Initialize parameters
        let n_sample = data.shape().0;
        let uni = 1.0 / n_sample as f64;
        self.dist = vec![uni; n_sample];

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
        where W: WeakLearner<Clf = F>,
    {
        if self.max_iter < iteration {
            return State::Terminate;
        }


        // Get a new hypothesis
        let h = weak_learner.produce(data, target, &self.dist);


        // Each element in `margins` is the product of
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


        // If `h` predicted all the examples in `sample` correctly,
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
        where W: WeakLearner<Clf = F>
    {
        CombinedHypothesis::from(self.weighted_classifiers.clone())
    }
}

