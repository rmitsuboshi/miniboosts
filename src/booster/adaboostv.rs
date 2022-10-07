//! Provides the `AdaBoost*` by Rätsch & Warmuth, 2005.
use polars::prelude::*;
use rayon::prelude::*;


use crate::{Classifier, CombinedClassifier};
use crate::BaseLearner;
use crate::Booster;



/// Struct `AdaBoostV` has 4 parameters.
/// 
/// - `tolerance` is the gap parameter,
/// - `rho` is a guess of the optimal margin,
/// - `gamma` is the minimum edge over the past edges,
/// - `dist` is the distribution over training examples,
pub struct AdaBoostV {
    tolerance: f64,
    rho: f64,
    gamma: f64,
    dist: Vec<f64>,
}


impl AdaBoostV {
    /// Initialize the `AdaBoostV<D, L>`.
    pub fn init(df: &DataFrame) -> Self {
        let (m, _) = df.shape();
        assert!(m != 0);


        let uni = 1.0 / m as f64;
        let dist = (0..m).into_par_iter()
            .map(|_| uni)
            .collect::<Vec<_>>();

        AdaBoostV {
            tolerance: 0.0,
            rho:       1.0,
            gamma:     1.0,
            dist,
        }
    }



    /// Set the tolerance parameter.
    #[inline]
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
    }


    /// `max_loop` returns the maximum iteration
    /// of the `AdaBoostV` to find a combined hypothesis
    /// that has error at most `eps`.
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


impl<C> Booster<C> for AdaBoostV
    where C: Classifier,
{


    fn run<B>(&mut self,
              base_learner: &B,
              data: &DataFrame,
              target: &Series,
              eps: f64)
        -> CombinedClassifier<C>
        where B: BaseLearner<Clf = C>,
    {
        // Initialize parameters
        let (m, _) = data.shape();
        self.dist = vec![1.0 / m as f64; m];
        self.set_tolerance(eps);

        let mut weighted_classifier = Vec::new();


        let max_loop = self.max_loop();
        println!("max_loop: {max_loop}");

        for _t in 1..=max_loop {
            // Get a new hypothesis
            let h = base_learner.produce(data, target, &self.dist);


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
                weighted_classifier = vec![(edge.signum(), h)];
                println!("Break loop after: {_t} iterations");
                break;
            }


            // Compute the weight on the new hypothesis
            let weight = self.update_params(margins, edge);
            weighted_classifier.push((weight, h));
        }

        CombinedClassifier::from(weighted_classifier)

    }
}

