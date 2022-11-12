//! Provides `AdaBoost` by Freund & Schapire, 1995.
use polars::prelude::*;
use rayon::prelude::*;


use crate::{Classifier, CombinedClassifier};
use crate::BaseLearner;
use crate::Booster;


/// Defines `AdaBoost`.
pub struct AdaBoost {
    dist: Vec<f64>,
    tolerance: f64,
}


impl AdaBoost {
    /// Initialize the `AdaBoost`.
    /// This method just sets the parameter `AdaBoost` holds.
    pub fn init(data: &DataFrame, _target: &Series) -> Self {
        assert!(!data.is_empty());
        let (m, _) = data.shape();

        let uni = 1.0 / m as f64;
        AdaBoost {
            dist: vec![uni; m],
            tolerance: 1.0 / (m as f64 + 1.0),
        }
    }


    /// `max_loop` returns the maximum iteration
    /// of the `AdaBoost` to find a combined hypothesis
    /// that has error at most `eps`.
    /// After the `self.max_loop()` iterations,
    /// `AdaBoost` guarantees zero training error in terms of zero-one loss
    /// if the training examples are linearly separable.
    pub fn max_loop(&self) -> usize {
        let m = self.dist.len();

        ((m as f64).ln() / self.tolerance.powi(2)) as usize
    }


    /// Set the tolerance parameter.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }


    /// Returns a weight on the new hypothesis.
    /// `update_params` also updates `self.dist`
    #[inline]
    fn update_params(&mut self,
                     margins: Vec<f64>,
                     edge: f64)
        -> f64
    {
        let m = self.dist.len();


        // Compute the weight on new hypothesis.
        // This is the returned value of this function.
        let weight = ((1.0 + edge) / (1.0 - edge)).ln() / 2.0;


        // To prevent overflow, take the logarithm.
        self.dist.par_iter_mut()
            .zip(margins)
            .for_each(|(d, p)| *d = d.ln() - weight * p);


        // Sort indices by ascending order
        let mut indices = (0..m).into_par_iter()
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


impl<C> Booster<C> for AdaBoost
    where C: Classifier,
{
    fn run<B>(
        &mut self,
        base_learner: &B,
        data: &DataFrame,
        target: &Series,
    ) -> CombinedClassifier<C>
        where B: BaseLearner<Clf = C>,
    {
        // Initialize parameters
        let (m, _) = data.shape();
        let uni = 1.0 / m as f64;
        self.dist = vec![uni; m];

        let mut weighted_classifier = Vec::new();


        let max_loop = self.max_loop();
        println!("max_loop: {max_loop}");

        for step in 1..=max_loop {
            // Get a new hypothesis
            let h = base_learner.produce(data, target, &self.dist);


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
                weighted_classifier = vec![(edge.signum(), h)];
                println!("Break loop at: {step}");
                break;
            }


            // Compute the weight on the new hypothesis
            let weight = self.update_params(margins, edge);
            weighted_classifier.push((weight, h));
        }

        CombinedClassifier::from(weighted_classifier)

    }
}

