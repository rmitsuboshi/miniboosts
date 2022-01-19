//! Provides the `AdaBoost` by Freund & Schapire, 1995.
use crate::Sample;
use crate::{Classifier, CombinedClassifier};
use crate::BaseLearner;
use crate::Booster;


/// Struct `AdaBoost` has one parameter.
/// 
/// - `dist` is the distribution over training examples,
pub struct AdaBoost {
    pub(crate) dist: Vec<f64>,
}


impl AdaBoost {
    /// Initialize the `AdaBoost`.
    pub fn init(sample: &Sample) -> AdaBoost {
        let m = sample.len();
        assert!(m != 0);
        let uni = 1.0 / m as f64;
        AdaBoost {
            dist: vec![uni; m],
        }
    }


    /// `max_loop` returns the maximum iteration
    /// of the Adaboost to find a combined hypothesis
    /// that has error at most `eps`.
    pub fn max_loop(&self, eps: f64) -> u64 {
        let m = self.dist.len();

        ((m as f64).ln() / (eps * eps)) as u64
    }


    /// `update_params` updates `self.distribution`
    /// and determine the weight on hypothesis
    /// that the algorithm obtained at current iteration.
    fn update_params(&mut self, predictions: Vec<f64>, edge: f64) -> f64 {
        let m = self.dist.len();


        // Compute the weight on new hypothesis.
        // This is the returned value of this function.
        let weight = ((1.0 + edge) / (1.0 - edge)).ln() / 2.0;


        // To prevent overflow, take the logarithm.
        for (d, p) in self.dist.iter_mut().zip(predictions.iter()) {
            *d = d.ln() - weight * p;
        }


        // Sort indices by ascending order
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


        // Update the distribution
        for d in self.dist.iter_mut() {
            *d = (*d - normalizer).exp();
        }


        weight
    }
}


impl<C> Booster<C> for AdaBoost
    where C: Classifier + Eq + PartialEq
{
    fn run<B>(&mut self, base_learner: &B, sample: &Sample, eps: f64)
        -> CombinedClassifier<C>
        where B: BaseLearner<Clf = C>,
    {
        // Initialize parameters
        let m   = sample.len();
        let uni = 1.0 / m as f64;
        self.dist = vec![uni; m];

        let mut weighted_classifier = Vec::new();


        let max_loop = self.max_loop(eps);
        println!("max_loop: {max_loop}");

        for _t in 1..=max_loop {
            // Get a new hypothesis
            let h = base_learner.best_hypothesis(sample, &self.dist);


            // Each element in `predictions` is the product of
            // the predicted vector and the correct vector
            let predictions = sample.iter()
                .map(|ex| ex.label * h.predict(&ex.data))
                .collect::<Vec<f64>>();


            let edge = predictions.iter()
                .zip(self.dist.iter())
                .fold(0.0, |acc, (&yh, &d)| acc + yh * d);


            // If `h` predicted all the examples in `sample` correctly,
            // use it as the combined classifier.
            if edge >= 1.0 {
                weighted_classifier = vec![(1.0, h)];
                println!("Break loop after: {_t} iterations");
                break;
            }


            // Compute the weight on the new hypothesis
            let weight = self.update_params(predictions, edge);
            weighted_classifier.push(
                (weight, h)
            );
        }

        CombinedClassifier {
            weighted_classifier
        }
    }
}

