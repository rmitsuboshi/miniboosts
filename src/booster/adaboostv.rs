//! Provides the `AdaBoost*` by Rätsch & Warmuth, 2005.
use crate::{Data, Sample};
// use crate::{Data, Label, Sample};
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
    pub(crate) tolerance: f64,
    pub(crate) rho:       f64,
    pub(crate) gamma:     f64,
    pub(crate) dist:      Vec<f64>,
}


impl AdaBoostV {
    /// Initialize the `AdaBoostV<D, L>`.
    pub fn init<D, L>(sample: &Sample<D, L>) -> AdaBoostV {
        let m = sample.len();
        assert!(m != 0);
        let uni = 1.0 / m as f64;
        AdaBoostV {
            tolerance:   0.0,
            rho:         1.0,
            gamma:       1.0,
            dist:        vec![uni; m],
        }
    }


    /// `max_loop` returns the maximum iteration
    /// of the `AdaBoostV` to find a combined hypothesis
    /// that has error at most `eps`.
    /// After the `self.max_loop()` iterations,
    /// `AdaBoostV` guarantees no miss-classification on `sample<T>`
    /// if the training examples are linearly separable.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// use lycaon::{Sample, AdaBoostV};
    /// 
    /// let examples = vec![
    ///     vec![1.0, 2.0, 3.0],
    ///     vec![4.0, 5.0, 6.0],
    /// ];
    /// let labels = vec![1.0, -1.0];
    /// 
    /// let sample = Sample::from((examples, labels));
    /// 
    /// let booster = AdaBoostV::init(&sample);
    /// let eps = 0.01_f64;
    /// 
    /// let expected = 2.0 * (sample.len() as f64).ln() / eps.powi(2);
    /// 
    /// assert_eq!(booster.max_loop(eps), expected as usize);
    /// ```
    /// 
    pub fn max_loop(&self, eps: f64) -> usize {
        let m = self.dist.len();

        2 * ((m as f64).ln() / (eps * eps)) as usize
    }

    /// Returns a weight on the new hypothesis.
    /// `update_params` also updates `self.distribution`
    #[inline]
    fn update_params(&mut self,
                     margins: Vec<f64>,
                     edge:    f64)
        -> f64
    {


        // Update edge & margin estimation parameters
        self.gamma = edge.min(self.gamma);
        self.rho   = self.gamma - self.tolerance;


        let weight = {
            let e = ((1.0 + edge) / (1.0 - edge)).ln() / 2.0;
            let m = ((1.0 + self.rho) / (1.0 - self.rho)).ln() / 2.0;

            e - m
        };


        // To prevent overflow, take the logarithm.
        for (d, yh) in self.dist.iter_mut().zip(margins.iter()) {
            *d = d.ln() - weight * yh;
        }


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

        for d in self.dist.iter_mut() {
            *d = (*d - normalizer).exp();
        }

        weight
    }
}


impl<D, C> Booster<D, f64, C> for AdaBoostV
    where D: Data,
          C: Classifier<D, f64> + Eq + PartialEq,
{


    fn run<B>(&mut self,
              base_learner: &B,
              sample:       &Sample<D, f64>,
              eps:          f64)
        -> CombinedClassifier<D, f64, C>
        where B: BaseLearner<D, f64, Clf = C>,
    {
        // Initialize parameters
        let m   = sample.len();
        self.dist      = vec![1.0 / m as f64; m];
        self.tolerance = eps;

        let mut weighted_classifier = Vec::new();


        let max_loop = self.max_loop(eps);
        println!("max_loop: {max_loop}");

        for _t in 1..=max_loop {
            // Get a new hypothesis
            let h = base_learner.best_hypothesis(sample, &self.dist);


            // Each element in `predictions` is the product of
            // the predicted vector and the correct vector
            let margins = sample.iter()
                .map(|(dat, lab)| *lab * h.predict(dat))
                .collect::<Vec<f64>>();


            let edge = margins.iter()
                .zip(self.dist.iter())
                .fold(0.0, |acc, (&yh, &d)| acc + yh * d);


            // If `h` predicted all the examples in `sample` correctly,
            // use it as the combined classifier.
            if edge.abs() >= 1.0 {
                let sgn = edge.signum();
                weighted_classifier = vec![(sgn, h)];
                println!("Break loop after: {_t} iterations");
                break;
            }


            // Compute the weight on the new hypothesis
            let weight = self.update_params(margins, edge);
            weighted_classifier.push(
                (weight, h)
            );
        }

        CombinedClassifier::from(weighted_classifier)

    }
}

