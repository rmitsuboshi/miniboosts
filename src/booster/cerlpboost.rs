//! This file defines `CERLPBoost` based on the paper
//! "On the Equivalence of Weak Learnability and Linaer Separability:
//!     New Relaxations and Efficient Boosting Algorithms"
//! by Shai Shalev-Shwartz and Yoram Singer.
//! I named this algorithm `CERLPBoost`
//! since it is referred as `the Corrective version of CERLPBoost`
//! in "Entropy Regularized LPBoost" by Warmuth et al.
//! 
use crate::Sample;
use crate::{Classifier, CombinedClassifier};
use crate::BaseLearner;
use crate::Booster;
use std::collections::HashMap;
use std::hash::Hash;



/// Struct `CERLPBoost` has 3 main parameters.
///     - `dist` is the distribution over training examples,
///     - `weights` is the weights over `classifiers`
///        that the CERLPBoost obtained up to iteration `t`.
///     - `classifiers` is the classifier that the CERLPBoost obtained.
/// The length of `weights` and `classifiers` must be same.
pub struct CERLPBoost {
    pub(crate) dist: Vec<f64>,
    // A regularization parameter defined in the paper
    eta:             f64,

    tolerance:       f64,
    capping_param:   f64,

    // Optimal value (Dual problem)
    dual_optval:     f64,
}


impl CERLPBoost {
    /// Initialize the `CERLPBoost`.
    pub fn init(sample: &Sample) -> CERLPBoost {
        let m = sample.len();
        assert!(m != 0);


        // Set uni as an uniform weight
        let uni = 1.0 / m as f64;


        // Set tolerance, sub_tolerance
        let tolerance = uni;


        // Set regularization parameter
        let capping_param = 1.0;
        let eta = 2.0 * (m as f64 / capping_param).ln() / tolerance;


        CERLPBoost {
            dist: vec![uni; m],
            tolerance,
            eta,
            capping_param: 1.0,
            dual_optval:   1.0,
        }
    }


    /// This method updates the capping parameter.
    pub fn capping(mut self, capping_param: f64) -> Self {
        assert!(
            1.0 <= capping_param
            &&
            capping_param <= self.dist.len() as f64
        );
        self.capping_param = capping_param;

        self.regularization_param();

        self
    }


    /// Update set_tolerance parameter `tolerance`.
    #[inline(always)]
    fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance / 2.0;
        self.regularization_param();
    }


    /// Compute the dual objective value
    #[inline(always)]
    fn dual_objval_mut<C>(&mut self,
                          sample:         &Sample,
                          classifier_map: &HashMap<C, f64>)
        where C: Classifier + Eq + PartialEq + Hash
    {
        self.dual_optval = classifier_map.keys()
            .fold(f64::MIN, |acc, h| {
                let temp = sample.iter()
                    .zip(self.dist.iter())
                    .fold(0.0, |acc2, (ex, &d)|
                        acc2 + d * ex.label * h.predict(&ex.data)
                    );

                acc.max(temp)
            });
    }

    /// Returns an optimal value of the dual problem.
    /// This value is an `self.tolerance`-accurate value of the primal one.
    pub fn opt_val(&self) -> f64 {
        self.dual_optval
    }


    /// Update regularization parameter.
    /// (the regularization parameter on
    ///  `self.tolerance` and `self.capping_param`.)
    #[inline(always)]
    fn regularization_param(&mut self) {
        let ln_part = (self.dist.len() as f64 / self.capping_param).ln();
        self.eta = 2.0 * ln_part / self.tolerance;
    }



    /// returns the maximum iteration of the CERLPBoost
    /// to find a combined hypothesis that has error at most `tolerance`.
    pub fn max_loop(&mut self, tolerance: f64) -> u64 {
        if 2.0 * self.tolerance != tolerance {
            self.set_tolerance(tolerance);
        }

        let m = self.dist.len() as f64;

        let numer = 32.0 * ( m / self.capping_param).ln();
        let denom = self.tolerance * self.tolerance;
        let max_iter = numer / denom;

        max_iter.ceil() as u64
    }




    /// Updates weight on hypotheses and `self.dist` in this order.
    fn update_distribution_mut<C>(&mut self,
                                  classifier_map: &HashMap<C, f64>,
                                  sample: &Sample)
        where C: Classifier + Eq + PartialEq + Hash
    {
        for (d, ex) in self.dist.iter_mut().zip(sample.iter()) {
            // Compute the confidence of the current combined hypothesis
            let p = classifier_map.iter()
                .fold(0.0, |acc, (h, &w)| acc + w * h.predict(&ex.data));

            *d = - self.eta * ex.label * p;
        }

        let m  = self.dist.len();
        // Sort the indices over `self.dist` in non-increasing order.
        let mut indices = (0..m).collect::<Vec<_>>();
        indices.sort_unstable_by(|&i, &j|
            self.dist[j].partial_cmp(&self.dist[i]).unwrap()
        );


        let logsums = indices.iter().rev()
            .fold(Vec::with_capacity(m), |mut vec, &i| {
                // TODO use `get_unchecked`
                let temp = match vec.last() {
                    None => self.dist[i],
                    Some(&val) => {
                        let mut a = val;
                        let mut b = self.dist[i];
                        if a < b { std::mem::swap(&mut a, &mut b) };

                        a + (1.0 + (b - a).exp()).ln()
                    }
                };
                vec.push(temp);
                vec
            })
            .into_iter()
            .rev()
            .collect::<Vec<_>>();


        let ub = 1.0 / self.capping_param;
        let log_cap = self.capping_param.ln();

        let mut indices_with_logsum = indices.into_iter()
            .zip(logsums.into_iter())
            .enumerate();

        while let Some((i, (i_sorted, logsum))) = indices_with_logsum.next() {
            let log_xi = (1.0 - ub * i as f64).ln() - logsum;
            // TODO replace this line into `get_unchecked`
            let d = self.dist[i_sorted];

            // Stopping criterion of this while loop
            if log_xi + d + log_cap <= 0.0 {
                self.dist[i_sorted] = (log_xi + d).exp();
                while let Some((_, (ii, _))) = indices_with_logsum.next() {
                    self.dist[ii] = (log_xi + self.dist[ii]).exp();
                }
                break;
            }

            self.dist[i_sorted] = ub;
        }
    }


    /// Update the weights on hypotheses
    fn update_clf_weight_mut<C>(&self,
                                clfs:    &mut HashMap<C, f64>,
                                new_clf: C,
                                gap_vec: Vec<f64>)
        where C: Classifier + Eq + PartialEq + Hash
    {
        // Numerator
        let numer = gap_vec.iter()
            .zip(self.dist.iter())
            .fold(0.0, |acc, (&v, &d)| acc + v * d);

        let squared_inf_norm = gap_vec.into_iter()
            .fold(f64::MIN, |acc, v| acc.max(v.abs()))
            .powi(2);

        // Denominator
        let denom = self.eta * squared_inf_norm;


        // Name the weight on new hypothesis as `weight`
        let weight = 0.0_f64.max(
            1.0_f64.min(numer / denom)
        );


        // Shrink the weights on past hypotheses
        for val in clfs.values_mut() {
            *val *= 1.0 - weight;
        }


        // Add a new hypothesis with weight
        if let Some(w) = clfs.get_mut(&new_clf) {
            *w += weight;
        } else {
            clfs.insert(new_clf, weight);
        }
    }
}


impl<C> Booster<C> for CERLPBoost
    where C: Classifier + Eq + PartialEq + Hash
{


    fn run<B>(&mut self, base_learner: &B, sample: &Sample, tolerance: f64)
        -> CombinedClassifier<C>
        where B: BaseLearner<Clf = C>
    {
        let max_iter = self.max_loop(tolerance);


        let mut classifier_map: HashMap<C, f64> = HashMap::new();


        // Get a new hypothesis and assign 1.0 as its weight.
        // This block does not exist in the original paper.
        // The stopping criterion does not work without this block.
        // since the sum of weights on hypotheses is not equals to one.
        // If we can ensure the weight on the first hypothesis equals one,
        // we can eliminate this block.
        {
            let h = base_learner.best_hypothesis(sample, &self.dist);
            classifier_map.insert(h, 1.0);
        }


        for t in 1..=max_iter {
            // Update the distribution over examples
            self.update_distribution_mut(&classifier_map, &sample);

            // Receive a hypothesis from the base learner
            let h = base_learner.best_hypothesis(sample, &self.dist);


            let gap_vec = sample.iter()
                .map(|ex| {
                    let old_pred = classifier_map.iter()
                        .fold(0.0, |acc, (g, w)|
                            acc + *w * g.predict(&ex.data)
                        );
                    let new_pred = h.predict(&ex.data);

                    ex.label * (new_pred - old_pred)
                })
                .collect::<Vec<_>>();

            // Compute the difference between the new hypothesis
            // and the current combined hypothesis
            let diff = gap_vec.iter()
                .zip(self.dist.iter())
                .fold(0.0, |acc, (&v, &d)| acc + v * d);


            // Update the parameters
            if diff <= self.tolerance {
                println!("Break loop at: {t}");
                break;
            }

            // Update the weight on hypotheses
            self.update_clf_weight_mut(&mut classifier_map, h, gap_vec);
        }

        // Compute the dual optimal value for debug
        self.dual_objval_mut(&sample, &classifier_map);



        let weighted_classifier = classifier_map.into_iter()
            .filter_map(|(h, w)|
                if w != 0.0 { Some((w, h)) } else { None }
            )
            .collect::<Vec<_>>();


        CombinedClassifier {
            weighted_classifier
        }
    }
}
