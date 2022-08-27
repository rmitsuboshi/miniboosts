//! This file defines `CERLPBoost` based on the paper
//! "On the Equivalence of Weak Learnability and Linaer Separability:
//!     New Relaxations and Efficient Boosting Algorithms"
//! by Shai Shalev-Shwartz and Yoram Singer.
//! I named this algorithm `CERLPBoost`
//! since it is referred as `the Corrective version of CERLPBoost`
//! in "Entropy Regularized LPBoost" by Warmuth et al.
//! 
use polars::prelude::*;
use rayon::prelude::*;


use crate::{Classifier, CombinedClassifier};
use crate::BaseLearner;
use crate::Booster;



/// Corrective ERLPBoost struct.
/// This algorithm is based on the [paper](https://link.springer.com/content/pdf/10.1007/s10994-010-5173-z.pdf).
pub struct CERLPBoost {
    dist: Vec<f64>,
    // A regularization parameter defined in the paper
    eta: f64,

    tolerance: f64,
    capping_param: f64,

    // Optimal value (Dual problem)
    dual_optval: f64,
}


impl CERLPBoost {
    /// Initialize the `CERLPBoost`.
    pub fn init(df: &DataFrame) -> Self {
        assert!(!df.is_empty());
        let (m, _) = df.shape();


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
            dual_optval: 1.0,
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
                          data: &DataFrame,
                          target: &Series,
                          classifiers: &[(C, f64)])
        where C: Classifier + PartialEq,
    {
        self.dual_optval = classifiers.iter()
            .map(|(h, _)|
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
        let m = self.dist.len() as f64;
        let ln_part = (m / self.capping_param).ln();
        self.eta = ln_part / self.tolerance;
    }



    /// returns the maximum iteration of the CERLPBoost
    /// to find a combined hypothesis that has error at most `tolerance`.
    pub fn max_loop(&mut self, tolerance: f64) -> u64 {
        if 2.0 * self.tolerance != tolerance {
            self.set_tolerance(tolerance);
        }

        let m = self.dist.len() as f64;

        let ln_m = (m / self.capping_param).ln();
        let max_iter = 8.0 * ln_m / self.tolerance.powi(2);

        max_iter.ceil() as u64
    }




    /// Updates weight on hypotheses and `self.dist` in this order.
    fn update_distribution_mut<C>(&mut self,
                                  classifiers: &[(C, f64)],
                                  data: &DataFrame,
                                  target: &Series)
        where C: Classifier + PartialEq,
    {
        self.dist.iter_mut()
            .zip(
                target.i64()
                    .expect("The target is not a dtype i64")
            )
            .enumerate()
            .for_each(|(i, (d, y))| {
                let p = prediction(i, data, classifiers);
                *d = - self.eta * y.unwrap() as f64 * p
            });


        let m = self.dist.len();
        // Sort the indices over `self.dist` in non-increasing order.
        let mut indices = (0..m).collect::<Vec<_>>();
        indices.sort_by(|&i, &j|
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
            .rev();


        let ub = 1.0 / self.capping_param;
        let log_cap = self.capping_param.ln();

        let mut idx_with_logsum = indices.into_iter()
            .zip(logsums)
            .enumerate();

        while let Some((i, (i_sorted, logsum))) = idx_with_logsum.next() {
            let log_xi = (1.0 - ub * i as f64).ln() - logsum;
            // TODO replace this line into `get_unchecked`
            let d = self.dist[i_sorted];

            // Stopping criterion of this while loop
            if log_xi + d + log_cap <= 0.0 {
                self.dist[i_sorted] = (log_xi + d).exp();
                while let Some((_, (ii, _))) = idx_with_logsum.next() {
                    self.dist[ii] = (log_xi + self.dist[ii]).exp();
                }
                break;
            }

            self.dist[i_sorted] = ub;
        }
    }


    /// Update the weights on hypotheses
    fn update_clf_weight_mut<C>(&self,
                                clfs: &mut Vec<(C, f64)>,
                                new_clf: C,
                                gap_vec: Vec<f64>)
        where C: Classifier + PartialEq,
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


        let mut already_exist = false;
        for (clf, w) in clfs.iter_mut() {
            if *clf == new_clf {
                already_exist = true;
                *w += weight;
            } else {
                *w *= 1.0 - weight;
            }
        }

        if !already_exist {
            clfs.push((new_clf, weight));
        }
    }
}


impl<C> Booster<C> for CERLPBoost
    where C: Classifier + PartialEq + std::fmt::Debug
{


    fn run<B>(&mut self,
              base_learner: &B,
              data: &DataFrame,
              target: &Series,
              tolerance: f64)
        -> CombinedClassifier<C>
        where B: BaseLearner<Clf = C>,
    {
        let max_iter = self.max_loop(tolerance);


        let mut classifiers: Vec<(C, f64)> = Vec::new();


        // {
        //     let h = base_learner.produce(data, target, &self.dist);
        //     classifiers.push((h, 1.0));
        // }


        for t in 1..=max_iter {
            // Update the distribution over examples
            self.update_distribution_mut(&classifiers, data, target);


            // Receive a hypothesis from the base learner
            let h = base_learner.produce(data, target, &self.dist);

            // println!("h: {h:?}");


            let gap_vec = target.i64()
                .expect("The target class is not a dtype of i64")
                .into_iter()
                .enumerate()
                .map(|(i, y)| {
                    let old_pred = prediction(i, data, &classifiers[..]);
                    let new_pred = h.confidence(data, i);

                    y.unwrap() as f64 * (new_pred - old_pred)
                })
                .collect::<Vec<_>>();


            // Compute the difference between the new hypothesis
            // and the current combined hypothesis
            let diff = gap_vec.par_iter()
                .zip(&self.dist[..])
                .map(|(v, d)| v * d)
                .sum::<f64>();


            // Update the parameters
            if diff <= self.tolerance {
                println!("Break loop at: {t}");
                break;
            }

            // Update the weight on hypotheses
            self.update_clf_weight_mut(&mut classifiers, h, gap_vec);
        }

        // Compute the dual optimal value for debug
        self.dual_objval_mut(data, target, &classifiers[..]);


        let weighted_classifier = classifiers.into_iter()
            .filter_map(|(h, w)|
                if w != 0.0 { Some((w, h)) } else { None }
            )
            .collect::<Vec<_>>();


        CombinedClassifier::from(weighted_classifier)
    }
}


fn prediction<C>(i: usize, data: &DataFrame, classifiers: &[(C, f64)])
    -> f64
    where C: Classifier
{
    classifiers.iter()
        .map(|(h, w)| w * h.confidence(data, i))
        .sum()
}
