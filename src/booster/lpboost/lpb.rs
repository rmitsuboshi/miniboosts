//! This file defines `LPBoost` based on the paper
//! "Boosting algorithms for Maximizing the Soft Margin"
//! by Warmuth et al.
//! 
use polars::prelude::*;
// use rayon::prelude::*;

use super::lp_model::LPModel;

use crate::{Classifier, CombinedClassifier};
use crate::BaseLearner;
use crate::Booster;


use std::cell::RefCell;



/// LPBoost struct. See [this paper](https://proceedings.neurips.cc/paper/2007/file/cfbce4c1d7c425baf21d6b6f2babe6be-Paper.pdf).
pub struct LPBoost {
    // Distribution over examples
    dist: Vec<f64>,

    // min-max edge of the new hypothesis
    gamma_hat: f64,

    // Tolerance parameter
    tolerance: f64,


    // Number of examples
    size: usize,


    // Capping parameter
    nu: f64,


    // GRBModel.
    lp_model: Option<RefCell<LPModel>>,


    terminated: usize,
}


impl LPBoost {
    /// Initialize the `LPBoost`.
    pub fn init(df: &DataFrame) -> Self {
        let (size, _) = df.shape();
        assert!(size != 0);


        let uni = 1.0 / size as f64;
        LPBoost {
            dist:      vec![uni; size],
            gamma_hat: 1.0,
            tolerance: uni,
            size,
            nu:        1.0,
            lp_model: None,

            terminated: 0_usize,
        }
    }


    /// This method updates the capping parameter.
    /// This parameter must be in `[1, sample_size]`.
    pub fn nu(mut self, nu: f64) -> Self {
        assert!(1.0 <= nu && nu <= self.size as f64);
        self.nu = nu;

        self
    }


    fn init_solver(&mut self) {
        let upper_bound = 1.0 / self.nu;

        assert!((0.0..=1.0).contains(&upper_bound));

        let lp_model = RefCell::new(LPModel::init(self.size, upper_bound));

        self.lp_model = Some(lp_model);
    }


    /// Set the tolerance parameter.
    /// Default is `1.0 / sample_size`/
    #[inline(always)]
    fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
    }


    /// Returns the terminated iteration.
    /// This method returns `0` before the boosting step.
    #[inline(always)]
    pub fn terminated(&self) -> usize {
        self.terminated
    }


    /// This method updates `self.dist` and `self.gamma_hat`
    /// by solving a linear program
    /// over the hypotheses obtained in past steps.
    #[inline(always)]
    fn update_distribution_mut<C>(&self,
                                  data: &DataFrame,
                                  target: &Series,
                                  h: &C)
        -> f64
        where C: Classifier
    {
        self.lp_model.as_ref()
            .unwrap()
            .borrow_mut()
            .update(data, target, h)
    }
}


impl<C> Booster<C> for LPBoost
    where C: Classifier,
{
    fn run<B>(&mut self,
              base_learner: &B,
              data: &DataFrame,
              target: &Series,
              tolerance: f64)
        -> CombinedClassifier<C>
        where B: BaseLearner<Clf = C>,
    {
        self.set_tolerance(tolerance);

        self.init_solver();

        let mut classifiers = Vec::new();

        self.terminated = usize::MAX;

        // Since the LPBoost does not have non-trivial iteration,
        // we run this until the stopping criterion is satisfied.
        loop {
            let h = base_learner.produce(data, target, &self.dist);

            // Each element in `margins` is the product of
            // the predicted vector and the correct vector

            let ghat = target.i64()
                .expect("The target class is not a dtype of i64")
                .into_iter()
                .enumerate()
                .map(|(i, y)| y.unwrap() as f64 * h.confidence(data, i))
                .zip(self.dist.iter())
                .map(|(yh, &d)| d * yh)
                .sum::<f64>();

            self.gamma_hat = ghat.min(self.gamma_hat);


            let gamma_star = self.update_distribution_mut(
                data, target, &h
            );


            classifiers.push(h);

            if gamma_star >= self.gamma_hat - self.tolerance {
                println!("Break loop at: {t}", t = classifiers.len());
                self.terminated = classifiers.len();
                break;
            }

            // Update the distribution over the training examples.
            self.dist = self.lp_model.as_ref()
                .unwrap()
                .borrow()
                .distribution();
        }


        let clfs = self.lp_model.as_ref()
            .unwrap()
            .borrow()
            .weight()
            .zip(classifiers)
            .filter(|(w, _)| *w != 0.0)
            .collect::<Vec<(f64, C)>>();


        CombinedClassifier::from(clfs)
    }
}


