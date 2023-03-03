//! This file defines some options of MLPBoost.

use rayon::prelude::*;

use super::{
    utils::*,
    dist::*,
};
use crate::{Sample, Classifier};


/// Primary updates.
/// These options correspond to the Frank-Wolfe strategies.
#[derive(Clone, Copy)]
pub enum Primary {
    /// Classic step size, `2 / (t + 2)`.
    Classic,

    /// Short-step size,
    /// Adopt the step size that minimizes the strongly-smooth bound.
    ShortStep,

    /// Line-search step size,
    /// Adopt the best step size on the descent direction.
    LineSearch,

    // /// Pairwise strategy, 
    // /// See [this paper](https://arxiv.org/abs/1511.05932). 
    // Pairwise,
}


impl Primary {
    /// Returns a weight vector updated 
    /// by the Frank-Wolfe rule.
    #[inline(always)]
    pub(super) fn update<C>(
        &self,
        eta: f64,
        nu: f64,
        sample: &Sample,
        dist: &[f64],
        position: usize,
        classifiers: &[C],
        mut weights: Vec<f64>,
        iterate: usize
    ) -> Vec<f64>
        where C: Classifier
    {
        match self {
            Primary::Classic => {
                // Compute the step-size
                let lambda = 2.0_f64 / ((iterate + 1) as f64);

                // Update the weights
                weights.iter_mut()
                    .enumerate()
                    .for_each(|(i, w)| {
                        let e = if i == position { 1.0 } else { 0.0 };
                        *w = lambda * e + (1.0 - lambda) * *w;
                    });
                weights
            },


            Primary::ShortStep => {
                // Compute the step-size
                if classifiers.len() == 1 {
                    return vec![1.0];
                }


                let new_h: &C = &classifiers[position];

                let mut numer: f64 = 0.0;
                let mut denom: f64 = f64::MIN;

                sample.target()
                    .into_iter()
                    .zip(dist)
                    .enumerate()
                    .for_each(|(i, (y, &d))| {
                        let np = new_h.confidence(sample, i);
                        let op = confidence(
                            i, sample, classifiers, &weights[..]
                        );

                        let gap = y * (np - op);
                        numer += d * gap;
                        denom = denom.max(gap.abs());
                    });

                let step = numer / (eta * denom.powi(2));

                let lambda = (step.max(0.0_f64)).min(1.0_f64);


                // Update the weights
                weights.iter_mut()
                    .enumerate()
                    .for_each(|(i, w)| {
                        let e = if position == i { 1.0 } else { 0.0 };
                        *w = lambda * e + (1.0 - lambda) * *w;
                    });
                weights
            },


            Primary::LineSearch => {
                let n_sample = sample.shape().0;
                let f: &C = &classifiers[position];


                // base: -Aw
                // dir:  -A(ej - w)
                let mut base = vec![0.0; n_sample];
                let mut dir  = base.clone();
                sample.target()
                    .into_iter()
                    .enumerate()
                    .for_each(|(i, y)| {
                        let fc = f.confidence(sample, i);
                        let wc = confidence(
                            i, sample, classifiers, &weights[..]
                        );

                        dir[i]  = y * (wc - fc);
                        base[i] = -y * wc;
                    });


                let mut ub = 1.0;
                let mut lb = 0.0;


                // Check the step n_sample `1`.
                let dist = dist_at(eta, nu, sample, classifiers, &dir[..]);


                let edge = edge_of(sample, &dist[..], classifiers, &dir[..]);


                if edge >= 0.0 {
                    weights.par_iter_mut()
                        .enumerate()
                        .for_each(|(i, w)| {
                            *w = if position == i { 1.0 } else { 0.0 };
                        });
                    return weights;
                }

                const SUB_TOLERANCE: f64 = 1e-9;


                while ub - lb > SUB_TOLERANCE {
                    let stepsize = (lb + ub) / 2.0;

                    let tmp = base.iter()
                        .zip(dir.iter())
                        .map(|(b, d)| b + stepsize * d)
                        .collect::<Vec<f64>>();


                    let dist = dist_at(
                        eta, nu, sample, classifiers, &tmp[..]
                    );


                    // Compute the gradient for the direction `dir`.
                    let edge = edge_of(
                        sample, &dist[..], classifiers, &dir[..]
                    );

                    if edge > 0.0 {
                        lb = stepsize;
                    } else if edge < 0.0 {
                        ub = stepsize;
                    } else {
                        break;
                    }
                }

                let stepsize = (lb + ub) / 2.0;

                // Update the weights


                weights.iter_mut()
                    .enumerate()
                    .for_each(|(i, w)| {
                        let e = if position == i { 1.0 } else { 0.0 };
                        *w = stepsize * e + (1.0 - stepsize) * *w;
                    });
                weights
            },
            // Primary::Pairwise => {
            // },
        }
    }
}


/// Secondary updates.
/// You can choose the heuristic updates from these options.
#[derive(Clone, Copy)]
pub enum Secondary {
    /// LPBoost update.
    LPB,
    // /// ERLPBoost update.
    // ERLPB,
    // /// No heuristic update.
    // Nothing,
}


/// The stopping criterion of the algorithm.
/// The default criterion uses `ObjVal`.
#[derive(Clone, Copy)]
pub enum StopCondition {
    /// Uses the edge of current combined hypothesis.
    Edge,

    /// Uses the objective value.
    ObjVal,
}



