//! This file defines some options of MLPBoost.

use polars::prelude::*;
use rayon::prelude::*;

use super::{
    utils::*,
    dist::*,
};
use crate::Classifier;


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
    pub(super) fn update<C>(&self,
                            eta: f64,
                            nu: f64,
                            data: &DataFrame,
                            target: &Series,
                            dist: &[f64],
                            position: usize,
                            classifiers: &[C],
                            mut weights: Vec<f64>,
                            iterate: usize)
        -> Vec<f64>
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
                let target = target.i64()
                    .expect("The target is not a dtype i64");


                let new_h: &C = &classifiers[position];

                let mut numer: f64 = 0.0;
                let mut denom: f64 = f64::MIN;

                target.into_iter()
                    .zip(dist)
                    .enumerate()
                    .for_each(|(i, (y, &d))| {
                        let y = y.unwrap() as f64;
                        let np = new_h.confidence(data, i);
                        let op = confidence(
                            i, data, classifiers, &weights[..]
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
                let size = data.shape().0;
                let f: &C = &classifiers[position];


                // base: -Aw
                // dir:  -A(ej - w)
                let mut base = vec![0.0; size];
                let mut dir  = base.clone();
                target.i64()
                    .expect("The target is not a dtype i64")
                    .into_iter()
                    .enumerate()
                    .for_each(|(i, y)| {
                        let y = y.unwrap() as f64;
                        let fc = f.confidence(data, i);
                        let wc = confidence(
                            i, data, classifiers, &weights[..]
                        );

                        dir[i]  = y * (wc - fc);
                        base[i] = -y * wc;
                    });


                let mut ub = 1.0;
                let mut lb = 0.0;


                // Check the step size `1`.
                let dist = dist_at(
                    eta, nu, data, target, classifiers, &dir[..]
                );


                let edge = edge_of(
                    data, target, &dist[..], classifiers, &dir[..]
                );


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
                        eta, nu, data, target, classifiers, &tmp[..]
                    );


                    let edge = edge_of(
                        data, target, &dist[..], classifiers, &dir[..]
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



