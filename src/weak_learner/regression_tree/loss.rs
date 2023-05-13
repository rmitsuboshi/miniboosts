//! Defines some criterions for regression tree.
use rayon::prelude::*;
use serde::{
    Serialize,
    Deserialize
};

use std::fmt;
use crate::Sample;
use crate::weak_learner::common::type_and_struct::*;

use super::bin::*;

use std::collections::HashMap;

/// The type of loss (error) function.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum LossType {
    /// Least Absolute Error
    L1,


    /// Least Squared Error
    L2,


    /// Huber loss with parameter `delta`.
    /// Huber loss maps the given scalar `z` to
    /// `0.5 * z.powi(2)` if `z.abs() < delta`,
    /// `delta * (z.abs() - 0.5 * delta)`, otherwise.
    Huber(f64),


    // /// Quantile loss
    // Quantile(f64),
}


impl fmt::Display for LossType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let loss = match self {
            Self::L1 => "L1 (Least Absolute) loss".to_string(),
            Self::L2 => "L2 (Least  Squared) loss".to_string(),
            Self::Huber(delta) => format!("Huber loss (delta = {delta})"),
        };
        write!(f, "{loss}")
    }
}


impl LossType {
    pub(crate) fn gradient_and_hessian(
        &self,
        targets: &[f64],
        predictions: &[f64],
    ) -> Vec<GradientHessian>
    {
        match self {
            Self::L1 => {
                targets.iter()
                    .zip(predictions)
                    .map(|(y, p)| {
                        let grad = (y - p).signum();
                        GradientHessian::new(grad, 0.0)
                    })
                    .collect::<Vec<_>>()
            },
            Self::L2 => {
                targets.iter()
                    .zip(predictions)
                    .map(|(y, p)| {
                        let grad = p - y;
                        GradientHessian::new(grad, 1.0)
                    })
                    .collect::<Vec<_>>()
            },

            Self::Huber(delta) => {
                targets.iter()
                    .zip(predictions)
                    .map(|(y, p)| {
                        let diff = p - y;
                        let (grad, hess) = if diff.abs() < *delta {
                            (diff, 1.0)
                        } else {
                            (delta * diff.signum(), 0.0)
                        };

                        GradientHessian::new(grad, hess)
                    })
                    .collect::<Vec<_>>()
            },
        }
    }


    /// Returns the best splitting rule based on the loss function.
    pub(super) fn best_split<'a>(
        &self,
        bins_map: &HashMap<&'_ str, Bins>,
        sample: &'a Sample,
        gh: &[GradientHessian],
        idx: &[usize],
        lambda_l2: f64,
    ) -> (&'a str, Threshold)
    {

        sample.features()
            .par_iter()
            .map(|feature| {
                let name = feature.name();
                let bin = bins_map.get(name).unwrap();
                let pack = bin.pack(idx, feature, gh);
                let (score, threshold) = self.best_split_at(pack, lambda_l2);

                (score, name, threshold)
            })
            .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
            .map(|(_, name, threshold)| (name, threshold))
            .expect("No feature that maximizes the score.")
    }


    fn best_split_at(
        &self,
        pack: Vec<(Bin, GradientHessian)>,
        lambda_l2: f64,
    ) -> (LossValue, Threshold)
    {
        let mut right_grad_sum = pack.par_iter()
            .map(|(_, gh)| gh.grad)
            .sum::<f64>();
        let mut right_hess_sum = pack.par_iter()
            .map(|(_, gh)| gh.hess)
            .sum::<f64>();


        let mut left_grad_sum = 0.0;
        let mut left_hess_sum = 0.0;


        let mut best_score = f64::MIN;
        let mut best_threshold = f64::MIN;


        for (bin, gh) in pack {
            left_grad_sum  += gh.grad;
            left_hess_sum  += gh.hess;
            right_grad_sum -= gh.grad;
            right_hess_sum -= gh.hess;


            let score = 
                left_grad_sum.powi(2) / (left_hess_sum + lambda_l2)
                + right_grad_sum.powi(2) / (right_hess_sum + lambda_l2);
            if best_score < score {
                best_score = score;
                best_threshold = bin.0.end;
            }
        }

        (best_score.into(), best_threshold.into())
    }


    pub(super) fn prediction_and_loss(
        &self,
        indices: &[usize],
        gh: &[GradientHessian],
        lambda_l2: f64,
    ) -> (Prediction<f64>, LossValue)
    {
        let grad_sum = indices.par_iter()
            .map(|&i| gh[i].grad)
            .sum::<f64>();

        let hess_sum = indices.par_iter()
            .map(|&i| gh[i].hess)
            .sum::<f64>();

        let prediction = - grad_sum / (hess_sum + lambda_l2);
        let loss_value = -0.5 * grad_sum.powi(2) / (hess_sum + lambda_l2);

        (prediction.into(), loss_value.into())
    }

}


