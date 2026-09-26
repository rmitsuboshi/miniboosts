use miniboosts_core::{
    Sample,
};
use std::iter;

pub trait RegressionTreeLoss {
    fn name(&self) -> &str;
    fn gradient(&self, predictions: &[f64], labels: &[f64]) -> Vec<f64>;
    fn diag_hessian(&self, predictions: &[f64], labels: &[f64])
        -> Vec<f64>;
}

pub struct L1Loss;

impl RegressionTreeLoss for L1Loss {
    fn name(&self) -> &str { "L1 loss" }
    fn gradient(&self, predictions: &[f64], labels: &[f64]) -> Vec<f64> {
        predictions.iter()
            .zip(labels)
            .map(|(y, p)| (y - p).abs())
            .collect()
    }

    fn diag_hessian(&self, predictions: &[f64], labels: &[f64])
        -> Vec<f64>
    {
        let n_examples = predictions.len();
        iter::repeat_n(0f64, n_examples).collect()
    }
}

pub struct L2Loss;

impl RegressionTreeLoss for L2Loss {
    fn name(&self) -> &str { "L2 loss" }
    fn gradient(&self, predictions: &[f64], labels: &[f64]) -> Vec<f64> {
        predictions.iter()
            .zip(labels)
            .map(|(p, y)| p - y)
            .collect()
    }

    fn diag_hessian(&self, predictions: &[f64], labels: &[f64])
        -> Vec<f64>
    {
        let n_examples = predictions.len();
        iter::repeat_n(1f64, n_examples).collect()
    }
}

