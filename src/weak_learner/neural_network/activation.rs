use crate::common::utils;

/// Activation functions available to neural networks.
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    /// Soft Max function
    SoftMax(f64),
    /// Sigmoid function
    Sigmoid(f64),
    /// ReLU function.
    ReLu(f64),
    /// Identity function
    Id,
}


impl Activation {
    pub(crate) fn eval<T: AsRef<[f64]>>(&self, x: T) -> Vec<f64> {
        match self {
            Self::SoftMax(eta) => softmax(*eta, x),
            Self::Sigmoid(eta) => sigmoid(*eta, x),
            Self::ReLu(threshold) => relu(*threshold, x),
            Self::Id => id(x),
        }
    }

    pub(crate) fn diff<T: AsRef<[f64]>>(&self, x: T) -> Vec<f64> {
        match self {
            Self::SoftMax(eta) => softmax_diff(*eta, x),
            Self::Sigmoid(eta) => sigmoid_diff(*eta, x),
            Self::ReLu(threshold) => relu_diff(*threshold, x),
            Self::Id => id_diff(x),
        }
    }
}


#[inline]
fn softmax<T: AsRef<[f64]>>(eta: f64, x: T) -> Vec<f64> {
    let x = x.as_ref();
    let iter = x.into_iter().map(|xi| eta * xi);

    utils::project_log_distribution_to_capped_simplex(1.0, iter)
}


#[inline]
fn sigmoid<T: AsRef<[f64]>>(eta: f64, x: T) -> Vec<f64> {
    let x = x.as_ref();
    x.into_iter()
        .map(|xi| 1.0 / (1.0 + (-eta * xi).exp()))
        .collect()
}


#[inline]
fn relu<T: AsRef<[f64]>>(threshold: f64, x: T) -> Vec<f64> {
    let x = x.as_ref();
    x.into_iter()
        .map(|xi| if *xi <= threshold { 0.0 } else { *xi })
        .collect()
}


#[inline]
fn id<T: AsRef<[f64]>>(x: T) -> Vec<f64> {
    x.as_ref().to_vec()
}


#[inline]
fn softmax_diff<T: AsRef<[f64]>>(eta: f64, x: T) -> Vec<f64> {
    let x = x.as_ref();
    let probabilities = softmax(eta, x);

    probabilities.into_iter()
        .map(|p| eta * p * (1.0 - p))
        .collect()
}


#[inline]
fn sigmoid_diff<T: AsRef<[f64]>>(eta: f64, x: T) -> Vec<f64> {
    let x = x.as_ref();
    x.into_iter()
        .map(|xi| {
            let e = (-eta * xi).exp();
            eta * e / (1.0 + e)
        })
        .collect()
}


#[inline]
fn relu_diff<T: AsRef<[f64]>>(threshold: f64, x: T) -> Vec<f64> {
    let x = x.as_ref();
    x.into_iter()
        .map(|xi| if *xi < threshold { 0.0 } else { 1.0 })
        .collect()
}


#[inline]
fn id_diff<T: AsRef<[f64]>>(x: T) -> Vec<f64> {
    let x = x.as_ref();
    let len = x.len();
    vec![1.0; len]
}
