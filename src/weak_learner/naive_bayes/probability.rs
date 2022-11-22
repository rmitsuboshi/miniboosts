use polars::prelude::*;
use rayon::prelude::*;
use serde::{
    Serialize,
    Deserialize,
};

use core::f64::consts::PI;


pub trait Probability {
    fn log_probability(&self, data: &DataFrame, row: usize) -> f64;

    fn probability(&self, data: &DataFrame, row: usize) -> f64 {
        self.log_probability(data, row).exp()
    }
}


/// Gaussian density
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Gaussian {
    pub(super) means: Vec<f64>,
    pub(super) vars: Vec<f64>,
}

impl Gaussian {
    pub(super) fn new(means: Vec<f64>, vars: Vec<f64>) -> Self {
        assert_eq!(means.len(), vars.len());
        Self { means, vars }
    }
}


impl Probability for Gaussian {
    #[inline(always)]
    fn log_probability(&self, data: &DataFrame, row: usize) -> f64 {
        let n_features = self.means.len() as f64;

        let gauss_const: f64 = n_features * (2.0_f64 * PI).ln();

        let non_const = self.means.par_iter()
            .zip(&self.vars[..])
            .zip(data.get_columns())
            .map(|((&mean, &var), column)| {
                let column = column.f64()
                    .expect("The data is not a dtype f64");
                let x = column.get(row).unwrap();

                ((x - mean).powi(2) / var) + var.ln()
            })
            .sum::<f64>();

        - 0.5 * (gauss_const + non_const)
    }
}
