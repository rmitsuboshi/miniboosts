use polars::prelude::*;
use rayon::prelude::*;

use crate::WeakLearner;

use super::probability::{
    Gaussian,
};
use super::nbayes_classifier::*;


/// A factory that produces a `GaussianNBClassifier`
/// for a given distribution over training examples.
/// The struct name comes from scikit-learn.
pub struct GaussianNB;


impl GaussianNB {
    /// Initializes the GaussianNB instance.
    pub fn init(_data: &DataFrame, _target: &Series) -> Self {
        Self {}
    }
}


impl WeakLearner for GaussianNB {
    type Hypothesis = NBayesClassifier<Gaussian>;

    fn produce(&self, data: &DataFrame, target: &Series, dist: &[f64])
        -> Self::Hypothesis
    {
        let mut prior_p: f64 = 0.0;
        let mut prior_n: f64 = 0.0;
        target.i64()
            .expect("The target class is not a dtype i64")
            .into_iter()
            .zip(dist)
            .for_each(|(y, d)| {
                let y = y.unwrap();
                if y > 0 {
                    prior_p += d;
                } else {
                    prior_n += d;
                }
            });

        assert!((prior_p + prior_n - 1.0_f64).abs() < 1e-9);


        // Compute the mean/variance for each feature over all instances
        let (means, vars) = overall_mean_var(data, dist);


        let density = Gaussian::new(means, vars);


        // Compute the mean/variance for each feature 
        // over positive/negative instances
        let means_p = data.get_columns()
            .into_par_iter()
            .map(|column|
                mean_for_a_feature2(column, target, dist, 1, prior_p)
            )
            .collect::<Vec<f64>>();
        let vars_p = data.get_columns()
            .into_par_iter()
            .zip(&means_p[..])
            .map(|(column, &mu)|
                variance_for_a_feature2(column, target, dist, 1, prior_p, mu)
            )
            .collect::<Vec<f64>>();
        assert!(vars_p.iter().all(|v| *v != 0.0));
        let cond_density_p = Gaussian::new(means_p, vars_p);


        let means_n = data.get_columns()
            .into_par_iter()
            .map(|column|
                mean_for_a_feature2(column, target, dist, -1, prior_n)
            )
            .collect::<Vec<f64>>();
        let vars_n = data.get_columns()
            .into_par_iter()
            .zip(&means_n[..])
            .map(|(column, &mu)|
                variance_for_a_feature2(column, target, dist, -1, prior_n, mu)
            )
            .collect::<Vec<f64>>();


        assert!(vars_n.iter().all(|v| *v != 0.0));
        let cond_density_n = Gaussian::new(means_n, vars_n);


        NBayesClassifier {
            prior_p,
            prior_n,


            cond_density_p,
            cond_density_n,

            density,
        }
    }
}


/// Compute the means and variances for all feature over all examples.
pub(self) fn overall_mean_var(
    data: &DataFrame,
    dist: &[f64]
) -> (Vec<f64>, Vec<f64>)
{
    let means = data.get_columns()
        .into_par_iter()
        .map(|col| mean_for_a_feature(col, dist))
        .collect::<Vec<f64>>();


    let vars = data.get_columns()
        .into_par_iter()
        .zip(&means[..])
        .map(|(col, mean)| variance_for_a_feature(col, dist, *mean))
        .collect::<Vec<f64>>();
    (means, vars)
}


/// Compute the mean for the given feature `data`.
pub(self) fn mean_for_a_feature(data: &Series, dist: &[f64]) -> f64
{
    let data = data.f64()
        .expect("The target class is not a dtype i64");


    data.into_iter()
        .zip(dist)
        .map(|(x, d)| {
            let x = x.unwrap() as f64;
            *d * x
        })
        .sum::<f64>()
}


/// Compute the variance for the given feature `data`.
pub(self) fn variance_for_a_feature(data: &Series, dist: &[f64], mean: f64)
    -> f64
{

    let data = data.f64()
        .expect("The target class is not a dtype i64");


    data.into_iter()
        .zip(dist)
        .map(|(x, d)| {
            let x = x.unwrap() as f64;
            *d * (x - mean).powi(2)
        })
        .sum::<f64>()
}


/// Compute the mean for the given feature `data`.
pub(self) fn mean_for_a_feature2(
    data: &Series,
    target: &Series,
    dist: &[f64],
    label: i64,
    prior: f64
) -> f64
{
    let data = data.f64()
        .expect("The target class is not a dtype i64");
    let target = target.i64()
        .expect("The target class is not a dtype i64");


    data.into_iter()
        .zip(target)
        .zip(dist)
        .filter_map(|((x, y), d)| {
            let y = y.unwrap() as i64;
            if y == label {
                let x = x.unwrap() as f64;
                Some(x * *d)
            } else {
                None
            }
        })
        .sum::<f64>()
        / prior
}


/// Compute the variance for the given feature `data`.
pub(self) fn variance_for_a_feature2(
    data: &Series,
    target: &Series,
    dist: &[f64],
    label: i64,
    prior: f64,
    mean: f64
) -> f64
{

    let data = data.f64()
        .expect("The target class is not a dtype i64");
    let target = target.i64()
        .expect("The target class is not a dtype i64");


    data.into_iter()
        .zip(target)
        .zip(dist)
        .filter_map(|((x, y), d)| {
            let y = y.unwrap() as i64;
            if y == label {
                let x = x.unwrap() as f64;
                Some((x - mean).powi(2) * *d)
            } else {
                None
            }
        })
        .sum::<f64>()
        / prior
}


