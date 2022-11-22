use polars::prelude::*;

use serde::{
    Serialize,
    Deserialize,
};

use crate::Classifier;

use super::probability::Probability;


/// Naive Bayes classifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NBayesClassifier<P> {
    pub(super) prior_p: f64,
    pub(super) prior_n: f64, // equals to `1.0 - prior_p`


    pub(super) cond_density_p: P,
    pub(super) cond_density_n: P,
    pub(super) density: P,
}


impl<P> NBayesClassifier<P>
    where P: Probability
{
    /// Computes the logarithmic probability of the classes +/- 
    /// for the given instance.
    pub fn log_probabilities(&self, data: &DataFrame, row: usize)
        -> (f64, f64)
    {
        let ln_cond_p = self.cond_density_p.log_probability(data, row);
        let ln_cond_n = self.cond_density_n.log_probability(data, row);
        let ln_all    = self.density.log_probability(data, row);


        let p_prob = self.prior_p * (ln_cond_p - ln_all).exp();
        let n_prob = self.prior_n * (ln_cond_n - ln_all).exp();


        (p_prob, n_prob)
    }

    /// Computes the probability of the classes +/- 
    /// for the given instance.
    pub fn probabilities(&self, data: &DataFrame, row: usize) -> (f64, f64)
    {
        let (ln_p, ln_n) = self.log_probabilities(data, row);
        (ln_p.exp(), ln_n.exp())
    }
}


impl<P: Probability> Classifier for NBayesClassifier<P>
{
    fn confidence(&self, data: &DataFrame, row: usize) -> f64 {

        let (p, n) = self.probabilities(data, row);

        if p >= n { 1.0 } else { -1.0 }
    }
}


