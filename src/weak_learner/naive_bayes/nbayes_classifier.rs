use serde::{
    Serialize,
    Deserialize,
};

use crate::{Sample, Classifier};

use super::probability::Probability;


/// Naive Bayes classifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NBayesClassifier<P> {
    pub(super) conditionals: Vec<(f64, f64, P)>,
    pub(super) density: P,
}


impl<P> NBayesClassifier<P>
    where P: Probability
{
    /// Construct a new instance of `NBayesClassifier`
    /// from the given components.
    pub(super) fn from_components(
        conditionals: Vec<(f64, f64, P)>,

        density: P,
    ) -> Self
    {
        Self {
            conditionals,
            density
        }
    }
    /// Computes the logarithmic probability of the classes +/- 
    /// for the given instance.
    pub fn probabilities(&self, sample: &Sample, row: usize)
        -> Vec<(f64, f64)>
    {
        let log_all = self.density.log_probability(sample, row);

        self.conditionals.iter()
            .map(|(y, prior, density)| {
                let log_cond = density.log_probability(sample, row);
                let prob = prior * (log_cond - log_all).exp();
                (*y, prob)
            })
            .collect::<Vec<_>>()
    }

    // /// Computes the probability of the classes +/- 
    // /// for the given instance.
    // pub fn probabilities(&self, sample: &Sample, row: usize)
    //     -> Vec<(f64, f64)>
    // {
    //     self.log_probabilities(sample, row)
    //         .into_iter()
    //         .map(|(y, log_prob)| (y, log_prob.exp()))
    //         .collect::<Vec<_>>()
    // }
}


impl<P: Probability> Classifier for NBayesClassifier<P>
{
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {

        self.probabilities(sample, row).into_iter()
            .reduce(|a, b| if a.1 > b.1 { a } else { b })
            .expect("Faied to compare the probabilities")
            .0
    }
}


