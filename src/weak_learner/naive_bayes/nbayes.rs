use crate::{Sample, WeakLearner};
use crate::common::utils;


use super::probability::{
    Gaussian,
};


use super::nbayes_classifier::*;


/// A factory that produces a `GaussianNBClassifier`
/// for a given distribution over training examples.
/// The struct name comes from scikit-learn.
pub struct GaussianNB {}


impl<'a> GaussianNB {
    /// Initializes the GaussianNB instance.
    pub fn init() -> Self {
        Self {}
    }
}


impl WeakLearner for GaussianNB {
    type Hypothesis = NBayesClassifier<Gaussian>;

    fn produce(&self, sample: &Sample, dist: &[f64])
        -> Self::Hypothesis
    {

        let uniq = sample.unique_target();
        let target = sample.target();

        let mut conditionals = Vec::new();
        for y in uniq {
            let prior = utils::total_weight_for_label(y, target, dist)
                .clamp(0.0, 1.0);
            let (means, vars) = sample
                .weighted_mean_and_variance_for_label(y, dist)
                .into_iter()
                .unzip();
            let density = Gaussian::new(means, vars);
            conditionals.push((y, prior, density));
        }

        let (means, vars) = sample.weighted_mean_and_variance(dist)
            .into_iter()
            .unzip();
        let density = Gaussian::new(means, vars);


        NBayesClassifier::from_components(conditionals, density)
    }
}
