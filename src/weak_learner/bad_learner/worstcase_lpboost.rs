use crate::{
    Sample,
    WeakLearner,
};
use crate::common::utils;

use super::BadClassifier;


/// The worst-case weak leaerner for `LPBoost`.
pub struct BadBaseLearner {
    // The set of classifiers
    classifiers: Vec<BadClassifier>,
}


impl BadBaseLearner {
    /// Build a new instance of `self.`
    pub(super) fn new(n_sample: usize, tolerance: f64, nu: f64) -> Self {
        let half = (n_sample + 1) / 2;
        let tolerance = tolerance / (2f64 * n_sample as f64);
        let mut gap = 3f64;
        let mut classifiers = Vec::new();

        let shift = nu.ceil() as usize;

        let mut tag = 0;
        classifiers.push(BadClassifier::new(tag, half, shift, gap, tolerance));
        tag = 1;
        for k in 0..half-1 {
            let ix = half + k;
            gap += 2f64;
            classifiers.push(
                BadClassifier::new(tag, ix, shift, gap, tolerance)
            );
            assert!(
                tolerance * gap <= 1f64,
                "[BUG] \
                The tolerance parameter and \
                gap parameter is not set appropriately. \
                {tolerance} * {gap} > 1f64",
            );
        }
        tag = 2;
        classifiers.push(BadClassifier::new(tag, half, shift, gap, tolerance));

        Self { classifiers, }
    }
}


impl WeakLearner for BadBaseLearner {
    type Hypothesis = BadClassifier;


    fn name(&self) -> &str {
        "Bad Learner for LPBoost"
    }

    fn produce(&self, sample: &Sample, dist: &[f64])
        -> Self::Hypothesis
    {
        self.classifiers.iter()
            .map(|h| {
                let edge = utils::edge_of_hypothesis(sample, dist, h);
                (edge, h)
            })
            .max_by(|(e1, _), (e2, _)| e1.partial_cmp(e2).unwrap())
            .unwrap().1
            .clone()
    }
}

