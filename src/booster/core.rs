use super::super::base_learner::core::Classifier;
use super::super::base_learner::core::BaseLearner;

pub trait Booster {
    fn update_params(&mut self, h: Box<dyn Classifier>, examples: &[Vec<f64>], labels: &[f64]) -> Option<()>;

    fn run(&mut self, base_learner: Box<dyn BaseLearner>, sample: &[Vec<f64>], labels: &[f64], eps: f64);

    fn predict(&self, example: &[f64]) -> f64;

    fn predict_all(&self, examples: &[Vec<f64>]) -> Vec<f64> {
        examples.iter()
                .map(|example| self.predict(&example))
                .collect()
    }
}
