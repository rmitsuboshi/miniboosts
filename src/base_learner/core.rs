
pub trait Classifier {
    fn predict(&self, example: &[f64]) -> f64;


    fn predict_all(&self, examples: &[Vec<f64>]) -> Vec<f64> {
        examples.iter()
                .map(|example| self.predict(&example))
                .collect()
    }
}


pub trait BaseLearner {
    fn best_hypothesis(&self, sample: &[Vec<f64>], labels: &[f64], distribution: &[f64]) -> Box<dyn Classifier>;
}

