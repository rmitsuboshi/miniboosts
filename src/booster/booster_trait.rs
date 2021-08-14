pub trait Booster {
    pub update_params(&mut self, h: Box<dyn Fn(&[f64]) -> f64>, examples: &[Vec<f64>], labels: &[f64]);

    pub fn predict(&self, example: &[f64]) -> f64;
    pub fn predict_all(&self, examples: &[Vec<f64>]) -> Vec<f64>;
}
