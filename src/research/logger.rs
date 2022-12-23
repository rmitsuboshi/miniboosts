use polars::prelude::*;


/// Defines some stats around boosting process.
pub trait Logger {
    /// Update the weights on hypotheses
    fn weights_on_hypotheses(&mut self, _data: &DataFrame, _target: &Series) {}


    /// Objective value at an intermediate state of boosting process
    fn objective_value(&self, data: &DataFrame, target: &Series) -> f64;


    /// Prediction at an intermediate state of boosting process
    fn prediction(&self, data: &DataFrame, index: usize) -> f64;


    /// Train/Test error at an intermediate state of boosting process
    fn loss<F>(
        &self,
        loss_function: &F, // Loss function
        data: &DataFrame, // Dataset to be predicted
        target: &Series,  // True target values
    ) -> f64
        where F: Fn(f64, f64) -> f64
    {
        let n_sample = data.shape().0 as f64;

        target.i64()
            .expect("The target class is not a dtype i64")
            .into_iter()
            .map(|y| y.unwrap() as f64)
            .enumerate()
            .map(|(i, y)| loss_function(y, self.prediction(data, i)))
            .sum::<f64>()
            / n_sample
    }
}
