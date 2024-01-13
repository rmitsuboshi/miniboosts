use crate::Sample;


/// A trait that defines the behavor of classifier.
/// You only need to implement `confidence` method.
pub trait Classifier {
    /// Computes the confidence of the i'th row of the `df`.
    /// This code assumes that
    /// `Classifier::confidence` returns a value in `[-1.0, 1.0]`.
    /// Those hypotheses are called as **confidence-rated hypotheses**.
    fn confidence(&self, sample: &Sample, row: usize) -> f64;


    /// Predicts the label of the i'th row of the `df`.
    fn predict(&self, sample: &Sample, row: usize) -> i64 {
        let conf = self.confidence(sample, row);
        if conf >= 0.0 { 1 } else { -1 }
    }


    /// Computes the confidence of `df`.
    fn confidence_all(&self, sample: &Sample) -> Vec<f64> {
        let n_sample = sample.shape().0;
        (0..n_sample).map(|row| self.confidence(sample, row))
            .collect::<Vec<_>>()
    }


    /// Predicts the labels of `df`.
    fn predict_all(&self, sample: &Sample) -> Vec<i64>
    {
        let n_sample = sample.shape().0;
        (0..n_sample).map(|row| self.predict(sample, row))
            .collect::<Vec<_>>()
    }
}


/// A trait that defines the behavor of regressor.
/// You only need to implement `predict` method.
pub trait Regressor {
    /// Predicts the target value of the i'th row of the `df`.
    fn predict(&self, sample: &Sample, row: usize) -> f64;


    /// Predicts the labels of `df`.
    fn predict_all(&self, sample: &Sample) -> Vec<f64>
    {
        let n_sample = sample.shape().0;
        (0..n_sample).map(|row| self.predict(sample, row))
            .collect::<Vec<_>>()
    }
}



