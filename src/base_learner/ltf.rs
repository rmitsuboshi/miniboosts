/// Linear threshold function class.
use crate::data_type::{DType, Data, Label, Sample};
use crate::base_learner::core::BaseLearner;
use crate::base_learner::core::Classifier;


#[derive(PartialEq)]
pub struct LTFClassifier {
    normal_vector: Vec<f64>,
    threshold: f64,
}

impl Classifier<f64, f64> for LTFClassifier {
    fn predict(&self, data: &Data<f64>) -> Label<f64> {
        let dim = self.normal_vector.len();

        let mut val = 0.0;
        for i in 0..dim {
            val += self.normal_vector[i] * data.value_at(i);
        }

        if val > self.threshold {
            1.0
        } else {
            -1.0
        }
    }
}
