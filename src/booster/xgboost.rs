
use polars::prelude::*;
use rayon::prelude::*;


use crate::{Classifier, CombinedClassifier};
use crate::BaseLearner;
use crate::Booster;


pub struct XGBoost {
    tolerance: f64,

    size: usize,
}


impl XGBoost {
    pub fn init(df: &DataFrame) -> Self {
        let size = df.shape().0;

        Self {
            tolerance: 0.01,
            size,
        }
    }
}
