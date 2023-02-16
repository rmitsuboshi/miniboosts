//! This file defines split rules for decision tree.
use polars::prelude::*;
use serde::*;

use crate::weak_learner::type_and_struct::*;


/// The output of the function `split` of `SplitRule`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LR {
    Left,
    Right,
}


#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct Splitter {
    pub(super) feature: String,
    pub(super) threshold: Threshold,
}


impl Splitter {
    #[inline]
    pub(super) fn new(name: &str, threshold: Threshold) -> Self {
        let feature = name.to_string();
        Self {
            feature,
            threshold
        }
    }


    /// Defines the splitting.
    #[inline]
    pub fn split(&self, data: &DataFrame, row: usize) -> LR {
        let name = self.feature.as_ref();
        let value = data[name]
            .f64()
            .expect("The target class is not a dtype f64")
            .get(row).unwrap();

        if value < self.threshold.0 {
            LR::Left
        } else {
            LR::Right
        }
    }
}
