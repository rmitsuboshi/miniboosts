//! This file defines split rules for decision tree.
use polars::prelude::*;


use serde::*;


/// The output of the function `split` of `SplitRule`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LR {
    Left,
    Right,
}


#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct Splitter {
    feature: String,
    threshold: f64,
}


impl Splitter {
    #[inline]
    pub(super) fn new(name: &str, threshold: f64) -> Self {
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

        if value < self.threshold {
            LR::Left
        } else {
            LR::Right
        }
    }
}
