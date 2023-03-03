//! This file defines split rules for decision tree.
use serde::*;

use crate::weak_learner::type_and_struct::*;
use crate::Sample;


/// The output of the function `split` of `SplitRule`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LR {
    Left,
    Right,
}


#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Splitter {
    pub(crate) feature: String,
    pub(crate) threshold: Threshold,
}


impl Splitter {
    #[inline]
    pub(crate) fn new(name: &str, threshold: Threshold) -> Self {
        let feature = name.to_string();
        Self {
            feature,
            threshold
        }
    }


    /// Defines the splitting.
    #[inline]
    pub fn split(&self, data: &Sample, row: usize) -> LR {
        let name = &self.feature;

        let value = data[name][row];

        if value < self.threshold.0 {
            LR::Left
        } else {
            LR::Right
        }
    }
}
