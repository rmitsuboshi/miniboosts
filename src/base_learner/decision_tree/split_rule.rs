//! This file defines split rules for decision tree.
use polars::prelude::*;


use serde::*;


/// The output of the function `split` of `SplitRule`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LR {
    Left,
    Right,
}


/// Defines the splitting rules.
/// Currently, you can use the stump rule.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum SplitRule {
    /// If data[j] < threshold then go left and go right otherwise.
    Stump(StumpSplit),
}


impl SplitRule {
    pub(super) fn create_stump(name: &str, threshold: f64) -> Self {
        let feature = name.to_string();
        let stump = StumpSplit {
            feature,
            threshold,
        };
        Self::Stump(stump)
    }
}


/// Defines the split based on a feature.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct StumpSplit {
    feature: String,
    threshold: f64,
}




impl SplitRule {
    /// Defines the splitting.
    #[inline]
    pub fn split(&self, data: &DataFrame, row: usize) -> LR {
        match self {
            SplitRule::Stump(ref stump) => {
                let name = stump.feature.as_ref();
                let value = data[name]
                    .f64()
                    .expect("The target class is not a dtype f64")
                    .get(row).unwrap();

                if value < stump.threshold {
                    LR::Left
                } else {
                    LR::Right
                }
            },
        }
    }
}
