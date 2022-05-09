//! This file defines split rules for decision tree.
use crate::Data;


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
pub enum SplitRule<O> {
    /// If data[j] < threshold then go left and go right otherwise.
    Stump(StumpSplit<O>),
}


/// Defines the split based on a feature.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct StumpSplit<O> {
    index:     usize,
    threshold: O,
}


impl<O> From<(usize, O)> for StumpSplit<O> {
    #[inline]
    fn from((index, threshold): (usize, O)) -> Self {
        Self { index, threshold }
    }
}


impl<O> SplitRule<O>
    where O: PartialOrd,
{
    /// Defines the splitting.
    #[inline]
    pub fn split<D>(&self, data: &D) -> LR
        where D: Data<Output = O>
    {
        match self {
            SplitRule::Stump(ref stump) => {
                let value = data.value_at(stump.index);

                if value < stump.threshold {
                    LR::Left
                } else {
                    LR::Right
                }
            },
        }
    }
}


