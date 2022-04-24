//! This file defines split rules for decision tree.
use crate::Data;


/// The output of the function `split` of `SplitRule`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LR {
    Left,
    Right,
}


/// A trait that defines the splitting rule
/// in the decision tree.
pub trait SplitRule<D> {
    /// Returns `LR::Left` if `data` descends to the left child,
    /// `LR::Right` otherwise.
    fn split(&self, data: &D) -> LR;
}


/// Defines the split based on a feature.
#[derive(Debug)]
pub struct StumpSplit<D, O>
    where D: Data<Output = O>,
          O: PartialOrd
{
    index:     usize,
    threshold: D::Output,
}


impl<D, O> From<(usize, O)> for StumpSplit<D, O>
    where D: Data<Output = O>,
          O: PartialOrd,
{
    #[inline]
    fn from((index, threshold): (usize, O)) -> Self {
        Self { index, threshold }
    }
}


impl<D, O> SplitRule<D> for StumpSplit<D, O>
    where D: Data<Output = O>,
          O: PartialOrd,
{
    #[inline]
    fn split(&self, data: &D) -> LR {
        let value = data.value_at(self.index);

        if value < self.threshold {
            LR::Left
        } else {
            LR::Right
        }
    }
}


