use crate::Data;


/// The output of the function `split` of `SplitRule`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LR {
    Left,
    Right,
}


/// A trait that defines the splitting rule
/// in the decision tree.
pub trait SplitRule {
    type Input;
    fn split(&self, data: &Self::Input) -> LR;
}


/// Defines the split based on a feature.
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


impl<D, O> SplitRule for StumpSplit<D, O>
    where D: Data<Output = O>,
          O: PartialOrd,
{
    type Input = D;
    #[inline]
    fn split(&self, data: &Self::Input) -> LR {
        let value = data.value_at(self.index);

        if value < self.threshold {
            LR::Left
        } else {
            LR::Right
        }
    }
}


