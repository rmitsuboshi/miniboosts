//! Defines the decision tree classifier.
use crate::{Data, Label};
use crate::Classifier;


use super::split_rule::*;
use super::node::*;


/// Decision tree classifier.
/// This struct is just a wrapper of `Node`.
pub struct DTreeClassifier<S>
    where S: SplitRule
{
    root: Node<S>
}


impl<O, D, S> Classifier<D> for DTreeClassifier<S>
    where S: SplitRule<Input = D>,
          D: Data<Output = O>,
          O: PartialOrd
{
    fn predict(&self, data: &D) -> Label {
        self.root.predict(data)
    }
}
