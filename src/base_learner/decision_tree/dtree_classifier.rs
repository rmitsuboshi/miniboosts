//! Defines the decision tree classifier.
use crate::Data;
use crate::Classifier;


use super::split_rule::*;
use super::node::*;


/// Decision tree classifier.
/// This struct is just a wrapper of `Node`.
#[derive(Debug)]
pub struct DTreeClassifier<S, L> {
    root: Node<S, L>
}


impl<O, D, S, L> Classifier<D, L> for DTreeClassifier<S, L>
    where S: SplitRule<D>,
          D: Data<Output = O>,
          L: PartialEq + Clone,
          O: PartialOrd
{
    fn predict(&self, data: &D) -> L {
        self.root.predict(data)
    }
}
