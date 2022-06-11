//! Defines the decision tree classifier.
use crate::Data;
use crate::Classifier;


use super::node::*;
use serde::{Serialize, Deserialize};


/// Decision tree classifier.
/// This struct is just a wrapper of `Node`.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct DTreeClassifier<O, L> {
    root: Node<O, L>
}


impl<O, L> From<Node<O, L>> for DTreeClassifier<O, L> {
    #[inline]
    fn from(root: Node<O, L>) -> Self {
        Self { root }
    }
}


impl<O, D, L> Classifier<D, L> for DTreeClassifier<O, L>
    where D: Data<Output = O>,
          L: PartialEq + Clone,
          O: PartialOrd
{
    fn predict(&self, data: &D) -> L {
        self.root.predict(data)
    }
}


