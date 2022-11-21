//! Defines the decision tree classifier.
use polars::prelude::*;
use crate::Classifier;


use super::node::*;
use serde::{Serialize, Deserialize};


/// Decision tree classifier.
/// This struct is just a wrapper of `Node`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DTreeClassifier {
    root: Node
}


impl From<Node> for DTreeClassifier {
    #[inline]
    fn from(root: Node) -> Self {
        Self { root }
    }
}


impl Classifier for DTreeClassifier {
    fn confidence(&self, data: &DataFrame, row: usize) -> f64 {
        self.root.confidence(data, row)
    }
}


