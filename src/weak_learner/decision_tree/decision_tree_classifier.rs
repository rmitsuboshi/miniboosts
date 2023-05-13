//! Defines the decision tree classifier.
use crate::{Classifier, Sample};


use super::node::*;
use serde::{Serialize, Deserialize};

use std::path::Path;
use std::fs::File;
use std::io::prelude::*;


/// Decision tree classifier.
/// This struct is just a wrapper of `Node`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionTreeClassifier {
    root: Node
}


impl From<Node> for DecisionTreeClassifier {
    #[inline]
    fn from(root: Node) -> Self {
        Self { root }
    }
}


impl Classifier for DecisionTreeClassifier {
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        self.root.confidence(sample, row)
    }
}


impl DecisionTreeClassifier {
    /// Write the current decision tree to dot file.
    #[inline]
    pub fn to_dot_file<P>(&self, path: P) -> std::io::Result<()>
        where P: AsRef<Path>
    {
        let mut f = File::create(path)?;
        f.write_all(b"graph DecisionTree {")?;


        let info = self.root.to_dot_info(0).0;
        info.into_iter()
            .for_each(|row| {
                f.write_all(row.as_bytes()).unwrap();
            });

        f.write_all(b"}")?;

        Ok(())
    }
}
