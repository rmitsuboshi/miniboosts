//! [`DecisionTreeClassifier`]: the hypothesis returned by
//! [`crate::DecisionTree`].
use crate::core::{Classifier, Sample};

use crate::weak_learners::decision_tree::node::*;
use serde::{Deserialize, Serialize};

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

/// A binary decision tree classifier, returned by
/// [`DecisionTree`](crate::DecisionTree)'s
/// [`produce`](crate::WeakLearner::produce).
///
/// It wraps the root [`Node`]. [`Classifier::confidence`] routes an example
/// from the root through the branches (`x[f] < t` goes left) and returns the
/// confidence stored at the reached leaf, a value in `[-1, 1]`.
/// The feature names in the tree must exist in the sample being predicted.
/// Serializable with `serde`.
///
/// # Examples
/// Predict on new data that has the same feature names as the training data.
/// ```
/// use std::io::BufReader;
/// use miniboosts::{Classifier, DecisionTreeBuilder, Sample, WeakLearner};
///
/// let train = b"x,class\n-2,-1\n-1,-1\n1,1\n2,1\n";
/// let train = Sample::from_reader(BufReader::new(&train[..]), true)
///     .unwrap()
///     .set_target("class");
/// let tree = DecisionTreeBuilder::new(&train).max_depth(1).build();
/// let h = tree.produce(&train, &[0.25; 4]);
///
/// // Unlabeled data: no `set_target` is needed for prediction.
/// let new = b"x\n-5\n0.7\n10\n";
/// let new = Sample::from_reader(BufReader::new(&new[..]), true).unwrap();
/// // The learned threshold lies in (-1, 1].
/// assert_eq!(h.predict(&new, 0), -1);
/// assert_eq!(h.predict(&new, 2), 1);
/// assert!(h.confidence_all(&new).iter().all(|c| (-1.0..=1.0).contains(c)));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionTreeClassifier {
    root: Box<Node>,
}

impl From<Box<Node>> for DecisionTreeClassifier {
    #[inline]
    fn from(root: Box<Node>) -> Self {
        Self { root }
    }
}

impl Classifier for DecisionTreeClassifier {
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        self.root.confidence(sample, row)
    }
}

impl DecisionTreeClassifier {
    /// Writes the tree to `path` in Graphviz DOT format.
    /// Branch nodes are labeled `feature < threshold ?` (threshold rounded to
    /// two decimals) with edges `Yes` (left) and `No` (right); leaf nodes
    /// show their confidence.
    ///
    /// # Errors
    /// Returns `Err` if the file cannot be created or the opening/closing
    /// lines cannot be written.
    ///
    /// # Panics
    /// Panics if writing a node or edge line fails.
    ///
    /// # Examples
    /// ```
    /// use std::io::BufReader;
    /// use miniboosts::{DecisionTreeBuilder, Sample, WeakLearner};
    ///
    /// let csv = b"x,class\n-2,-1\n-1,-1\n1,1\n2,1\n";
    /// let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    ///     .unwrap()
    ///     .set_target("class");
    /// let tree = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    /// let h = tree.produce(&sample, &[0.25; 4]);
    ///
    /// let path = std::env::temp_dir()
    ///     .join(format!("miniboosts_doc_tree_{}.dot", std::process::id()));
    /// h.to_dot_file(&path).unwrap();
    /// let dot = std::fs::read_to_string(&path).unwrap();
    /// # std::fs::remove_file(&path).unwrap();
    /// assert!(dot.starts_with("graph DecisionTree {"));
    /// assert!(dot.contains("x < ")); // the root's splitting rule
    /// // Render with Graphviz, e.g. `dot -Tpng tree.dot -o tree.png`.
    /// ```
    #[inline]
    pub fn to_dot_file<P>(&self, path: P) -> std::io::Result<()>
    where
        P: AsRef<Path>,
    {
        let mut f = File::create(path)?;
        f.write_all(b"graph DecisionTree {")?;

        let info = self.root.to_dot_info(0).0;
        info.into_iter().for_each(|row| {
            f.write_all(row.as_bytes()).unwrap();
        });

        f.write_all(b"}")?;

        Ok(())
    }
}
