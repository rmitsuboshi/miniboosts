//! [`Node`]: the recursive tree structure inside a
//! [`crate::DecisionTreeClassifier`].
use crate::core::{Classifier, Regressor, Sample, tree::*};
use serde::{Deserialize, Serialize};
use std::fmt;

/// A node of a binary decision tree.
///
/// Evaluation ([`Classifier::confidence`], or equivalently
/// [`Regressor::predict`]) follows `splitter` from this node down to a leaf
/// and returns that leaf's confidence.
///
/// # Examples
/// Build the stump `x < 0.5 ? -1 : +0.8` by hand.
/// ```
/// use miniboosts::{Classifier, DecisionTreeClassifier, Sample};
/// use miniboosts::core::tree::Splitter;
/// use miniboosts::node::Node;
///
/// let root = Node::branch(
///     Splitter::new("feature_0", 0.5),
///     Box::new(Node::leaf(-1.0)),
///     Box::new(Node::leaf(0.8)),
///     0.0, // stored for inspection only; not used for prediction
/// );
/// let rows = vec![vec![0.0], vec![0.5], vec![3.0]];
/// let sample = Sample::from_dense(&rows, None).unwrap();
/// // Values strictly below the threshold go left.
/// assert_eq!(root.confidence_all(&sample), vec![-1.0, 0.8, 0.8]);
///
/// // Wrap it into the classifier type returned by `DecisionTree`.
/// let h = DecisionTreeClassifier::from(Box::new(root));
/// assert_eq!(h.predict_all(&sample), vec![-1, 1, 1]);
/// ```
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum Node {
    /// An internal node that routes examples by a threshold rule.
    Branch {
        /// The rule: examples with `x[feature] < threshold` go `left`.
        splitter: Splitter,
        /// The subtree for examples with `x[feature] < threshold`.
        left: Box<Node>,
        /// The subtree for the other examples.
        right: Box<Node>,
        /// The weighted-majority confidence of the training examples that
        /// reached this node (see [`crate::DecisionTree`]). Not used for
        /// prediction.
        confidence: f64,
    },
    /// A terminal node.
    Leaf {
        /// The confidence output for examples reaching this leaf, in
        /// `[-1, 1]` for trees grown by [`crate::DecisionTree`].
        confidence: f64,
    },
}

impl Node {
    /// Creates a [`Node::Branch`].
    pub fn branch(splitter: Splitter, left: Box<Node>, right: Box<Node>, confidence: f64) -> Self {
        Self::Branch {
            splitter,
            left,
            right,
            confidence,
        }
    }

    /// Creates a [`Node::Leaf`] that outputs `confidence`.
    pub fn leaf(confidence: f64) -> Self {
        Self::Leaf { confidence }
    }

    pub(crate) fn to_dot_info(&self, id: usize) -> (Vec<String>, usize) {
        match self {
            Node::Branch {
                splitter,
                left,
                right,
                ..
            } => {
                let splitter = format!(
                    "\tnode_{id} [ label = \"{feat} < {thr:.2} ?\" ];\n",
                    feat = splitter.feature,
                    thr = splitter.threshold,
                );

                let left_id = id + 1;
                let (left, right_id) = left.to_dot_info(left_id);
                let (mut right, return_id) = right.to_dot_info(right_id);

                let mut info = left;
                info.push(splitter);
                info.append(&mut right);

                let left_edge = format!("\tnode_{id} -- node_{left_id} [ label = \"Yes\" ];\n",);
                info.push(left_edge);
                let right_edge = format!("\tnode_{id} -- node_{right_id} [ label = \"No\" ];\n",);
                info.push(right_edge);

                (info, return_id)
            }
            Node::Leaf { confidence, .. } => {
                let info = format!("\tnode_{id} [ label = \"{confidence}\", shape = box ];\n",);

                (vec![info], id + 1)
            }
        }
    }
}

impl Classifier for Node {
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        match self {
            Self::Branch {
                splitter,
                left,
                right,
                ..
            } => match splitter.split(sample, row) {
                LeftRight::Left => left.confidence(sample, row),
                LeftRight::Right => right.confidence(sample, row),
            },
            Self::Leaf { confidence, .. } => *confidence,
        }
    }
}

impl Regressor for Node {
    fn predict(&self, sample: &Sample, row: usize) -> f64 {
        match self {
            Self::Branch {
                splitter,
                left,
                right,
                ..
            } => match splitter.split(sample, row) {
                LeftRight::Left => left.confidence(sample, row),
                LeftRight::Right => right.confidence(sample, row),
            },
            Self::Leaf { confidence, .. } => *confidence,
        }
    }
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Branch {
                splitter,
                left,
                right,
                confidence,
            } => f
                .debug_struct("Branch")
                .field("splitter", &splitter)
                .field("confidence", &confidence)
                .field("left", &left)
                .field("right", &right)
                .finish(),
            Self::Leaf { confidence } => f
                .debug_struct("Leaf")
                .field("confidence", &confidence)
                .finish(),
        }
    }
}
