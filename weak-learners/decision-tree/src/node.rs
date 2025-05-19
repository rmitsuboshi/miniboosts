//! A node struct used in the decision tree algorithm.
use miniboosts_core::{
    tree::*,
    Classifier,
    Regressor,
    Sample,
};
use serde::{Serialize, Deserialize};
use std::fmt;

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum Node {
    Branch {
        splitter:   Splitter,
        left:       Box<Node>,
        right:      Box<Node>,
        confidence: f64,
    },
    Leaf {
        confidence: f64,
    },
}

impl Node {
    pub fn branch(
        splitter:   Splitter,
        left:       Box<Node>,
        right:      Box<Node>,
        confidence: f64,
    ) -> Self
    {
        Self::Branch {
            splitter,
            left,
            right,
            confidence,
        }
    }

    pub fn leaf(confidence: f64) -> Self {
        Self::Leaf { confidence, }
    }

    pub(crate) fn to_dot_info(&self, id: usize) -> (Vec<String>, usize) {
        match self {
            Node::Branch { splitter, left, right, .. } => {
                let splitter = format!(
                    "\tnode_{id} [ label = \"{feat} < {thr:.2} ?\" ];\n",
                    feat = splitter.feature,
                    thr  = splitter.threshold,
                );

                let left_id = id + 1;
                let (     left,  right_id) = left.to_dot_info(left_id);
                let (mut right, return_id) = right.to_dot_info(right_id);

                let mut info = left;
                info.push(splitter);
                info.append(&mut right);

                let left_edge = format!(
                    "\tnode_{id} -- node_{left_id} [ label = \"Yes\" ];\n",
                );
                info.push(left_edge);
                let right_edge = format!(
                    "\tnode_{id} -- node_{right_id} [ label = \"No\" ];\n",
                );
                info.push(right_edge);

                (info, return_id)
            },
            Node::Leaf { confidence, .. } => {
                let info = format!(
                    "\tnode_{id} [ label = \"{confidence}\", shape = box ];\n",
                );

                (vec![info], id + 1)
            }
        }
    }
}

impl Classifier for Node {
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        match self {
            Self::Branch { splitter, left, right, .. } => {
                match splitter.split(sample, row) {
                    LeftRight::Left  => left.confidence(sample, row),
                    LeftRight::Right => right.confidence(sample, row),
                }
            },
            Self::Leaf { confidence, .. } => {
                *confidence
            },
        }
    }
}

impl Regressor for Node {
    fn predict(&self, sample: &Sample, row: usize) -> f64 {
        match self {
            Self::Branch { splitter, left, right, .. } => {
                match splitter.split(sample, row) {
                    LeftRight::Left  => left.confidence(sample, row),
                    LeftRight::Right => right.confidence(sample, row),
                }
            },
            Self::Leaf { confidence, .. } => {
                *confidence
            },
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
            } => {
                f.debug_struct("Branch")
                    .field("splitter", &splitter)
                    .field("confidence", &confidence)
                    .field("left", &left)
                    .field("right", &right)
                    .finish()
            },
            Self::Leaf {
                confidence,
            } => {
                f.debug_struct("Leaf")
                    .field("confidence", &confidence)
                    .finish()
            },
        }
    }
}

