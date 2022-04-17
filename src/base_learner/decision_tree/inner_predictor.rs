//! Defines the function that partition out the examples
//! on each node.

use crate::{Data, Label, Sample};

use serde::{Serialize, Deserialize};

use std::hash::{Hash, Hasher};

/// Defines the output of `InnerPredictor::transit(&data)`.
#[derive(Debug, Eq, PartialEq, Clone, Hash, Serialize, Deserialize)]
pub(super) enum Child {
    /// Data with the specified value is greater than some threshold
    /// transit to `node.positive`.
    Positive,
    /// Data with the specified value is lower than some threshold
    /// transit to `node.negative`.
    Negative
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct InnerPredictor {
    pub(super) index: usize,
    pub(super) value: f64,
    pub(super) large: Child,
}


impl PartialEq for InnerPredictor {
    fn eq(&self, other: &Self) -> bool {
        let value = self.value == other.value;
        let index = self.index == other.index;
        let large = self.large == other.large;

        value && index && large
    }
}


impl Eq for InnerPredictor {}

impl Hash for InnerPredictor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // TODO Check whether this hashing works as expected.
        let v: u64 = unsafe { std::mem::transmute(self.value) };
        v.hash(state);
        self.index.hash(state);
        self.large.hash(state);
    }
}


impl InnerPredictor {
    pub(super) fn transit<D>(&self, data: &D)
        -> Child
        where D: Data<Output = f64>
    {
        let value = data.value_at(self.index);

        let diff = match self.large {
            Child::Positive => (value - self.value),
            Child::Negative => (self.value - value),
        };

        if diff > 0.0 {
            Child::Positive
        } else {
            Child::Negative
        }
    }
}



