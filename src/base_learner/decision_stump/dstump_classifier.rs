//! Provides the decision stump class.
use crate::Data;
// use crate::{Data, Label};
use crate::Classifier;


use std::hash::{Hash, Hasher};
use serde::{Serialize, Deserialize};


/// Defines the ray that are predicted as +1.0.
#[derive(Debug, Eq, PartialEq, Clone, Hash, Serialize, Deserialize)]
pub enum PositiveSide {
    /// The right-hand-side ray is predicted as +1.0
    RHS,
    /// The left-hand-side ray is predicted as +1.0
    LHS
}


/// The struct `DStumpClassifier` defines the decision stump class.
/// Given a point over the `d`-dimensional space,
/// A classifier predicts its label as
/// `sgn(x[i] - b)`, where `b` is the some intercept.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DStumpClassifier {

    /// The intercept of the stump
    pub threshold: f64,

    /// The index of the feature used in prediction.
    pub feature_index: usize,

    /// A ray to be predicted as +1.0
    pub positive_side: PositiveSide
}


impl PartialEq for DStumpClassifier {
    fn eq(&self, other: &Self) -> bool {
        let threshold     = self.threshold     == other.threshold;
        let feature       = self.feature_index == other.feature_index;
        let positive_side = self.positive_side == other.positive_side;

        threshold && feature && positive_side
    }
}


impl Eq for DStumpClassifier {}

impl Hash for DStumpClassifier {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // TODO Check whether this hashing works as expected.
        let v: u64 = unsafe { std::mem::transmute(self.threshold) };
        v.hash(state);
        self.feature_index.hash(state);
        self.positive_side.hash(state);
    }
}


impl DStumpClassifier {
    /// Produce an empty `DStumpClassifier`.
    pub fn new() -> DStumpClassifier {
        DStumpClassifier {
            threshold:     0.0,
            feature_index: 0,
            positive_side: PositiveSide::RHS
        }
    }
}


impl<D> Classifier<D, f64> for DStumpClassifier
    where D: Data<Output = f64>
{
    fn predict(&self, data: &D) -> f64
    {
        let val = data.value_at(self.feature_index);
        match self.positive_side {
            PositiveSide::RHS => (val - self.threshold).signum(),
            PositiveSide::LHS => (self.threshold - val).signum()
        }
    }
}

