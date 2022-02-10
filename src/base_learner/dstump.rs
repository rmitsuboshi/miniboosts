//! Provides the decision stump class.
use crate::{Data, Label, Sample};
use crate::BaseLearner;
use crate::Classifier;

use serde::{Serialize, Deserialize};

use std::hash::{Hash, Hasher};




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


impl<D> Classifier<D> for DStumpClassifier
    where D: Data<Output = f64>
{
    fn predict(&self, data: &D) -> Label
    {
        let val = data.value_at(self.feature_index);
        match self.positive_side {
            PositiveSide::RHS => (val - self.threshold).signum(),
            PositiveSide::LHS => (self.threshold - val).signum()
        }
    }
}


pub(self) type IndicesByValue = Vec<usize>;
pub(self) type FeatureIndex   = Vec<IndicesByValue>;


/// The struct `DStump` generates a `DStumpClassifier`
/// for each call of `self.best_hypothesis(..)`.
pub struct DStump {
    pub(crate) indices: Vec<FeatureIndex>,
}


impl DStump {
    /// Construct an empty Decision Stump class.
    pub fn new() -> DStump {
        DStump { indices: Vec::new() }
    }


    /// Initializes and produce an instance of `DStump`.
    pub fn init<T>(sample: &Sample<T>) -> DStump
        where T: Data<Output = f64>,
    {
        let dim = sample.dim();


        // indices: Vec<FeatureIndex>
        // the j'th element of this vector stores
        // the grouped indices by value.
        let mut indices: Vec<_> = Vec::with_capacity(dim);
        for j in 0..dim {
            let mut vals = sample.iter()
                .enumerate()
                .map(|(i, (dat, _))| (i, dat.value_at(j)))
                // .map(|(i, example)| (i, example.value_at(j)))
                .collect::<Vec<(usize, f64)>>();

            vals.sort_by(|(_, a), (_, b)| a.partial_cmp(&b).unwrap());

            let mut vals = vals.into_iter();

            // Group the indices by j'th value
            // recall that IndicesByValue = Vec<usize>
            let mut temp: IndicesByValue;
            let mut v;
            {
                // Initialize `temp` and `v`
                let (i, _v) = vals.next().unwrap();
                temp = vec![i];
                v    = _v;
            }

            // recall that
            // FeatureIndex = Vec<IndicesByValue>
            //              = Vec<Vec<usize>>
            let mut index: FeatureIndex = Vec::new();
            while let Some((i, vv)) = vals.next() {
                if vv == v {
                    temp.push(i);
                } else {
                    v = vv;
                    index.push(temp);
                    temp = vec![i];
                }
            }
            index.push(temp);

            indices.push(index);
        }

        // Construct DStump
        DStump { indices }
    }
}



impl<D: Data<Output = f64>> BaseLearner<D> for DStump {
    type Clf = DStumpClassifier;
    fn best_hypothesis(&self, sample: &Sample<D>, distribution: &[f64])
        -> Self::Clf
    {
        let init_edge = distribution.iter()
            .zip(sample.iter())
            .fold(0.0, |acc, (dist, (_, lab))| acc + dist * *lab);

        let mut best_edge = init_edge - 1e-2;


        // This is the output of this function.
        // Initialize with some init value.
        let mut dstump = {
            let (min_dat, _) = &sample[self.indices[0][0][0]];
            DStumpClassifier {
                threshold:     min_dat.value_at(0) - 1.0,
                feature_index: 0_usize,
                positive_side: PositiveSide::RHS
            }
        };

        {
            // `self.indidces[i][j][k]` is the `k`th index
            // of the `j`th block of the `i`th feature
            // TODO this line may fail since self.indices[0][0] 
            // may have no element.
            let i   = self.indices[0][0][0];
            let (ith_dat, _) = &sample[i];
            let val = ith_dat.value_at(0);
            if val > 0.0 {
                dstump.threshold = val / 2.0;
            }
        }


        let mut update_params_mut = |edge: f64, threshold: f64, j: usize| {
            if best_edge < edge.abs() {
                dstump.threshold     = threshold;
                dstump.feature_index = j;
                best_edge = edge.abs();
                if edge > 0.0 {
                    dstump.positive_side = PositiveSide::RHS;
                } else {
                    dstump.positive_side = PositiveSide::LHS;
                }
            }
        };

        for (j, index) in self.indices.iter().enumerate() {
            let mut edge = init_edge;

            let mut index = index.iter().peekable();


            let mut right = {
                let idx = index.peek().unwrap();
                let (first_dat, _) = &sample[idx[0]];
                first_dat.value_at(j)
            };
            let mut left;


            while let Some(idx) = index.next() {
                let temp = idx.iter()
                    .fold(0.0, |acc, &i| {
                        let (_, lab) = &sample[i];
                        acc + distribution[i] * lab
                    });

                edge -= 2.0 * temp;

                left  = right;
                right = match index.peek() {
                    Some(next_index) => {
                        // TODO: This line can be replaced by
                        // `get_unchecked`
                        let i = next_index[0];
                        let (ith_dat, _) = &sample[i];
                        ith_dat.value_at(j)
                    },
                    None => {
                        left + 2.0
                    }
                };
                update_params_mut(edge, (left + right) / 2.0, j);
            }
        }


        dstump
    }
}

