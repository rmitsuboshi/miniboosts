//! Provides the decision stump class.
use crate::data_type::{DType, Data, Label, Sample};
use crate::base_learner::core::BaseLearner;
use crate::base_learner::core::Classifier;

use std::collections::HashSet;
use std::hash::{Hash, Hasher};


/// Defines the ray that are predicted as +1.0.
#[derive(Debug, Eq, PartialEq, Clone, Hash)]
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


impl Classifier<f64, f64> for DStumpClassifier {
    fn predict(&self, data: &Data<f64>) -> Label<f64> {
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
    pub fn init(sample: &Sample<f64, f64>) -> DStump {
        let feature_size = sample.feature_len();


        // indices: Vec<FeatureIndex>
        // the j'th element of this vector stores
        // the grouped indices by value.
        let mut indices: Vec<_> = Vec::with_capacity(feature_size);
        for j in 0..feature_size {
            let mut vals = match sample.dtype {
                DType::Sparse => {
                    sample.iter().enumerate()
                        .filter_map(|(i, ex)| {
                            let v = ex.data.value_at(j);
                            if v != 0.0 {
                                Some((i, v))
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<(usize, f64)>>()
                },
                DType::Dense => {
                    sample.iter()
                        .enumerate()
                        .map(|(i, ex)| (i, ex.data.value_at(j)))
                        .collect::<Vec<(usize, f64)>>()
                }
            };


            // If all the values in the j'th feature are zero, then skip
            if vals.is_empty() {
                indices.push(
                    Vec::with_capacity(0_usize)
                );
                continue;
            }

            vals.sort_by(|(_, v1), (_, v2)| v1.partial_cmp(&v2).unwrap());

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



impl BaseLearner<f64, f64> for DStump {
    fn best_hypothesis(&self, sample: &Sample<f64, f64>, distribution: &[f64])
        -> Box<dyn Classifier<f64, f64>>
    {
        let init_edge = distribution.iter()
            .zip(sample.iter())
            .fold(0.0, |acc, (d, example)| acc + d * example.label);

        let mut best_edge = init_edge - 1e-2;


        // This is the output of this function.
        // Initialize with some init value.
        let mut dstump = DStumpClassifier {
            threshold: sample[self.indices[0][0][0]].data.value_at(0) - 1.0,
            feature_index: 0_usize,
            positive_side: PositiveSide::RHS
        };

        {
            // `self.indidces[i][j][k]` is the `k`th index
            // of the `j`th block of the `i`th feature
            // TODO this line may fail since self.indices[0][0] 
            // may have no element.
            let i   = self.indices[0][0][0];
            let val = sample[i].data.value_at(0);
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

            // Compute the sum of `sample[i].label * distribution[i]`,
            // where i is the sample index that has 0 in j'th feature.
            let zero_value = match sample.dtype {
                DType::Dense  => 0.0,
                DType::Sparse => {
                    let idx = index.iter()
                        .flatten()
                        .collect::<HashSet<_>>();

                    sample.iter()
                        .zip(distribution.iter())
                        .enumerate()
                        .fold(0.0, |acc, (i, (ex, &d))| {
                            if idx.contains(&&i) {
                                acc
                            } else {
                                acc + ex.label * d
                            }
                        })
                }
            };


            let mut edge = init_edge;

            let mut index = index.iter().peekable();


            // If all the value in the j'th feature are zero,
            // check the best edge and continue
            if None == index.peek() {
                edge -= 2.0 * zero_value;
                update_params_mut(edge, 1.0, j);
                continue;
            }


            let mut right = {
                let idx = index.peek().unwrap();
                let i   = idx[0];
                sample[i].data.value_at(j)
            };
            let mut left;


            // All values are non-negative
            if right > 0.0 {
                //                          right
                //          0.0         (first value)
                //           v                v
                // ----------|----------------|-------------->
                //                   ^                       R
                //              threshold

                edge -= 2.0 * zero_value;
                update_params_mut(edge, right / 2.0, j);
            }

            while let Some(idx) = index.next() {
                let temp = idx.iter()
                    .fold(0.0, |acc, &i| {
                        acc + distribution[i] * sample[i].label
                    });

                edge -= 2.0 * temp;

                left = right;
                match index.peek() {
                    Some(next_index) => {
                        right = {
                            let i = next_index[0];
                            sample[i].data.value_at(j)
                        };

                        if left * right < 0.0
                            && sample.dtype == DType::Sparse
                        {
                            update_params_mut(edge, left / 2.0, j);

                            edge -= 2.0 * zero_value;
                            left  = 0.0;
                        }
                        update_params_mut(edge, (left + right) / 2.0, j);
                    },
                    None => {
                        if left < 0.0 && sample.dtype == DType::Sparse {
                            update_params_mut(edge, left / 2.0, j);

                            edge -= 2.0 * zero_value;
                            left  = 0.0;
                            right = 2.0;
                        } else {
                            right = left + 2.0;
                        }
                        update_params_mut(edge, (left + right) / 2.0, j);
                    }
                }
            }
        }


        Box::new(dstump)
    }
}

