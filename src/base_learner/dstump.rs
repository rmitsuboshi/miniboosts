use crate::data_type::{DType, Data, Label, Sample};
use crate::base_learner::core::BaseLearner;
use crate::base_learner::core::Classifier;

use std::collections::HashSet;
use std::hash::{Hash, Hasher};


#[derive(Debug, Eq, PartialEq, Clone, Hash)]
enum PositiveSide { RHS, LHS }


/// The struct `DStumpClassifier` defines the decision stump class.
/// Given a point over the `d`-dimensional space,
/// A classifier predicts its label as
/// sgn(x[i] - b), where b is the some intercept.
pub struct DStumpClassifier {
    pub(self) threshold: f64,
    pub(self) feature_index: usize,
    pub(self) positive_side: PositiveSide
}


impl PartialEq for DStumpClassifier {
    fn eq(&self, other: &Self) -> bool {
        let v1: u64 = unsafe { std::mem::transmute( self.threshold) };
        let v2: u64 = unsafe { std::mem::transmute(other.threshold) };

        v1 == v2 && self.feature_index == other.feature_index && self.positive_side == other.positive_side
    }
}


impl Eq for DStumpClassifier {}

impl Hash for DStumpClassifier {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let v: u64 = unsafe { std::mem::transmute(self.threshold) };
        v.hash(state);
        self.feature_index.hash(state);
        self.positive_side.hash(state);
    }
}


impl DStumpClassifier {
    pub fn new() -> DStumpClassifier {
        DStumpClassifier { threshold: 0.0, feature_index: 0, positive_side: PositiveSide::RHS }
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


/// For clarity, we define an alias.
type FeatureIndex = Vec<usize>;

/// The struct `DStump` generates a `DStumpClassifier`
/// for each call of `self.best_hypothesis(..)`.
pub struct DStump {
    pub sample_size: usize,  // Number of training examples
    pub feature_size: usize, // Number of features per example
    pub indices: Vec<FeatureIndex>,
}


impl DStump {
    pub fn new() -> DStump {
        DStump { sample_size: 0, feature_size: 0, indices: Vec::new() }
    }


    pub fn init(sample: &Sample<f64, f64>) -> DStump {
        let sample_size = sample.len();
        let feature_size = sample.feature_len();


        let mut indices = Vec::with_capacity(feature_size);
        match sample.dtype {
            DType::Sparse => {
                for j in 0..feature_size {
                    let mut _vals: Vec<(f64, usize)> = Vec::with_capacity(sample_size);
                    for i in 0..sample_size {
                        let _data = &sample[i].data;
                        let _v = _data.value_at(j);
                        if _v != 0.0 {
                            _vals.push((_v, i));
                        }
                    }
                    _vals.sort_by(|_a, _b| _a.0.partial_cmp(&_b.0).unwrap());
                    let _index = _vals.iter()
                                      .map(|tuple| tuple.1)
                                      .collect::<Vec<usize>>();
                    indices.push(_index);
                }
            },

            DType::Dense => {
                for j in 0..feature_size {
                    let vals = sample.iter()
                        .map(|example| example.data.value_at(j))
                        .collect::<Vec<f64>>();

                    let mut _index = (0..sample_size).collect::<Vec<usize>>();
                    _index.sort_unstable_by(|&ii, &jj| vals[ii].partial_cmp(&vals[jj]).unwrap());

                    indices.push(_index);
                }
            }
        }

        // Construct DStump
        DStump { sample_size, feature_size, indices }
    }
}

impl BaseLearner<f64, f64> for DStump {
    fn best_hypothesis(&self, sample: &Sample<f64, f64>, distribution: &[f64]) -> Box<dyn Classifier<f64, f64>> {
        let init_edge = distribution.iter()
            .zip(sample.iter())
            .fold(0.0, |mut acc, (d, example)| {
                acc += d * example.label;
                acc
            });

        let mut best_edge = init_edge;

        let mut dstump_classifier = DStumpClassifier {
            threshold: sample[self.indices[0][0]].data.value_at(0) - 1.0,
            feature_index: 0_usize,
            positive_side: PositiveSide::RHS
        };


        let mut update_params = |best_edge: &mut f64, edge: f64, threshold: f64, j: usize| {
            if *best_edge < edge.abs() {
                dstump_classifier.threshold = threshold;
                dstump_classifier.feature_index = j;
                *best_edge = edge.abs();
                if edge > 0.0 {
                    dstump_classifier.positive_side = PositiveSide::RHS;
                } else {
                    dstump_classifier.positive_side = PositiveSide::LHS;
                }
            }
        };

        match sample.dtype {
            DType::Sparse => {
                for j in 0..self.feature_size {
                    let mut edge = init_edge;

                    let zero_value = {
                        let mut _zero_values = 0.0;
                        let _idx = self.indices[j].iter().collect::<HashSet<&usize>>();
                        for _i in 0..self.sample_size {
                            if !_idx.contains(&_i) {
                                let _label = sample[_i].label;
                                _zero_values += _label * distribution[_i];
                            }
                        }
                        _zero_values
                    };


                    let mut left: f64;
                    let mut right = sample[self.indices[j][0]].data.value_at(j);

                    let mut idx = self.indices[j].iter().peekable();

                    let mut still_negative = right < 0.0;
                    while let Some(&i) = idx.next() {
                        let data  = &sample[i].data;
                        let label = &sample[i].label;


                        if still_negative && data.value_at(i) > 0.0 {
                            still_negative = false;
                            left = right;
                            right = 0.0;

                            edge -= 2.0 * zero_value;

                            let threshold = (left + right) / 2.0;
                            update_params(&mut best_edge, edge, threshold, j);
                        }

                        edge -= 2.0 * distribution[i] * label;


                        left = right;
                        if let Some(&&i_next) = idx.peek() {
                            let next_data = &sample[i_next].data;
                            if right == next_data.value_at(j) {
                                continue;
                            }
                            right = next_data.value_at(j);
                        } else {
                            right = data.value_at(j) + 1.0;
                        }


                        let threshold = (left + right) / 2.0;
                        update_params(&mut best_edge, edge, threshold, j);
                    }

                    if still_negative {
                        left = right;
                        right = 0.0;

                        edge -= 2.0 * zero_value;

                        let threshold = (left + right) / 2.0;
                        update_params(&mut best_edge, edge, threshold, j);
                    }
                }
            },
            DType::Dense => {
                for j in 0..self.feature_size {
                    let mut edge = init_edge;


                    let mut left;
                    let mut right = sample[self.indices[j][0]].data.value_at(j);


                    let mut idx = self.indices[j].iter().peekable();

                    while let Some(&i) = idx.next() {
                        let label = &sample[i].label;

                        edge -= 2.0 * distribution[i] * label;

                        left = right;
                        if let Some(&&i_next) = idx.peek() {
                            let next_data = &sample[i_next].data;
                            if right == next_data.value_at(j) {
                                continue;
                            }
                            right = next_data.value_at(j);
                        } else {
                            right = sample[i].data.value_at(j) + 1.0;
                        }

                        let threshold = (left + right) / 2.0;
                        update_params(&mut best_edge, edge, threshold, j);
                    }
                }
            },
        }
        Box::new(dstump_classifier)
    }
}

// /// Since struct `DStumpClassifier` has an element that does not implement the `Eq` trait,
// /// We cannot use `DStumpClassifier` as a key of `HashMap`.
// /// `HashableDStumpClassifier` is the struct that separates `threshold` into
// /// the fractional and integral part.

// type Mantissa = u64;
// type Exponent = i16;
// type Sign     = i8;
// 
// #[derive(Debug, PartialEq, Eq, Hash)]
// struct HashableDStumpClassifier {
//     mantissa_exp:  (u64, i16, i8),
//     feature_index: usize,
//     positive_side: PositiveSide
// }
// 
// 
// use std::convert::From;
// 
// /// Convert `DStumpClassifier` to `HashableDStumpClassifier`
// impl From<DStumpClassifier> for HashableDStumpClassifier {
//     fn from(dstump: DStumpClassifier) -> Self {
//         let mantissa_exp = integer_decode(dstump.threshold);
//         HashableDStumpClassifier {
//             mantissa_exp, feature_index: dstump.feature_index, positive_side: dstump.positive_side
//         }
//     }
// }
// 
// 
// fn integer_decode(val: f64) -> (Mantissa, Exponent, Sign) {
//     use std::mem;
//     let bits: u64 = unsafe { mem::transmute(val) };
//     let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
//     let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
//     let mantissa = if exponent == 0 {
//         (bits & 0xfffffffffffff) << 1
//     } else {
//         (bits & 0xfffffffffffff) | 0x10000000000000
//     };
// 
//     exponent -= 1023 + 52;
//     (mantissa, exponent, sign)
// }
