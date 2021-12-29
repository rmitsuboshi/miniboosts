use crate::data_type::{DType, Data, Label, Sample};
use crate::base_learner::core::BaseLearner;
use crate::base_learner::core::Classifier;

use std::collections::HashSet;
use std::hash::{Hash, Hasher};


#[derive(Debug, Eq, PartialEq, Clone, Hash)]
pub enum PositiveSide { RHS, LHS }


/// The struct `DStumpClassifier` defines the decision stump class.
/// Given a point over the `d`-dimensional space,
/// A classifier predicts its label as
/// sgn(x[i] - b), where b is the some intercept.
pub struct DStumpClassifier {
    pub threshold: f64,
    pub feature_index: usize,
    pub positive_side: PositiveSide
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
        // TODO FIX BUG in this line
        // this unsafe does not work as expected.
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


/// For clarity, we define aliases.
type IndicesByValue = Vec<usize>;
type FeatureIndex = Vec<IndicesByValue>;
// type FeatureIndex = Vec<usize>;

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
        let sample_size  = sample.len();
        let feature_size = sample.feature_len();


        let mut indices: Vec<FeatureIndex> = Vec::with_capacity(feature_size);
        for j in 0..feature_size {
            let mut vals = match sample.dtype {
                DType::Sparse => {
                    sample.iter().enumerate()
                        .filter_map(|(i, ex)| {
                                let v = ex.data.value_at(j);
                                if v != 0.0 { Some((i, v)) } else { None }
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
            vals.sort_by(|(_, v1), (_, v2)| v1.partial_cmp(&v2).unwrap());

            let mut vals = vals.into_iter();

            // Group the indices by j'th value
            let mut temp: IndicesByValue;
            let mut v;
            {
                let (i, _v) = vals.next().unwrap();
                temp = vec![i];
                v = _v;
            }
            let mut index: Vec<IndicesByValue> = Vec::new();
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
        DStump { sample_size, feature_size, indices }
    }
}



impl BaseLearner<f64, f64> for DStump {
    fn best_hypothesis(&self, sample: &Sample<f64, f64>, distribution: &[f64]) -> Box<dyn Classifier<f64, f64>> {
        let init_edge = distribution.iter()
            .zip(sample.iter())
            .fold(0.0, |acc, (d, example)| acc + d * example.label);

        let mut best_edge = init_edge - 1e-2;
        // let mut best_edge = f64::MIN;

        // This is the output of this function.
        // Initialize with some init value.
        let mut dstump_classifier = DStumpClassifier {
            threshold: sample[self.indices[0][0][0]].data.value_at(0) - 1.0,
            feature_index: 0_usize,
            positive_side: PositiveSide::RHS
        };

        {
            // `self.indidces[i][j][k]` is the `k`th index
            // of the `j`th block of the `i`th feature
            let i   = self.indices[0][0][0];
            let val = sample[i].data.value_at(0);
            if val > 0.0 {
                dstump_classifier.threshold = val / 2.0;
            }
        }


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

        for (j, index) in self.indices.iter().enumerate() {

            // Compute the sum of `sample[i].label * distribution[i]`,
            // where i is the sample index that has 0 in j'th feature.
            let zero_value = match sample.dtype {
                DType::Dense => 0.0,
                DType::Sparse => {
                    let idx = index.iter().flatten().collect::<HashSet<_>>();

                    sample.iter()
                        .zip(distribution.iter())
                        .enumerate()
                        .fold(0.0, |acc, (i, (ex, &d))| {
                                if idx.contains(&&i) { acc } else { acc + ex.label * d }
                        })
                }
            };

            let mut edge = init_edge;

            let mut index = index.iter().peekable();
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
                update_params(&mut best_edge, edge, right / 2.0, j);
            }

            while let Some(idx) = index.next() {
                let temp = idx.iter()
                    .fold(0.0, |acc, &i| acc + distribution[i] * sample[i].label);

                edge -= 2.0 * temp;

                left = right;
                match index.peek() {
                    Some(next_index) => {
                        right = {
                            let i = next_index[0];
                            sample[i].data.value_at(j)
                        };

                        if left * right < 0.0 && sample.dtype == DType::Sparse {
                            update_params(&mut best_edge, edge, left / 2.0, j);

                            edge -= 2.0 * zero_value;
                            left = 0.0;
                        }
                        update_params(&mut best_edge, edge, (left + right) / 2.0, j);
                    },
                    None => {
                        if left < 0.0 && sample.dtype == DType::Sparse {
                            update_params(&mut best_edge, edge, left / 2.0, j);

                            edge -= 2.0 * zero_value;
                            left  = 0.0;
                            right = 2.0;
                        } else {
                            right = left + 2.0;
                        }
                        update_params(&mut best_edge, edge, (left + right) / 2.0, j);
                    }
                }
            }
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
