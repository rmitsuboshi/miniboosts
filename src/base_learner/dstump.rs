use crate::data_type::{DType, Data, Label, Sample};
use crate::base_learner::core::BaseLearner;
use crate::base_learner::core::Classifier;
use std::collections::HashSet;



#[derive(Eq, PartialEq)]
enum PositiveSide { RHS, LHS }


#[derive(PartialEq)]
pub struct DStumpClassifier {
    threshold: f64,
    feature_index: usize,
    positive_side: PositiveSide
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


type FeatureIndex = Vec<usize>;


pub struct DStump {
    pub sample_size: usize,  // Number of training examples
    pub feature_size: usize, // Number of features per example
    pub indices: Vec<FeatureIndex>,
}


impl DStump {
    pub fn new() -> DStump {
        DStump { sample_size: 0, feature_size: 0, indices: Vec::new() }
    }


    pub fn with_sample(sample: &Sample<f64, f64>) -> DStump {
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
                            _vals.push((_v, j));
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
                    let vals = {
                        let mut _vals = vec![0.0; sample_size];
                        for i in 0..sample_size {
                            let _data = &sample[i].data;
                            _vals[i] = _data.value_at(j);
                        }
                        _vals
                    };

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
        let init_edge = {
            let mut _edge = 0.0;
            for i in 0..self.sample_size {
                let label = sample[i].label;
                _edge += distribution[i] * label;
            }
            _edge
        };

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



