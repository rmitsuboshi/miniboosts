use crate::data_type::{DType, Data, Label, Sample};
use crate::base_learner::core::BaseLearner;
use crate::base_learner::core::Classifier;



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
    fn predict(&self, example: &Data<f64>) -> Label<f64> {
        let val = example.value_at(self.feature_index);
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
                        let _example = &sample[i].0;
                        let _v = _example.value_at(j);
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
                            // get ith example
                            let _example = &sample[i].0;
                            _vals[i] = _example.value_at(j);
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
                let label = sample[i].1;
                _edge += distribution[i] * label;
            }
            _edge
        };

        let mut best_edge = init_edge;

        let mut dstump_classifier = DStumpClassifier {
            threshold: sample[self.indices[0][0]].0.value_at(0) - 1.0,
            feature_index: 0_usize,
            positive_side: PositiveSide::RHS
        };



        match sample.dtype {
            DType::Sparse => {
                todo!();
            },
            DType::Dense => {
                for j in 0..self.feature_size {
                    let idx = &self.indices[j];

                    let mut edge = init_edge;


                    let mut left;
                    let mut right = sample[idx[0]].0.value_at(j);


                    for ii in 0..self.sample_size {
                        let i = idx[ii];

                        let (example, label) = &sample[i];

                        edge -= 2.0 * distribution[i] * label;

                        if ii + 1_usize != self.sample_size && (right == sample[idx[ii+1]].0.value_at(j) || *label == sample[idx[ii+1]].1) { continue; }

                        left  = right;
                        right = if ii + 1_usize == self.sample_size { example.value_at(j) + 1.0 } else { sample[idx[ii+1]].0.value_at(j) };

                        if best_edge < edge.abs() {
                            dstump_classifier.threshold = (left + right) / 2.0;
                            dstump_classifier.feature_index = j;
                            if edge > 0.0 {
                                best_edge  = edge;
                                dstump_classifier.positive_side = PositiveSide::RHS;
                            } else {
                                best_edge  = - edge;
                                dstump_classifier.positive_side = PositiveSide::LHS;
                            }
                        }
                    }
                }
                Box::new(dstump_classifier)
            },
        }
    }
}



