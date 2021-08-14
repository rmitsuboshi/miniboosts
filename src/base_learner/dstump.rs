enum PositiveSide { RHS, LHS }


pub struct DStump {
    pub example_size: usize,
    pub feature_size: usize,
    pub indices: Vec<Vec<usize>>,
}


impl DStump {
    pub fn new() -> DStump {
        DStump { example_size: 0, feature_size: 0, indices: Vec::new() }
    }

    pub fn with_sample(examples: &[Vec<f64>], labels: &[f64]) -> DStump {
        assert_eq!(examples.len(), labels.len());
        let example_size = examples.len();

        assert!(examples.len() > 0);
        let feature_size = examples[0].len();

        let mut indices = Vec::with_capacity(feature_size);

        for j in 0..feature_size {
            let vals = {
                let mut _vals = vec![0.0; example_size];
                for i in 0..example_size {
                    _vals[i] = examples[i][j];
                }
                _vals
            };

            let mut idx = (0..example_size).collect::<Vec<usize>>();
            idx.sort_unstable_by(|&ii, &jj| vals[ii].partial_cmp(&vals[jj]).unwrap());

            indices.push(idx);
        }
        DStump { example_size, feature_size, indices }
    }

    pub fn best_hypothesis(&self, examples: &[Vec<f64>], labels: &[f64], distribution: &[f64]) -> Box<dyn Fn(&[f64]) -> f64> {
        let init_edge = {
            let mut _edge = 0.0;
            for i in 0..self.example_size {
                _edge += distribution[i] * labels[i];
            }
            _edge
        };

        let mut best_threshold = examples[self.indices[0][0]][0] - 1.0;
        let mut best_feature = 0_usize;
        let mut best_sense = PositiveSide::RHS;
        let mut best_edge = init_edge;



        for j in 0..self.feature_size {
            let idx = &self.indices[j];

            let mut edge = init_edge;


            let mut left  = examples[idx[0]][j] - 1.0;
            let mut right = examples[idx[0]][j];


            for ii in 0..self.example_size {
                let i = idx[ii];

                edge -= 2.0 * distribution[i] * labels[i];

                if i + 1_usize != self.example_size && right == examples[i+1][j] { continue; }

                left  = right;
                right = if ii + 1_usize == self.example_size { examples[i][j] + 1.0 } else { examples[idx[ii+1]][j] };

                if best_edge < edge.abs() {
                    best_threshold = (left + right) / 2.0;
                    best_feature   = j;
                    if edge > 0.0 {
                        best_edge  = edge;
                        best_sense = PositiveSide::RHS;
                    } else {
                        best_edge  = - edge;
                        best_sense = PositiveSide::LHS;
                    }
                }
            }
        }

        let sense = match best_sense {
            PositiveSide::RHS => true,
            _ => false
        };
        println!("thr: {}, feature: {}, sense: {}, edge: {}", best_threshold, best_feature, sense, best_edge);

        let hypothesis = move |data: &[f64]| -> f64 {
            let val = data[best_feature];
            match best_sense {
                PositiveSide::RHS => {
                    (val - best_threshold).signum()
                },
                PositiveSide::LHS => {
                    (best_threshold - val).signum()
                }
            }
        };
        Box::new(hypothesis)
    }
}



