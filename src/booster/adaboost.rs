// We first implement adaboost only.
// then we generalize it to boosting framework.
//

pub struct AdaBoost {
    pub dist: Vec<f64>,
    pub weights: Vec<f64>,
    pub classifiers: Vec<Box<dyn Fn(&[f64]) -> f64>>,
    max_loop: usize,
}


impl AdaBoost {
    pub fn new() -> AdaBoost {
        AdaBoost {
            dist: Vec::new(), weights: Vec::new(), classifiers: Vec::new(), max_loop: 0
        }
    }


    pub fn with_samplesize(m: usize) -> AdaBoost {
        assert!(m != 0);
        let uni = 1.0 / m as f64;
        AdaBoost {
            dist: vec![uni; m], weights: Vec::new(), classifiers: Vec::new(), max_loop: 0
        }
    }


    pub fn update_params(&mut self, h: Box<dyn Fn(&[f64]) -> f64>, examples: &[Vec<f64>], labels: &[f64]) {

        assert_eq!(examples.len(), labels.len());

        let m = examples.len();

        let mut edge = 0.0;
        for i in 0..m {
            edge += self.dist[i] * labels[i] * h(&examples[i]);
        }


        let weight_of_h = ((1.0 + edge) / (1.0 - edge)).ln() / 2.0;


        for i in 0..m {
            self.dist[i] = self.dist[i].ln() - weight_of_h * (labels[i] * h(&examples[i]));
        }

        let mut indices = (0..m).collect::<Vec<usize>>();
        indices.sort_unstable_by(|&i, &j| self.dist[i].partial_cmp(&self.dist[j]).unwrap());


        let mut normalizer = 0.0;
        for i in 0..m {
            let mut a = normalizer;
            let mut b = self.dist[indices[i]];
            if self.dist[indices[i]] > normalizer {
                a = self.dist[indices[i]];
                b = normalizer;
            }

            normalizer = a + (b - a).exp().ln();
        }

        for i in 0..m {
            self.dist[i] /= normalizer;
        }

        self.classifiers.push(h);
        self.weights.push(weight_of_h);
    }


    pub fn predict(&self, example: &[f64]) -> f64 {
        assert_eq!(self.weights.len(), self.classifiers.len());
        let n = self.weights.len();

        let mut confidence = 0.0;
        for i in 0..n {
            confidence +=  self.weights[i] * self.classifiers[i](example);
        }


        confidence.signum()
    }

    pub fn predict_all(&self, examples: &[Vec<f64>]) -> Vec<f64> {
        let mut predictions = Vec::new();

        for example in examples.iter() {
            predictions.push(self.predict(&example));
        }
        predictions
    }
}
