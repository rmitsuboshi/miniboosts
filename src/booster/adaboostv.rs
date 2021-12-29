use crate::data_type::{Data, Label, Sample};
use crate::booster::core::Booster;
use crate::base_learner::core::Classifier;
use crate::base_learner::core::BaseLearner;



/// Struct `AdaBoostV` has 4 parameters.
/// - `tolerance` is the gap parameter,
/// - `rho` is the guess of the optimal margin,
/// - `gamma` is the minimum edge over the past edges,
/// - `dist` is the distribution over training examples,
/// - `weights` is the weights over `classifiers` that the AdaBoostV obtained up to iteration `t`.
/// - `classifiers` is the classifier that the AdaBoostV obtained.
/// The length of `weights` and `classifiers` must be same.
pub struct AdaBoostV<D, L> {
    pub tolerance: f64,
    pub rho: f64,
    pub gamma: f64,
    pub dist: Vec<f64>,
    pub weights: Vec<f64>,
    pub classifiers: Vec<Box<dyn Classifier<D, L>>>,
}


impl<D, L> AdaBoostV<D, L> {
    pub fn init(sample: &Sample<D, L>) -> AdaBoostV<D, L> {
        let m = sample.len();
        assert!(m != 0);
        let uni = 1.0 / m as f64;
        AdaBoostV {
            tolerance: 0.0, rho: 1.0, gamma: 1.0, dist: vec![uni; m], weights: Vec::new(), classifiers: Vec::new()
        }
    }


    /// `max_loop` returns the maximum iteration of the Adaboost to find a combined hypothesis
    /// that has error at most `eps`.
    pub fn max_loop(&self, eps: f64) -> usize {
        let m = self.dist.len();

        2 * ((m as f64).ln() / (eps * eps)) as usize
    }


}


impl<D> Booster<D, f64> for AdaBoostV<D, f64> {

    /// `update_params` updates `self.distribution` and determine the weight on hypothesis
    /// that the algorithm obtained at current iteration.
    fn update_params(&mut self, h: Box<dyn Classifier<D, f64>>, sample: &Sample<D, f64>) -> Option<()> {


        let m = sample.len();

        let edge = self.dist.iter()
            .zip(sample.iter())
            .fold(0.0_f64, |mut acc, (d, example)| {
                acc += d * example.label * h.predict(&example.data);
                acc
            });


        // This assertion may fail because of the numerical error
        // assert!(edge >= -1.0);
        // assert!(edge <=  1.0);


        if edge.abs() >= 1.0 {
            self.weights.clear();
            self.classifiers.clear();

            let sgn = edge.signum();
            self.weights.push(sgn);
            self.classifiers.push(h);

            return None;
        }

        self.gamma = edge.min(self.gamma);

        self.rho = self.gamma - self.tolerance;


        let weight = {
            let _first  = ((1.0 + edge) / (1.0 - edge)).ln() / 2.0;
            let _second = ((1.0 + self.rho) / (1.0 - self.rho)).ln() / 2.0;

            _first - _second
        };


        // To prevent overflow, take the logarithm.
        for (d, example) in self.dist.iter_mut().zip(sample.iter()) {
            *d = d.ln() - weight * example.label * h.predict(&example.data);
        }

        let mut indices = (0..m).collect::<Vec<usize>>();
        indices.sort_unstable_by(|&i, &j| self.dist[i].partial_cmp(&self.dist[j]).unwrap());


        let mut normalizer = self.dist[indices[0]];
        for i in indices.into_iter().skip(1) {
            let mut a = normalizer;
            let mut b = self.dist[i];
            if a < b {
                std::mem::swap(&mut a, &mut b);
            }

            normalizer = a + (1.0 + (b - a).exp()).ln();
        }

        for d in self.dist.iter_mut() {
            *d = (*d - normalizer).exp();
        }

        self.classifiers.push(h);
        self.weights.push(weight);

        Some(())
    }


    // fn run(&mut self, base_learner: Box<dyn BaseLearner<D, f64>>, sample: &Sample<D, f64>, eps: f64) {
    fn run(&mut self, base_learner: &dyn BaseLearner<D, f64>, sample: &Sample<D, f64>, eps: f64) {
        self.tolerance = eps;
        let max_loop = self.max_loop(eps);
        dbg!("max_loop: {}", max_loop);

        for _t in 1..=max_loop {
            let h = base_learner.best_hypothesis(sample, &self.dist);
            if let None = self.update_params(h, sample) {
                println!("Break loop at: {}", _t);
                break;
            }
        }
    }


    fn predict(&self, data: &Data<D>) -> Label<f64> {
        assert_eq!(self.weights.len(), self.classifiers.len());

        let mut confidence = 0.0;
        for (w, h) in self.weights.iter().zip(self.classifiers.iter()) {
            confidence += w * h.predict(data);
        }


        confidence.signum()
    }
}
