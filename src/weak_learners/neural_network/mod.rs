//! A network with one hidden layer as a weak learner.
//!
//! - [`TwoLayerNetwork`] implements [`crate::WeakLearner`]: it trains the
//!   network on the weighted sample with [`Adam`](crate::Adam) or
//!   [`Adopt`](crate::Adopt), flips it if its edge is negative, and retrains
//!   from a new initialization if the edge is too small.
//! - [`TwoLayerNetworkClassifier`] is the resulting hypothesis, with
//!   confidence `tanh(s(x))` or the sign of the score `s(x)`.
//! - [`Activation`], [`NetworkLoss`], and [`NetworkOutput`] configure the
//!   hidden layer, the training loss, and the output.
//!
//! The optimizers live in [`crate::optimization::adaptive`] and do not
//! depend on the network.

mod activation;
mod learner;
mod network;

pub use activation::Activation;
pub use learner::{DEFAULT_ITERATIONS, DEFAULT_LEARNING_RATE, TwoLayerNetwork};
pub use network::{NetworkLoss, NetworkOutput, TwoLayerNetworkClassifier};

#[cfg(test)]
mod tests {
    use super::network::{accumulate_gradient, flatten, unflatten};
    use super::*;
    use crate::core::{Classifier, Sample, WeakLearner};
    use crate::optimization::adaptive::{Adam, AdaptiveOptimizer};

    // Positive inside a disc, negative outside, on a 9 × 9 grid.
    fn disc() -> (Sample, Vec<f64>) {
        let mut rows = Vec::new();
        let mut y = Vec::new();
        for i in -4..=4 {
            for j in -4..=4 {
                let (a, b) = (f64::from(i) * 0.6, f64::from(j) * 0.6);
                rows.push(vec![a, b]);
                y.push(if a * a + b * b < 2.25 { 1.0 } else { -1.0 });
            }
        }
        (Sample::from_dense(&rows, Some(&y)).unwrap(), y)
    }

    fn accuracy(h: &TwoLayerNetworkClassifier, sample: &Sample, y: &[f64]) -> f64 {
        let correct = h
            .predict_all(sample)
            .iter()
            .zip(y)
            .filter(|&(&p, &t)| p as f64 == t)
            .count();
        correct as f64 / y.len() as f64
    }

    fn uniform(n: usize) -> Vec<f64> {
        vec![1.0 / n as f64; n]
    }

    #[test]
    fn the_gradient_matches_finite_differences() {
        let (sample, _) = disc();
        let learner = TwoLayerNetwork::new(3).iterations(1);
        for activation in [Activation::Tanh, Activation::Softplus, Activation::Relu] {
            let network = learner
                .clone()
                .activation(activation)
                .max_retries(0)
                .min_edge(0.0)
                .produce(&sample, &uniform(81));
            let z = [0.3, -1.2];
            let mut hidden = vec![0.0; 3];
            let mut grad = vec![0.0; flatten(&network).len()];
            network.score_of_standardized(&z, &mut hidden);
            accumulate_gradient(&network, &z, &hidden, 1.0, &mut grad);
            let theta = flatten(&network);
            let score = |theta: &[f64]| {
                let mut copy = network.clone();
                unflatten(&mut copy, theta);
                copy.score_of_standardized(&z, &mut [0.0; 3])
            };
            // Central differences: error O(e²) plus rounding 1e-16 / e.
            let e = 1e-6;
            for k in 0..theta.len() {
                let (mut up, mut down) = (theta.clone(), theta.clone());
                up[k] += e;
                down[k] -= e;
                let numeric = (score(&up) - score(&down)) / (2.0 * e);
                assert!(
                    (numeric - grad[k]).abs() < 1e-6,
                    "{activation}, parameter {k}: {numeric} vs {}",
                    grad[k]
                );
            }
        }
    }

    #[test]
    fn loss_derivatives_match_finite_differences() {
        let e = 1e-6;
        for loss in [NetworkLoss::Logistic, NetworkLoss::Squared] {
            for (y, s) in [(1.0, 0.3), (-1.0, 0.3), (1.0, -2.0), (-1.0, 5.0)] {
                let numeric = (loss.value(y, s + e) - loss.value(y, s - e)) / (2.0 * e);
                assert!((numeric - loss.derivative(y, s)).abs() < 1e-7, "{loss:?}");
            }
        }
    }

    #[test]
    fn learns_a_nonlinear_boundary_with_each_optimizer_and_batching() {
        let (sample, y) = disc();
        for learner in [
            TwoLayerNetwork::new(8).iterations(400),
            TwoLayerNetwork::new(8)
                .iterations(400)
                .optimizer(AdaptiveOptimizer::Adam(Adam::new(0.01))),
            TwoLayerNetwork::new(8).iterations(600).batch_size(16),
            TwoLayerNetwork::new(8)
                .iterations(400)
                .activation(Activation::Relu),
            TwoLayerNetwork::new(8)
                .iterations(400)
                .loss(NetworkLoss::Squared),
        ] {
            let h = learner.produce(&sample, &uniform(81));
            // A halfspace is right on at most 64 of the 81 rows (the
            // negative label); the disc needs a nonlinear boundary.
            assert!(accuracy(&h, &sample, &y) >= 0.9, "{:?}", learner.info());
        }
    }

    #[test]
    fn a_call_depends_on_the_seed_and_the_distribution_only() {
        let (sample, _) = disc();
        let learner = TwoLayerNetwork::new(4).iterations(20).seed(5);
        let d = uniform(81);
        let mut other = d.clone();
        other[0] *= 2.0;
        let h = learner.produce(&sample, &d);
        assert_eq!(h, learner.clone().produce(&sample, &d));
        assert_ne!(h, learner.produce(&sample, &other));
        assert_ne!(h, learner.clone().seed(6).produce(&sample, &d));
        // Scaling the distribution changes the stream but not the problem;
        // zero-weight rows do not enter the standardization.
        let mut sparse_dist = vec![0.0; 81];
        sparse_dist[..40].fill(1.0);
        let h = learner.produce(&sample, &sparse_dist);
        let (shift, _) = h.standardization();
        let mean: f64 = (0..40).map(|i| sample["feature_0"][i]).sum::<f64>() / 40.0;
        assert!((shift[0] - mean).abs() < 1e-12);
    }

    #[test]
    fn the_edge_is_made_nonnegative_and_shortfalls_are_counted() {
        let (sample, y) = disc();
        // Weight only the 13 positive rows and one negative row, so that a
        // poorly trained network could have either sign.
        let mut dist = vec![0.0; 81];
        for (i, &label) in y.iter().enumerate() {
            if label > 0.0 {
                dist[i] = 1.0;
            }
        }
        dist[0] = 1.0;
        let learner = TwoLayerNetwork::new(2).iterations(1).min_edge(0.0);
        for seed in 0..20 {
            let h = learner.clone().seed(seed).produce(&sample, &dist);
            let edge: f64 = h
                .confidence_all(&sample)
                .iter()
                .zip(&y)
                .zip(&dist)
                .map(|((c, t), d)| c * t * d)
                .sum();
            assert!(edge >= 0.0, "seed {seed}: {edge}");
        }
        assert_eq!(learner.shortfalls(), 0);

        // Opposite labels on identical inputs: every edge is 0, so all four
        // attempts run and the call is a shortfall.
        let rows = vec![vec![1.0, 2.0], vec![1.0, 2.0]];
        let tie = Sample::from_dense(&rows, Some(&[-1.0, 1.0])).unwrap();
        let learner = TwoLayerNetwork::new(2).iterations(3);
        let h = learner.produce(&tie, &[0.5, 0.5]);
        assert_eq!(learner.shortfalls(), 1);
        assert!(h.confidence_all(&tie).iter().all(|c| c.is_finite()));
    }

    #[test]
    fn predictions_agree_row_by_row_and_on_sparse_features() {
        let (sample, _) = disc();
        let h = TwoLayerNetwork::new(5)
            .iterations(30)
            .produce(&sample, &uniform(81));
        let all = h.confidence_all(&sample);
        for row in [0, 40, 80] {
            assert_eq!(all[row].to_bits(), h.confidence(&sample, row).to_bits());
        }
        assert!(all.iter().all(|c| c.abs() <= 1.0));
        // The same data with sparse columns trains the same network.
        let columns: Vec<Vec<f64>> = (0..2)
            .map(|k| {
                (0..81)
                    .map(|i| sample[&format!("feature_{k}")][i])
                    .collect()
            })
            .collect();
        let mut indptr = vec![0];
        let (mut indices, mut data) = (Vec::new(), Vec::new());
        for column in &columns {
            for (i, &v) in column.iter().enumerate() {
                if v != 0.0 {
                    indices.push(i);
                    data.push(v);
                }
            }
            indptr.push(indices.len());
        }
        let sparse = Sample::from_csc(81, &indptr, &indices, &data, Some(sample.target())).unwrap();
        let g = TwoLayerNetwork::new(5)
            .iterations(30)
            .produce(&sparse, &uniform(81));
        assert_eq!(h, g);
        assert_eq!(g.confidence_all(&sparse), all);
    }

    #[test]
    fn sign_output_and_serialization() {
        let (sample, _) = disc();
        let h = TwoLayerNetwork::new(4)
            .iterations(50)
            .output(NetworkOutput::Sign)
            .produce(&sample, &uniform(81));
        assert!(
            h.confidence_all(&sample)
                .iter()
                .all(|&c| c == 1.0 || c == -1.0)
        );
        let json = serde_json::to_string(&h).unwrap();
        let back: TwoLayerNetworkClassifier = serde_json::from_str(&json).unwrap();
        assert_eq!(back.confidence_all(&sample), h.confidence_all(&sample));
    }

    #[test]
    fn boosting_small_networks_beats_one_of_them() {
        use crate::prelude::*;
        let (sample, y) = disc();
        // Two hidden units cannot draw a disc; boosting combines them.
        let learner = TwoLayerNetwork::new(2).iterations(100);
        let single = accuracy(&learner.produce(&sample, &uniform(81)), &sample, &y);
        let f = AdaBoost::init(&sample).force_quit_at(20).run(&learner);
        let correct = f
            .predict_all(&sample)
            .iter()
            .zip(&y)
            .filter(|&(&p, &t)| p as f64 == t)
            .count();
        assert!(correct as f64 / 81.0 > single, "{correct} vs {single}");
        assert!(correct >= 75, "{correct}");
    }

    #[test]
    fn invalid_inputs_panic() {
        let (sample, _) = disc();
        let learner = TwoLayerNetwork::new(2).iterations(2);
        for dist in [vec![0.0; 81], vec![1.0; 80], {
            let mut d = vec![1.0; 81];
            d[3] = -1.0;
            d
        }] {
            assert!(std::panic::catch_unwind(|| learner.produce(&sample, &dist)).is_err());
        }
        let rows = vec![vec![0.0], vec![1.0]];
        let zero_one = Sample::from_dense(&rows, Some(&[0.0, 1.0])).unwrap();
        assert!(std::panic::catch_unwind(|| learner.produce(&zero_one, &[0.5, 0.5])).is_err());
        assert!(std::panic::catch_unwind(|| TwoLayerNetwork::new(0)).is_err());
        assert!(std::panic::catch_unwind(|| TwoLayerNetwork::new(2).iterations(0)).is_err());
        assert!(std::panic::catch_unwind(|| TwoLayerNetwork::new(2).batch_size(0)).is_err());
        assert!(std::panic::catch_unwind(|| TwoLayerNetwork::new(2).regularization(-1.0)).is_err());
        assert!(std::panic::catch_unwind(|| TwoLayerNetwork::new(2).min_edge(1.5)).is_err());
    }
}
