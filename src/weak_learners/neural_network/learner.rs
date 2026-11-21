//! The weak learner [`TwoLayerNetwork`].

use super::activation::Activation;
use super::network::{
    NetworkLoss, NetworkOutput, TwoLayerNetworkClassifier, accumulate_gradient, flatten, raw_rows,
    unflatten,
};
use crate::core::{Sample, WeakLearner};
use crate::optimization::adaptive::{AdaptiveOptimizer, AdaptiveState, Adopt};
use crate::optimization::sparsification::SplitMix64;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Trains a network with one hidden layer on the weighted sample.
///
/// # Algorithm
/// Given a distribution `d` over the sample, `produce`
/// 1. keeps the rows with `d_i > 0` and normalizes their weights to
///    `p_i = d_i / Σ_j d_j`;
/// 2. standardizes every feature by its `p`-weighted mean and standard
///    deviation over those rows (a constant feature becomes `0`);
/// 3. initializes the weights from a seeded stream (Glorot for `tanh`, He
///    for ReLU and softplus; see [`Activation`]) and the biases at `0`;
/// 4. minimizes
///    ```txt
///    F(θ) = Σ_i p_i ℓ(y_i, s_θ(x_i)) + (λ/2) (‖W‖² + ‖w‖²)
///    ```
///    with [`iterations`](TwoLayerNetwork::iterations) updates of
///    [`Adam`](crate::Adam) or [`Adopt`] (the biases are not penalized).
///    Each update uses the gradient of `F` on all `n` rows, or with
///    [`batch_size`](TwoLayerNetwork::batch_size) `b`, the unbiased estimate
///    `(n/b) Σ_{r=1}^{b} p_{i_r} ∇ℓ(y_{i_r}, s_θ(x_{i_r})) + λ (W, w)` from
///    `b` rows `i_r` drawn uniformly **with** replacement;
/// 5. computes the edge `γ = Σ_i p_i y_i h(x_i)` of the result `h` and, if
///    `γ < 0`, negates the output layer, which negates `h` (`tanh` is odd);
/// 6. if the edge is below [`min_edge`](TwoLayerNetwork::min_edge), trains
///    again from a new initialization, at most
///    [`max_retries`](TwoLayerNetwork::max_retries) times, and returns the
///    network with the largest edge.
///
/// Steps 5 and 6 are this crate's additions: they select among trained
/// networks by their edge and are not part of the minimization of `F`.
///
/// # Reproducibility
/// The random stream of a call is seeded by
/// [`seed`](TwoLayerNetwork::seed) and every entry of `d`, and the retries
/// draw their seeds from that stream. A call therefore depends only on its
/// inputs and the settings, not on the order of calls or on threads, and
/// different distributions get different streams. All sums run
/// sequentially in a fixed order.
///
/// # Guarantee
/// None for the weak-learning assumption: the edge after step 5 is at least
/// `0`, but it can be below `min_edge` (counted by
/// [`shortfalls`](TwoLayerNetwork::shortfalls)). With a smooth activation
/// and full-batch or with-replacement minibatch gradients, the assumptions
/// of the ADOPT analysis hold for `F` except the bounded-gradient condition,
/// which bounded weights would give; the analysis gives convergence to a
/// stationary point of `F` for a step size proportional to
/// `1/sqrt(iterations)` (see [`Adopt`]), not a minimizer of `F`, and nothing
/// about the edge.
///
/// # Limitations
/// - Training costs `O(iterations · b · h · p)` per attempt for `h` hidden
///   units and `p` features (`b = n` for full batches), far more than a
///   decision stump; every boosting round trains a new network.
/// - The result depends on the learning rate, the number of iterations, and
///   the initialization; the defaults are not tuned for any data set.
/// - Sparse features are read into dense rows.
/// - Labels must be `-1` or `+1`.
///
/// # Examples
/// ```
/// use miniboosts::{Classifier, Sample, prelude::*};
///
/// // Positive inside the disc of radius 1.5, negative outside: no
/// // halfspace separates the classes.
/// let mut rows = Vec::new();
/// let mut y = Vec::new();
/// for i in -4..=4 {
///     for j in -4..=4 {
///         let (a, b) = (f64::from(i) * 0.6, f64::from(j) * 0.6);
///         rows.push(vec![a, b]);
///         y.push(if a * a + b * b < 2.25 { 1.0 } else { -1.0 });
///     }
/// }
/// let sample = Sample::from_dense(&rows, Some(&y)).unwrap();
/// let n = y.len();
///
/// let learner = TwoLayerNetwork::new(8).iterations(400).seed(3);
/// let h = learner.produce(&sample, &vec![1.0 / n as f64; n]);
/// let correct = h.predict_all(&sample).iter().zip(&y).filter(|&(&p, &t)| p as f64 == t).count();
/// assert!(correct as f64 >= 0.9 * n as f64, "{correct} of {n}");
///
/// // The same inputs give the same network.
/// assert_eq!(h, learner.produce(&sample, &vec![1.0 / n as f64; n]));
/// ```
#[derive(Debug)]
pub struct TwoLayerNetwork {
    hidden_units: usize,
    activation: Activation,
    loss: NetworkLoss,
    output: NetworkOutput,
    optimizer: AdaptiveOptimizer,
    iterations: usize,
    batch_size: Option<usize>,
    regularization: f64,
    seed: u64,
    min_edge: f64,
    max_retries: usize,
    shortfalls: AtomicUsize,
}

impl Clone for TwoLayerNetwork {
    fn clone(&self) -> Self {
        Self {
            shortfalls: AtomicUsize::new(self.shortfalls()),
            ..*self
        }
    }
}

/// Default number of updates per attempt.
pub const DEFAULT_ITERATIONS: usize = 300;

/// Default learning rate of the default optimizer, ADOPT. Chosen on a small
/// two-dimensional example (a disc in a 9 × 9 grid, 8 hidden units), where
/// 0.01 needed about 1000 updates to fit and 0.1 needed 300; it is not tuned
/// for any data set, and ADOPT's own default is 1e-3.
pub const DEFAULT_LEARNING_RATE: f64 = 0.1;

impl TwoLayerNetwork {
    /// Constructs the learner with `hidden_units` hidden units and the
    /// defaults: `tanh` activation, logistic loss, `tanh` output, ADOPT with
    /// learning rate [`DEFAULT_LEARNING_RATE`], [`DEFAULT_ITERATIONS`]
    /// full-batch updates, `λ = 1e-4`, seed `0`, minimum edge `0.01`, and at
    /// most 3 retries.
    ///
    /// # Panics
    /// Panics if `hidden_units == 0`.
    ///
    /// # Examples
    /// See [`TwoLayerNetwork`].
    pub fn new(hidden_units: usize) -> Self {
        assert!(hidden_units > 0, "the hidden layer needs at least one unit");
        Self {
            hidden_units,
            activation: Activation::Tanh,
            loss: NetworkLoss::Logistic,
            output: NetworkOutput::Tanh,
            optimizer: AdaptiveOptimizer::Adopt(Adopt::new(DEFAULT_LEARNING_RATE)),
            iterations: DEFAULT_ITERATIONS,
            batch_size: None,
            regularization: 1e-4,
            seed: 0,
            min_edge: 0.01,
            max_retries: 3,
            shortfalls: AtomicUsize::new(0),
        }
    }

    /// Sets the activation function of the hidden layer.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::prelude::*;
    /// use miniboosts::weak_learners::neural_network::Activation;
    ///
    /// let learner = TwoLayerNetwork::new(4).activation(Activation::Relu);
    /// ```
    #[must_use]
    pub fn activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Sets the loss `ℓ`. Default is [`NetworkLoss::Logistic`].
    ///
    /// # Examples
    /// ```
    /// use miniboosts::prelude::*;
    /// use miniboosts::weak_learners::neural_network::NetworkLoss;
    ///
    /// let learner = TwoLayerNetwork::new(4).loss(NetworkLoss::Squared);
    /// ```
    #[must_use]
    pub fn loss(mut self, loss: NetworkLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Sets the output map. Default is [`NetworkOutput::Tanh`].
    ///
    /// # Examples
    /// ```
    /// use miniboosts::prelude::*;
    /// use miniboosts::weak_learners::neural_network::NetworkOutput;
    ///
    /// let learner = TwoLayerNetwork::new(4).output(NetworkOutput::Sign);
    /// ```
    #[must_use]
    pub fn output(mut self, output: NetworkOutput) -> Self {
        self.output = output;
        self
    }

    /// Sets the optimizer and its learning rate.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::prelude::*;
    /// use miniboosts::{Adam, AdaptiveOptimizer};
    ///
    /// let learner = TwoLayerNetwork::new(4).optimizer(AdaptiveOptimizer::Adam(Adam::new(0.01)));
    /// ```
    #[must_use]
    pub fn optimizer(mut self, optimizer: AdaptiveOptimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Sets the number of parameter updates per attempt. ADOPT computes one
    /// more gradient than this, to initialize its second moment.
    ///
    /// # Panics
    /// Panics if `iterations == 0`.
    ///
    /// # Examples
    /// See [`TwoLayerNetwork`].
    #[must_use]
    pub fn iterations(mut self, iterations: usize) -> Self {
        assert!(iterations > 0, "at least one update is needed");
        self.iterations = iterations;
        self
    }

    /// Estimates each gradient from `size` rows drawn uniformly with
    /// replacement instead of all rows. Rows drawn with replacement keep the
    /// estimate unbiased, as the ADOPT analysis assumes.
    ///
    /// # Panics
    /// Panics if `size == 0`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::prelude::*;
    ///
    /// let learner = TwoLayerNetwork::new(4).batch_size(32);
    /// ```
    #[must_use]
    pub fn batch_size(mut self, size: usize) -> Self {
        assert!(size > 0, "a minibatch needs at least one row");
        self.batch_size = Some(size);
        self
    }

    /// Sets the weight `λ` of the penalty `(λ/2)(‖W‖² + ‖w‖²)`. Default is
    /// `1e-4`.
    ///
    /// # Panics
    /// Panics unless `regularization` is finite and nonnegative.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::prelude::*;
    ///
    /// let learner = TwoLayerNetwork::new(4).regularization(0.0);
    /// ```
    #[must_use]
    pub fn regularization(mut self, regularization: f64) -> Self {
        assert!(
            regularization.is_finite() && regularization >= 0.0,
            "the regularization must be finite and nonnegative"
        );
        self.regularization = regularization;
        self
    }

    /// Sets the seed that, together with the distribution, determines the
    /// random stream of a call. Default is `0`.
    ///
    /// # Examples
    /// See [`TwoLayerNetwork`].
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the edge below which a network is trained again. Default is
    /// `0.01`.
    ///
    /// # Panics
    /// Panics unless `0 ≤ min_edge ≤ 1`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::prelude::*;
    ///
    /// let learner = TwoLayerNetwork::new(4).min_edge(0.05).max_retries(5);
    /// ```
    #[must_use]
    pub fn min_edge(mut self, min_edge: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&min_edge),
            "the minimum edge must lie in [0, 1]"
        );
        self.min_edge = min_edge;
        self
    }

    /// Sets the largest number of retries after the first attempt. Default
    /// is `3`; `0` trains once.
    ///
    /// # Examples
    /// See [`TwoLayerNetwork::min_edge`].
    #[must_use]
    pub fn max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Returns the number of calls to `produce` whose best network had an
    /// edge below [`min_edge`](TwoLayerNetwork::min_edge).
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{Sample, prelude::*};
    ///
    /// // Identical inputs with opposite labels: every edge is 0.
    /// let rows = vec![vec![1.0], vec![1.0]];
    /// let sample = Sample::from_dense(&rows, Some(&[-1.0, 1.0])).unwrap();
    /// let learner = TwoLayerNetwork::new(2).iterations(5).max_retries(1);
    /// let _ = learner.produce(&sample, &[0.5, 0.5]);
    /// assert_eq!(learner.shortfalls(), 1);
    /// ```
    pub fn shortfalls(&self) -> usize {
        self.shortfalls.load(Ordering::Relaxed)
    }

    /// Trains one network from the seed `seed` on the standardized rows `z`
    /// with labels `y` and weights `p`.
    fn train(
        &self,
        names: &std::sync::Arc<[String]>,
        (shift, scale): (&[f64], &[f64]),
        z: &[Vec<f64>],
        y: &[f64],
        p: &[f64],
        seed: u64,
    ) -> TwoLayerNetworkClassifier {
        let n = z.len();
        let dim = shift.len();
        let h = self.hidden_units;
        let mut rng = SplitMix64::new(seed);
        let mut uniform = |bound: f64| (2.0 * rng.uniform() - 1.0) * bound;
        let hidden_bound = self.activation.init_bound(dim, h);
        let hidden_weights = (0..h * dim).map(|_| uniform(hidden_bound)).collect();
        // The output unit is followed by `tanh`.
        let output_bound = Activation::Tanh.init_bound(h, 1);
        let output_weights = (0..h).map(|_| uniform(output_bound)).collect();
        let mut network = TwoLayerNetworkClassifier {
            feature_names: std::sync::Arc::clone(names),
            shift: shift.to_vec(),
            scale: scale.to_vec(),
            hidden_weights,
            hidden_bias: vec![0.0; h],
            output_weights,
            output_bias: 0.0,
            activation: self.activation,
            output: self.output,
        };

        let mut theta = flatten(&network);
        let n_weights = h * dim;
        let mut state = AdaptiveState::new(self.optimizer, theta.len());
        let mut hidden = vec![0.0; h];
        let mut batch = Vec::new();
        while state.updates() < self.iterations as u64 {
            let mut grad = vec![0.0; theta.len()];
            let mut add = |i: usize, factor: f64, grad: &mut [f64]| {
                let s = network.score_of_standardized(&z[i], &mut hidden);
                let coef = factor * p[i] * self.loss.derivative(y[i], s);
                accumulate_gradient(&network, &z[i], &hidden, coef, grad);
            };
            match self.batch_size {
                None => (0..n).for_each(|i| add(i, 1.0, &mut grad)),
                Some(b) => {
                    batch.clear();
                    batch.extend((0..b).map(|_| ((rng.uniform() * n as f64) as usize).min(n - 1)));
                    let factor = n as f64 / b as f64;
                    batch.iter().for_each(|&i| add(i, factor, &mut grad));
                }
            }
            // The penalty on W (first `h·p` entries) and w (entries after
            // the hidden biases); the biases are not penalized.
            let output_start = n_weights + h;
            for k in (0..n_weights).chain(output_start..output_start + h) {
                grad[k] += self.regularization * theta[k];
            }
            assert!(
                grad.iter().all(|g| g.is_finite()),
                "the network's gradient is not finite; lower the learning rate"
            );
            state.step(&mut theta, &grad);
            unflatten(&mut network, &theta);
        }
        network
    }
}

impl WeakLearner for TwoLayerNetwork {
    type Hypothesis = TwoLayerNetworkClassifier;

    fn name(&self) -> &str {
        "Two-layer network"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let batch = self
            .batch_size
            .map_or_else(|| "full".to_string(), |b| b.to_string());
        Some(vec![
            ("Hidden units", self.hidden_units.to_string()),
            ("Activation", self.activation.to_string()),
            ("Loss", format!("{:?}", self.loss)),
            ("Output", format!("{:?}", self.output)),
            ("Optimizer", self.optimizer.name().to_string()),
            ("Learning rate", self.optimizer.learning_rate().to_string()),
            ("Iterations", self.iterations.to_string()),
            ("Batch size", batch),
            ("Regularization", self.regularization.to_string()),
            ("Seed", self.seed.to_string()),
            ("Min edge", self.min_edge.to_string()),
            ("Max retries", self.max_retries.to_string()),
        ])
    }

    /// Trains a network on the weighted sample; see [`TwoLayerNetwork`].
    ///
    /// # Panics
    /// Panics if the sample has no labels or a label other than `±1`, if
    /// `dist` has the wrong length, a negative or non-finite entry, or zero
    /// total, or if training produces a non-finite gradient.
    fn produce(&self, sample: &Sample, dist: &[f64]) -> Self::Hypothesis {
        let n_sample = sample.shape().0;
        let target = sample.target();
        assert_eq!(target.len(), n_sample, "the sample has no target");
        assert!(
            target.iter().all(|&y| y == 1.0 || y == -1.0),
            "labels must be -1 or +1"
        );
        assert_eq!(
            dist.len(),
            n_sample,
            "the distribution has {} entries for {n_sample} examples",
            dist.len()
        );
        assert!(
            dist.iter().all(|d| d.is_finite() && *d >= 0.0),
            "the distribution has a negative or non-finite entry"
        );
        let active: Vec<usize> = (0..n_sample).filter(|&i| dist[i] > 0.0).collect();
        assert!(!active.is_empty(), "the distribution has zero total");
        let total: f64 = active.iter().map(|&i| dist[i]).sum();
        let p: Vec<f64> = active.iter().map(|&i| dist[i] / total).collect();
        let y: Vec<f64> = active.iter().map(|&i| target[i]).collect();

        let names = sample.shared_names();
        let rows = raw_rows(&sample.columns(&names), n_sample, Some(&active));
        let dim = names.len();
        let mut shift = vec![0.0; dim];
        let mut scale = vec![1.0; dim];
        for k in 0..dim {
            let mean: f64 = rows.iter().zip(&p).map(|(x, w)| w * x[k]).sum();
            let variance: f64 = rows
                .iter()
                .zip(&p)
                .map(|(x, w)| w * (x[k] - mean).powi(2))
                .sum();
            shift[k] = mean;
            if variance > 0.0 {
                scale[k] = 1.0 / variance.sqrt();
            }
        }
        let z: Vec<Vec<f64>> = rows
            .iter()
            .map(|x| {
                x.iter()
                    .zip(shift.iter().zip(&scale))
                    .map(|(v, (m, s))| (v - m) * s)
                    .collect()
            })
            .collect();

        let edge = |network: &TwoLayerNetworkClassifier| {
            let mut hidden = vec![0.0; self.hidden_units];
            z.iter()
                .zip(&y)
                .zip(&p)
                .map(|((x, yi), w)| {
                    w * yi * network.link(network.score_of_standardized(x, &mut hidden))
                })
                .sum::<f64>()
        };
        let mut seeds = SplitMix64::from_seed_and_weights(self.seed, dist);
        let mut best: Option<(f64, TwoLayerNetworkClassifier)> = None;
        for _ in 0..=self.max_retries {
            let mut network = self.train(&names, (&shift, &scale), &z, &y, &p, seeds.next_u64());
            let mut gamma = edge(&network);
            if gamma < 0.0 {
                network.negate();
                gamma = edge(&network);
            }
            if best.as_ref().is_none_or(|(g, _)| gamma > *g) {
                best = Some((gamma, network));
            }
            if gamma >= self.min_edge {
                break;
            }
        }
        let (gamma, network) = best.expect("at least one attempt runs");
        if gamma < self.min_edge {
            self.shortfalls.fetch_add(1, Ordering::Relaxed);
        }
        network
    }
}
