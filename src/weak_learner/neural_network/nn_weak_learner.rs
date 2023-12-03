use crate::{
    Sample,
    WeakLearner,
};


use crate::common::task::Task;


use super::{
    nn_loss::*,
    activation::*,
    nn_hypothesis::*,
};

use rand;
use rand::seq::index;

const N_EPOCH: usize = 100;
const N_ITER: usize = 200;
const LEARNING_RATE: f64 = 1e-3;
const MINIBATCH_SIZE: usize = 128;


type OutputDim = usize;

/// A neural network weak learner.
/// Since this is just a weak learner, a shallow network is preferred.
/// Of course, you can use a deep network 
/// if you don't care about running time.
/// # Example
/// ```no_run
/// use miniboosts::prelude::*;
/// 
/// // Read a dataset from a CSV file.
/// let path = "/path/to/dataset.csv";
/// let has_header = true;
/// let sample = Sample::from_csv(path, has_header)
///     .unwrap()
///     .set_target("class");
/// let n_sample = sample.shape().0;
/// let batch_size = n_sample / 10;
/// 
/// // Construct a new instance of a neural network learner.
/// let nn = NeuralNetwork::init(&sample)
///     .append(100, Activation::ReLu(1.0))
///     .append(2, Activation::SoftMax(1.0))
///     .n_epoch(10)
///     .n_iter(100)
///     .minibatch_size(batch_size);
/// 
/// // Construct the uniform distribution over examples.
/// let dist = vec![1.0 / n_sample as f64; n_sample];
/// 
/// // Train a neural network and obtain a hypothesis.
/// let f = nn.produce(&sample, &dist[..]);
/// 
/// // Train loss
/// let predictions = f.predict_all(&sample);
/// let loss = sample.target()
///     .into_iter()
///     .zip(predictions)
///     .map(|(t, p)| if *t != p as f64 { 1.0 } else { 0.0 })
///     .sum::<f64>() / n_sample as f64;
/// println!("0/1-loss (train): {loss}");
/// ```
pub struct NeuralNetwork {
    task: Task,
    learning_rate: f64,
    minibatch_size: usize,
    dimensions: Vec<OutputDim>,
    activations: Vec<Activation>,
    loss_func: NNLoss,
    n_epoch: usize,
    n_iter_per_epoch: usize,
}



impl NeuralNetwork {
    /// Construct a new instance of `NeuralNetwork`.
    /// At the initial state, `NeuralNetwork` constructs a network of depth 0.
    #[inline(always)]
    pub fn init(sample: &Sample) -> Self {
        let (n_samples, n_features) = sample.shape();

        let task = Task::Binary;
        let learning_rate = LEARNING_RATE;
        let mut minibatch_size = MINIBATCH_SIZE;

        // Halve the given mini-batch size
        // until it become smaller than sample size.
        while minibatch_size > n_samples {
            minibatch_size /= 2;
        }

        let dimensions = vec![n_features];
        let activations = Vec::new();

        let n_epoch = N_EPOCH;
        let n_iter_per_epoch = N_ITER;
        let loss_func = NNLoss::CrossEntropy;

        Self {
            task,
            learning_rate,
            minibatch_size,
            dimensions,
            activations,
            n_epoch,
            n_iter_per_epoch,
            loss_func,
        }
    }


    /// Append a new layer to the current network.
    #[inline(always)]
    pub fn append(
        mut self,
        dim: OutputDim,
        activation: Activation
    ) -> Self
    {
        self.dimensions.push(dim);
        self.activations.push(activation);

        self
    }


    /// Set the number of epochs
    #[inline(always)]
    pub fn n_epoch(mut self, epoch: usize) -> Self {
        self.n_epoch = epoch;
        self
    }


    /// Set the number of iterations per epoch
    #[inline(always)]
    pub fn n_iter(mut self, iter: usize) -> Self {
        self.n_iter_per_epoch = iter;
        self
    }


    /// Set the mini-batch size
    #[inline(always)]
    pub fn minibatch_size(mut self, batch_size: usize) -> Self {
        self.minibatch_size = batch_size;
        self
    }


    /// Set the task.
    /// Currently, Binary classification is available.
    #[inline(always)]
    pub fn task(mut self, task: Task) -> Self {
        self.task = task;
        self
    }
}


impl WeakLearner for NeuralNetwork {
    type Hypothesis = NNClassifier;

    fn name(&self) -> &str {
        "Neural Network"
    }


    fn info(&self) -> Option<Vec<(&str, String)>> {
        let info = Vec::from([
            ("Task", format!("{}", self.task)),
            ("Mini-batch size", format!("{}", self.minibatch_size)),
            ("Rounds per epoch", format!("{}", self.n_iter_per_epoch)),
            ("Learning rate", format!("{}", self.learning_rate)),
            ("# of epochs", format!("{}", self.n_epoch)),
            ("# of layers", format!("{}", self.activations.len())),
            ("Loss", format!("{}", self.loss_func)),
        ]);
        Some(info)
    }


    #[inline]
    fn produce(&self, sample: &Sample, dist: &[f64])
        -> Self::Hypothesis
    {
        let rate = self.learning_rate / self.minibatch_size as f64;
        let n_samples = sample.shape().0;
        let mut f = NNHypothesis::new(
            self.task, &self.dimensions[..], &self.activations[..]
        );
        let weights = |i: usize| dist[i];
        for _ in 1..=self.n_epoch {
            // Randomly chosen indices over training sample
            let mut rng = rand::thread_rng();
            let minibatch = index::sample_weighted(
                &mut rng, n_samples, weights, self.minibatch_size,
            ).unwrap();
            let minibatch = minibatch.into_iter().collect::<Vec<_>>();
            for _ in 1..=self.n_iter_per_epoch {
                f.train(rate, self.loss_func, sample, &minibatch);
            }
        }
        NNClassifier::new(f)
    }
}
