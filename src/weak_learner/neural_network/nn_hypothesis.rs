use crate::{Sample, Classifier};
use crate::common::{
    task,
    utils,
    task::Task,
};
use super::{
    layer::*,
    nn_loss::*,
    activation::*,
};

/// 2-layered neural network.
/// ```text
///          O
///  O
///          O        O
///  O
///          O        O
///  O
///          O
/// L0      L1       L2
/// ```
/// # Layer 1
/// Computes `z = activation_1(u)` for `u = Wx + b`.
/// Here, `x` is an `n`-dimensional vector,
/// `W` is a matrix of size `kxn`,
/// and `b` is an `k`-dimensional vector.
/// # Layer 2
/// Computes `activation_2(v)` for `v = W'z + b'`.
/// Here, `z` is a `k`-dimensional vector,
/// `W'` is a matrix of size `2xk`,
/// and `b` is a `2`-dimensional vector.
pub struct NNHypothesis {
    task: Task,
    layers: Vec<Layer>,
}


impl NNHypothesis {
    pub(crate) fn new<S, T>(
        task: Task,
        dimensions: S,
        activations: T,
    ) -> Self
        where S: AsRef<[usize]>,
              T: AsRef<[Activation]>,
    {
        let dimensions = dimensions.as_ref();
        let activations = activations.as_ref();
        assert!(!dimensions.is_empty());
        let n_layers = dimensions.len();

        let mut iter = dimensions.into_iter();
        let mut input_size = iter.next().unwrap();

        let mut iter = iter.zip(activations);

        let mut layers = Vec::with_capacity(n_layers);
        while let Some((output_size, act)) = iter.next() {
            let layer = Layer::new(*output_size, *input_size, *act);
            layers.push(layer);
            input_size = output_size;
        }

        Self { task, layers }
    }


    /// Evaluate the given data.
    pub(crate) fn eval<T>(&self, x: T) -> Vec<f64>
        where T: ToOwned<Owned = Vec<f64>>
    {
        let x = x.to_owned();
        self.layers.iter()
            .fold(x, |z, layer| layer.forward(z))
    }


    /// Prints stats of this network.
    pub fn stats(&self) {
        println!("Stats");
        println!("----------------");
        for (l, layer) in self.layers.iter().enumerate() {
            let (nrow, ncol) = layer.shape();
            let act = layer.activation;
            println!(
                "\t[Layer {k: >3}] \
                [input: {ncol: >7}]\t\
                [output: {nrow: >7}]\t\
                [activation: {act:?}]",
                k = l + 1
            );
        }
        println!("----------------");
    }


    fn output_dim(&self) -> usize {
        match self.layers.last() {
            Some(layer) => layer.output_dim(),
            None => {
                panic!("0-layerd neural network does not have output!");
            },
        }
    }


    /// Perform a gradient descent for the given mini-batch.
    pub(crate) fn train<T: AsRef<[usize]>>(
        &mut self,
        learning_rate: f64,
        loss_func: NNLoss,
        sample: &Sample,
        indices: T,
    )
    {
        let indices = indices.as_ref();
        let batch_size = indices.len();
        let n_layers = self.layers.len();
        // Keep the inputs for activation `f[l]`
        // of each layer for back propagation.
        let mut inputs = vec![Vec::with_capacity(batch_size); n_layers];
        // Keep the outputs for activation `f[l]`
        // of each layer for back propagation.
        let mut outputs = vec![Vec::with_capacity(batch_size); n_layers];

        // The `i`-th **row** of `batch_delta` corresponds to
        // the `delta` at `i`-th example `(xi, yi)`.
        let mut batch_delta = Vec::with_capacity(batch_size);

        // Compute the `delta` for the output layer.
        let dim = self.output_dim();
        for &i in indices {
            let (x, y) = sample.at(i);
            // Vectorize the target value to compute `delta`.
            let y = task::vectorize(y, dim);

            // Forward propergation
            let final_output = self.layers.iter()
                .enumerate()
                .fold(x, |z, (l, layer)| {
                    outputs[l].push(z.clone());
                    // Linear transformation: `u = Wx + b`
                    let u = layer.affine(&z);
                    // Nonlinear transformation: `z = f(u)`
                    let z = layer.nonlinear(&u);
                    inputs[l].push(u);

                    z
                });

            // Compute the `delta` for the last layer.
            let delta = loss_func.diff(final_output, y);
            batch_delta.push(delta);
        }

        // An iterator over outputs and inputs
        let out_in_iter = outputs.into_iter().zip(inputs);
        let mut iter = self.layers.iter_mut()
            .zip(out_in_iter)
            .rev()
            .peekable();


        let mut delta = batch_delta;
        while let Some((layer, (output, _))) = iter.next() {
            let weights = &layer.matrix[..];
            let delta_x_weights = matrix_product(&delta, &weights);

            // Compute the descent directions
            // `layer.matrix`, `layer.bias`.

            let dw = matrix_inner_product(&delta, &output);
            let db = column_sum(&delta);
            layer.backward(learning_rate, dw, db);


            if let Some((prev_layer, (_, prev_input))) = iter.peek() {
                let df = prev_input.iter()
                    .map(|x| prev_layer.activation.diff(x))
                    .collect::<Vec<_>>();

                delta = utils::hadamard_product(df, delta_x_weights);
            }
        }
    }
}


impl Classifier for NNHypothesis {
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        let (x, _) = sample.at(row);

        let conf = self.eval(x);
        match self.task {
            Task::Binary => task::binarize(conf),
            Task::MultiClass => todo!(),
            Task::Regression => todo!(),
        }
    }
}
