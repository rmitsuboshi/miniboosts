//! Provides [`AdaBoost`](AdaBoost) by Freund & Schapire, 1995.
use polars::prelude::*;
use rayon::prelude::*;


use crate::{
    Booster,
    WeakLearner,
    State,
    Classifier,
    CombinedHypothesis
};

use crate::research::Logger;


/// Defines `AdaBoost`.
/// This struct is based on the book: 
/// [Boosting: Foundations and Algorithms](https://direct.mit.edu/books/oa-monograph/5342/BoostingFoundations-and-Algorithms)
/// by Robert E. Schapire and Yoav Freund.
/// 
/// # Example
/// The following code shows a small example 
/// for running [`AdaBoost`](AdaBoost).  
/// See also:
/// - [`DTree`]
/// - [`DTreeClassifier`]
/// - [`CombinedHypothesis<F>`]
/// - [`DTree::max_depth`]
/// - [`DTree::criterion`]
/// - [`DataFrame`]
/// - [`Series`]
/// - [`DataFrame::shape`]
/// - [`CsvReader`]
/// 
/// [`DTree`]: crate::weak_learner::DTree
/// [`DTreeClassifier`]: crate::weak_learner::DTreeClassifier
/// [`CombinedHypothesis<F>`]: crate::hypothesis::CombinedHypothesis
/// [`DTree::max_depth`]: crate::weak_learner::DTree::max_depth
/// [`DTree::criterion`]: crate::weak_learner::DTree::criterion
/// [`DataFrame`]: polars::prelude::DataFrame
/// [`Series`]: polars::prelude::Series
/// [`DataFrame::shape`]: polars::prelude::DataFrame::shape
/// [`CsvReader`]: polars::prelude::CsvReader
/// 
/// 
/// ```no_run
/// use polars::prelude::*;
/// use miniboosts::prelude::*;
/// 
/// // Read the training data from the CSV file.
/// let mut data = CsvReader::from_path(path_to_csv_file)
///     .unwrap()
///     .has_header(true)
///     .finish()
///     .unwrap();
/// 
/// // Split the column corresponding to labels.
/// let target = data.drop_in_place(class_column_name).unwrap();
/// 
/// // Initialize `AdaBoost` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis whose training error is
/// // less than `0.01` if the traing examples are linearly separable.
/// // Note that the default tolerance parameter is set as `1 / n_sample`,
/// // where `n_sample = data.shape().0` is 
/// // the number of training examples in `data`.
/// let booster = AdaBoost::init(&data, &target)
///     .tolerance(0.01);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTree::init(&data, &target)
///     .max_depth(2)
///     .criterion(Criterion::Edge);
/// 
/// // Run `AdaBoost` and obtain the resulting hypothesis `f`.
/// let f: CombinedHypothesis<DTreeClassifier> = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions: Vec<i64> = f.predict_all(&data);
/// 
/// // Get the number of training examples.
/// let n_sample = data.shape().0 as f64;
/// 
/// // Calculate the training loss.
/// let training_loss = target.i64()
///     .unwrap()
///     .into_iter()
///     .zip(predictions)
///     .map(|(true_label, prediction) {
///         let true_label = true_label.unwrap();
///         if true_label == prediction { 0.0 } else { 1.0 }
///     })
///     .sum::<f64>()
///     / n_sample;
/// 
///
/// println!("Training Loss is: {training_loss}");
/// ```
pub struct AdaBoost<'a, F> {
    // Training data
    data: &'a DataFrame,
    // Correponding label
    target: &'a Series,

    // Distribution on data.
    dist: Vec<f64>,

    // Tolerance parameter
    tolerance: f64,


    // Weights on hypotheses in `classifiers`
    weights: Vec<f64>,

    // Hypohteses obtained by the weak-learner.
    classifiers: Vec<F>,


    // Max iteration until AdaBoost guarantees the optimality.
    max_iter: usize,

    // Terminated iteration.
    // AdaBoost terminates in eary step 
    // if the training set is linearly separable.
    terminated: usize,
}


impl<'a, F> AdaBoost<'a, F> {
    /// Initialize the `AdaBoost`.
    /// This method sets some parameters `AdaBoost` holds.
    pub fn init(data: &'a DataFrame, target: &'a Series) -> Self {
        assert!(!data.is_empty());

        let n_sample = data.shape().0;

        let uni = 1.0 / n_sample as f64;
        AdaBoost {
            dist: vec![uni; n_sample],
            tolerance: 1.0 / (n_sample as f64 + 1.0),

            weights: Vec::new(),
            classifiers: Vec::new(),

            data,
            target,

            max_iter: usize::MAX,

            terminated: usize::MAX,
        }
    }


    /// Returns the maximum iteration
    /// of the `AdaBoost` to find a combined hypothesis
    /// that has error at most `tolerance`.
    /// After the `self.max_loop()` iterations,
    /// `AdaBoost` guarantees zero training error in terms of zero-one loss
    /// if the training examples are linearly separable.
    pub fn max_loop(&self) -> usize {
        let n_sample = self.data.shape().0 as f64;

        (n_sample.ln() / self.tolerance.powi(2)) as usize
    }


    /// Set the tolerance parameter.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }


    /// Returns a weight on the new hypothesis.
    /// `update_params` also updates `self.dist`
    #[inline]
    fn update_params(
        &mut self,
        margins: Vec<f64>,
        edge: f64
    ) -> f64
    {
        let n_sample = self.data.shape().0;


        // Compute the weight on new hypothesis.
        // This is the returned value of this function.
        let weight = ((1.0 + edge) / (1.0 - edge)).ln() / 2.0;


        // To prevent overflow, take the logarithm.
        self.dist.par_iter_mut()
            .zip(margins)
            .for_each(|(d, p)| *d = d.ln() - weight * p);


        // Sort indices by ascending order
        let mut indices = (0..n_sample).into_par_iter()
            .collect::<Vec<usize>>();
        indices.sort_unstable_by(|&i, &j| {
            self.dist[i].partial_cmp(&self.dist[j]).unwrap()
        });


        let mut normalizer = self.dist[indices[0]];
        for i in indices.into_iter().skip(1) {
            let mut a = normalizer;
            let mut b = self.dist[i];
            if a < b {
                std::mem::swap(&mut a, &mut b);
            }

            normalizer = a + (1.0 + (b - a).exp()).ln();
        }



        // Update self.dist
        self.dist.par_iter_mut()
            .for_each(|d| *d = (*d - normalizer).exp());


        weight
    }
}


impl<F> Booster<F> for AdaBoost<'_, F>
    where F: Classifier + Clone,
{
    fn preprocess<W>(
        &mut self,
        _weak_learner: &W,
    )
        where W: WeakLearner<Hypothesis = F>
    {
        // Initialize parameters
        let n_sample = self.data.shape().0;
        let uni = 1.0 / n_sample as f64;
        self.dist = vec![uni; n_sample];

        self.weights = Vec::new();
        self.classifiers = Vec::new();


        self.max_iter = self.max_loop();
    }


    fn boost<W>(
        &mut self,
        weak_learner: &W,
        iteration: usize,
    ) -> State
        where W: WeakLearner<Hypothesis = F>,
    {
        if self.max_iter < iteration {
            return State::Terminate;
        }


        // Get a new hypothesis
        let h = weak_learner.produce(self.data, self.target, &self.dist);


        // Each element in `margins` is the product of
        // the predicted vector and the correct vector
        let margins = self.target.i64()
            .expect("The target class is not an dtype i64")
            .into_iter()
            .enumerate()
            .map(|(i, y)| (y.unwrap() as f64 * h.confidence(self.data, i)))
            .collect::<Vec<f64>>();


        let edge = margins.iter()
            .zip(&self.dist[..])
            .map(|(&yh, &d)| yh * d)
            .sum::<f64>();


        // If `h` predicted all the examples in `sample` correctly,
        // use it as the combined classifier.
        if edge.abs() >= 1.0 {
            self.terminated = iteration;
            self.weights = vec![edge.signum()];
            self.classifiers = vec![h];
            return State::Terminate;
        }


        // Compute the weight on the new hypothesis
        let weight = self.update_params(margins, edge);
        self.weights.push(weight);
        self.classifiers.push(h);

        State::Continue
    }


    fn postprocess<W>(
        &mut self,
        _weak_learner: &W,
    ) -> CombinedHypothesis<F>
        where W: WeakLearner<Hypothesis = F>
    {
        let f = self.weights.iter()
            .copied()
            .zip(self.classifiers.iter().cloned())
            .filter(|(w, _)| *w != 0.0)
            .collect::<Vec<_>>();
        CombinedHypothesis::from(f)
    }
}


impl<F> Logger for AdaBoost<'_, F>
    where F: Classifier
{
    /// AdaBoost optimizes the exp loss, so that this method returns
    /// the sum of `exp( -y * f(x) )`, 
    /// where `f` is the current confidence-rated combined hypothesis.
    fn objective_value(&self) -> f64 {
        let n_sample = self.data.shape().0 as f64;

        self.target.i64()
            .expect("The target class is not a dtype i64")
            .into_iter()
            .map(|y| y.unwrap() as f64)
            .enumerate()
            .map(|(i, y)| (- y * self.prediction(self.data, i)).exp())
            .sum::<f64>()
            / n_sample
    }


    fn prediction(&self, data: &DataFrame, i: usize) -> f64 {
        self.weights.iter()
            .zip(&self.classifiers)
            .map(|(w, h)| w * h.confidence(data, i))
            .sum::<f64>()
    }


    fn logging<L>(
        &self,
        loss_function: &L,
        test_data: &DataFrame,
        test_target: &Series,
    ) -> (f64, f64, f64)
        where L: Fn(f64, f64) -> f64
    {
        let objval = self.objective_value();
        let train = self.loss(loss_function, self.data, self.target);
        let test = self.loss(loss_function, test_data, test_target);

        (objval, train, test)
    }
}

