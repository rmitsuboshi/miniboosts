//! Provides `SparsiBoost` by Grønlund et al.
use polars::prelude::*;
// use rayon::prelude::*;


use crate::{
    Booster,
    WeakLearner,

    State,
    Classifier,
    CombinedHypothesis,
};


// use crate::research::Logger;
use crate::AdaBoostV;

/// Defines `SparsiBoost`.
/// This struct is based on the paper: 
/// [Optimal Minimal Margin Maximization with Boosting](https://proceedings.mlr.press/v97/mathiasen19a.html)
/// by Allan Grønlund, Kasper Green Larsen, and Alexander Mathiasen.
/// 
/// # Example
/// The following code shows a small example 
/// for running [`SparsiBoost`](SparsiBoost).  
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
/// // Initialize `SparsiBoost` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis 
/// // whose hard margin objective value is differs at most `0.01`
/// // from the optimal one, if the training examples are linearly separable.
/// let booster = SparsiBoost::init(&data, &target)
///     .tolerance(0.01);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTree::init(&data, &target)
///     .max_depth(2)
///     .criterion(Criterion::Edge);
/// 
/// // Run `SparsiBoost` and obtain the resulting hypothesis `f`.
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
pub struct SparsiBoost<'a, F> {
    // Training data
    data: &'a DataFrame,
    // Corresponding label
    target: &'a Series,

    // `AdaBoostV` algorithm.
    // I think this part can be replaced by any boosting algorithm
    // that maximizes the hard-margin.
    adaboostv: AdaBoostV<'a, F>,


    n_hypotheses: usize,
}


impl<'a, F> SparsiBoost<'a, F> {
    /// Initialize the `SparsiBoost<'a, F>`.
    pub fn init(data: &'a DataFrame, target: &'a Series) -> Self {
        let n_sample = data.shape().0;
        assert!(n_sample != 0);

        let adaboostv = AdaBoostV::init(data, target);

        let n_hypotheses = usize::MAX;

        Self { data, target, adaboostv, n_hypotheses }
    }


    /// Set the tolerance parameter.
    #[inline]
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.adaboostv = self.adaboostv.tolerance(tolerance);

        self
    }


    /// Set the desired number of hypotheses.
    #[inline]
    pub fn n_hypotheses(mut self, n_hypotheses: usize) -> Self {
        self.n_hypotheses = n_hypotheses;
        self
    }
}

impl<'a, F> SparsiBoost<'a, F>
    where F: Classifier
{
    /// Sparsify the given combined hypothesis
    /// with approximately keeping the margin.
    fn sparsify(&self, f: CombinedHypothesis<F>) -> CombinedHypothesis<F>
    {
        let (weights, hypotheses) = f.decompose();
        let n_hypotheses = hypotheses.len();

        // DEBUG
        assert_eq!(weights.len(), n_hypotheses);

        let mut indices = (0..n_hypotheses).collect::<Vec<usize>>();
        // Sort the indices in the descending order of `|w|`
        indices.sort_by(|&i, &j|
            weights[j].abs().partial_cmp(&weights[i].abs()).unwrap()
        );

        let one_third = n_hypotheses / 3;
        let smaller = indices.drain(one_third..).collect::<Vec<_>>();

        // The vector `larger` corresponds to the variable `R` in their paper.
        let larger = indices;
        let omega = weights[smaller[0]];

        todo!()
    }
}


impl<F> Booster<F> for SparsiBoost<'_, F>
    where F: Classifier + Clone,
{
    fn preprocess<W>(
        &mut self,
        weak_learner: &W,
    )
        where W: WeakLearner<Hypothesis = F>
    {
        self.adaboostv.preprocess(weak_learner)
    }


    fn boost<W>(
        &mut self,
        weak_learner: &W,
        iteration: usize,
    ) -> State
        where W: WeakLearner<Hypothesis = F>,
    {
        self.adaboostv.boost(weak_learner, iteration)
    }


    fn postprocess<W>(
        &mut self,
        weak_learner: &W,
    ) -> CombinedHypothesis<F>
        where W: WeakLearner<Hypothesis = F>
    {
        self.adaboostv.postprocess(weak_learner)
    }


    fn run<W>(
        &mut self,
        weak_learner: &W,
    ) -> CombinedHypothesis<F>
        where W: WeakLearner<Hypothesis = F>
    {
        let mut f = self.adaboostv.run(weak_learner);
        f.normalize();


        self.sparsify(f)
    }
}
