//! This file defines `TotalBoost` based on the paper
//! "Totally Corrective Boosting Algorithms that Maximize the Margin"
//! by Warmuth et al.
//! 
use crate::{
    Sample,
    Booster,
    WeakLearner,

    State,
    Classifier,
    CombinedHypothesis,

    SoftBoost,
};

// use crate::research::Logger;


/// `TotalBoost`.
/// This algorithm is originally invented in this paper:
/// [Totally corrective boosting algorithms that maximize the margin](https://dl.acm.org/doi/10.1145/1143844.1143970)
/// by Manfred K. Warmuth, Jun Liao, and Gunnar Rätsch.
/// `TotalBoost` is a special case of [`SoftBoost`].
/// That is, 
/// `TotalBoost` restricts [`SoftBoost::nu`] as `1.0`.  
/// For this reason, `TotalBoost` is just a wrapper of [`SoftBoost`].
/// 
/// # Example
/// The following code shows a small example 
/// for running [`SoftBoost`](SoftBoost).  
/// See also:
/// - [`SoftBoost`]
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
/// [`SoftBoost`]: SoftBoost
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
/// // Get the number of training examples.
/// let n_sample = data.shape().0 as f64;
/// 
/// // Initialize `TotalBoost` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis 
/// // whose hard margin objective value is differs at most `0.01`
/// // from the optimal one, if the training examples are linearly separable.
/// let booster = TotalBoost::init(&data, &target)
///     .tolerance(0.01);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTree::init(&data, &target)
///     .max_depth(2)
///     .criterion(Criterion::Edge);
/// 
/// // Run `TotalBoost` and obtain the resulting hypothesis `f`.
/// let f: CombinedHypothesis<DTreeClassifier> = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions: Vec<i64> = f.predict_all(&data);
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
pub struct TotalBoost<'a, F> {
    softboost: SoftBoost<'a, F>,
}


impl<'a, F> TotalBoost<'a, F>
    where F: Classifier,
{
    /// initialize the `TotalBoost`.
    pub fn init(sample: &'a Sample) -> Self {
        let softboost = SoftBoost::init(sample)
            .nu(1.0);

        TotalBoost { softboost }
    }


    /// Returns a optimal value of the optimization problem LPBoost solves
    pub fn opt_val(&self) -> f64 {
        self.softboost.opt_val()
    }


    /// Set the tolerance parameter.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.softboost = self.softboost.tolerance(tol);
        self
    }
}


impl<F> Booster<F> for TotalBoost<'_, F>
    where F: Classifier + Clone,
{
    fn preprocess<W>(
        &mut self,
        weak_learner: &W,
    )
        where W: WeakLearner<Hypothesis = F>
    {
        self.softboost.preprocess(weak_learner);
    }


    fn boost<W>(
        &mut self,
        weak_learner: &W,
        iteration: usize,
    ) -> State
        where W: WeakLearner<Hypothesis = F>
    {
        self.softboost.boost(weak_learner, iteration)
    }


    fn postprocess<W>(
        &mut self,
        weak_learner: &W,
    ) -> CombinedHypothesis<F>
        where W: WeakLearner<Hypothesis = F>
    {
        self.softboost.postprocess(weak_learner)
    }
}


// impl<F> Logger for TotalBoost<'_, F>
//     where F: Classifier
// {
//     fn weights_on_hypotheses(&mut self) {
//         self.softboost.weights_on_hypotheses();
//     }
// 
//     /// `TotalBoost` optimizes the hard margin objective,
//     /// which corresponds to the soft margin objective
//     /// with capping parameter `nu = 1.0`.
//     fn objective_value(&self)
//         -> f64
//     {
//         self.softboost.objective_value()
//     }
// 
// 
//     fn prediction(&self, data: &DataFrame, i: usize) -> f64 {
//         self.softboost.prediction(data, i)
//     }
// 
// 
//     fn logging<L>(
//         &self,
//         loss_function: &L,
//         test_data: &DataFrame,
//         test_target: &Series,
//     ) -> (f64, f64, f64)
//         where L: Fn(f64, f64) -> f64
//     {
//         self.softboost.logging(loss_function, test_data, test_target)
//     }
// }
