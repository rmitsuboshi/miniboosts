//! This file defines `SmoothBoost` based on the paper
//! ``Smooth Boosting and Learning with Malicious Noise''
//! by Rocco A. Servedio.


use polars::prelude::*;
use rayon::prelude::*;

use crate::{
    Booster,
    WeakLearner,

    State,
    Classifier,
    CombinedHypothesis,
};


use crate::research::{
    Logger,
    soft_margin_objective,
};


/// `SmoothBoost`.
/// Variable names, such as `kappa`, `gamma`, and `theta`, 
/// come from the original paper.  
/// **Note that** `SmoothBoost` needs to know 
/// the weak learner guarantee `gamma`.  
/// See Figure 1 in this paper: 
/// [Smooth Boosting and Learning with Malicious Noise](https://link.springer.com/chapter/10.1007/3-540-44581-1_31) by Rocco A. Servedio.
/// 
/// # Example
/// The following code shows a small example 
/// for running [`SmoothBoost`](SmoothBoost).  
/// See also:
/// - [`SmoothBoost::tolerance`]
/// - [`SmoothBoost::gamma`]
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
/// [`SmoothBoost::tolerance`]: SmoothBoost::tolerance
/// [`SmoothBoost::gamma`]: SmoothBoost::gamma
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
/// // Initialize `SmoothBoost` and 
/// // set the weak learner guarantee `gamma` as `0.05`.
/// // For this case, weak learner returns a hypothesis
/// // that returns a hypothesis with weighted loss 
/// // at most `0.45 = 0.5 - 0.05`.
/// let booster = SmoothBoost::init(&data, &target)
///     .tolerance(0.01)
///     .gamma(0.05);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTree::init(&data, &target)
///     .max_depth(2)
///     .criterion(Criterion::Edge);
/// 
/// // Run `SmoothBoost` and obtain the resulting hypothesis `f`.
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
pub struct SmoothBoost<'a, F> {
    data: &'a DataFrame,
    target: &'a Series,

    /// Desired accuracy
    kappa: f64,

    /// Desired margin for the final hypothesis.
    /// To guarantee the convergence rate, `theta` should be
    /// `gamma / (2.0 + gamma)`.
    theta: f64,

    /// Weak-learner guarantee;
    /// for any distribution over the training examples,
    /// the weak-learner returns a hypothesis
    /// with edge at least `gamma`.
    gamma: f64,

    /// The number of training examples.
    n_sample: usize,

    current: usize,

    /// Terminated iteration.
    terminated: usize,

    max_iter: usize,

    classifiers: Vec<F>,


    m: Vec<f64>,
    n: Vec<f64>,
}


impl<'a, F> SmoothBoost<'a, F> {
    /// Initialize `SmoothBoost`.
    pub fn init(data: &'a DataFrame, target: &'a Series) -> Self {
        let n_sample = data.shape().0;

        let gamma = 0.5;


        Self {
            data,
            target,

            kappa: 0.5,
            theta: gamma / (2.0 + gamma), // gamma / (2.0 + gamma)
            gamma,

            n_sample,

            current: 0_usize,

            terminated: usize::MAX,
            max_iter: usize::MAX,

            classifiers: Vec::new(),

            m: Vec::new(),
            n: Vec::new(),
        }
    }


    /// Set the tolerance parameter `kappa`.
    #[inline(always)]
    pub fn tolerance(mut self, kappa: f64) -> Self {
        self.kappa = kappa;

        self
    }


    /// Set the parameter `gamma`.
    /// `gamma` is the weak learner guarantee;  
    /// `SmoothBoost` assumes the weak learner to returns a hypothesis `h`
    /// such that
    /// `0.5 * sum_i D[i] |h(x[i]) - y[i]| <= 0.5 - gamma`
    /// for the given distribution.  
    /// **Note that** this is an extremely assumption.
    #[inline(always)]
    pub fn gamma(mut self, gamma: f64) -> Self {
        // `gamma` must be in [0.0, 0.5)
        assert!((0.0..0.5).contains(&gamma));
        self.gamma = gamma;

        self
    }


    /// Set the parameter `theta`.
    fn theta(&mut self) {
        self.theta = self.gamma / (2.0 + self.gamma);
    }


    /// Returns the maximum iteration
    /// of SmoothBoost to satisfy the stopping criterion.
    fn max_loop(&self) -> usize {
        let denom = self.kappa
            * self.gamma.powi(2)
            * (1.0 - self.gamma).sqrt();


        (2.0 / denom).ceil() as usize
    }


    fn check_preconditions(&self) {
        // Check `kappa`.
        if !(0.0..1.0).contains(&self.kappa) || self.kappa <= 0.0 {
            panic!(
                "Invalid kappa. \
                 The parameter `kappa` must be in (0.0, 1.0)"
            );
        }

        // Check `gamma`.
        if !(self.theta..0.5).contains(&self.gamma) {
            panic!(
                "Invalid gamma. \
                 The parameter `gamma` must be in [self.theta, 0.5)"
            );
        }
    }
}



impl<F> Booster<F> for SmoothBoost<'_, F>
    where F: Classifier + Clone,
{
    fn preprocess<W>(
        &mut self,
        _weak_learner: &W,
    )
        where W: WeakLearner<Hypothesis = F>
    {
        self.n_sample = self.data.shape().0;
        // Set the paremeter `theta`.
        self.theta();

        // Check whether the parameter satisfies the pre-conditions.
        self.check_preconditions();


        self.current = 0_usize;
        self.max_iter = self.max_loop();
        self.terminated = self.max_iter;

        self.classifiers = Vec::new();


        self.m = vec![1.0; self.n_sample];
        self.n = vec![1.0; self.n_sample];
    }


    fn boost<W>(
        &mut self,
        weak_learner: &W,
        iteration: usize,
    ) -> State
        where W: WeakLearner<Hypothesis = F>
    {

        if self.max_iter < iteration {
            return State::Terminate;
        }

        self.current = iteration;


        let sum = self.m.iter().sum::<f64>();
        // Check the stopping criterion.
        if sum < self.n_sample as f64 * self.kappa {
            self.terminated = iteration - 1;
            return State::Terminate;
        }


        // Compute the distribution.
        let dist = self.m.iter()
            .map(|mj| *mj / sum)
            .collect::<Vec<_>>();


        // Call weak learner to obtain a hypothesis.
        self.classifiers.push(
            weak_learner.produce(self.data, self.target, &dist[..])
        );
        let h: &F = self.classifiers.last().unwrap();


        let margins = self.target.i64()
            .expect("The target is not a dtype i64")
            .into_iter()
            .enumerate()
            .map(|(i, y)| y.unwrap() as f64 * h.confidence(self.data, i));


        // Update `n`
        self.n.iter_mut()
            .zip(margins)
            .for_each(|(nj, yh)| {
                *nj = *nj + yh - self.theta;
            });


        // Update `m`
        self.m.par_iter_mut()
            .zip(&self.n[..])
            .for_each(|(mj, nj)| {
                if *nj <= 0.0 {
                    *mj = 1.0;
                } else {
                    *mj = (1.0 - self.gamma).powf(*nj * 0.5);
                }
            });

        State::Continue
    }


    fn postprocess<W>(
        &mut self,
        _weak_learner: &W,
    ) -> CombinedHypothesis<F>
        where W: WeakLearner<Hypothesis = F>
    {
        let weight = 1.0 / self.terminated as f64;
        let clfs = self.classifiers.clone()
            .into_iter()
            .map(|h| (weight, h))
            .collect::<Vec<(f64, F)>>();

        CombinedHypothesis::from(clfs)
    }
}


impl<F> Logger for SmoothBoost<'_, F>
    where F: Classifier
{
    fn objective_value(&self)
        -> f64
    {
        let unit = if self.current > 0 {
            1.0 / self.current as f64
        } else {
            0.0
        };
        let weights = vec![unit; self.current];


        let n_sample = self.data.shape().0 as f64;
        let nu = self.kappa * n_sample;

        soft_margin_objective(
            self.data, self.target, &weights[..], &self.classifiers[..], nu
        )
    }


    fn prediction(&self, data: &DataFrame, i: usize) -> f64 {
        let unit = if self.current > 0 {
            1.0 / self.current as f64
        } else {
            0.0
        };
        let weights = vec![unit; self.current];

        weights.iter()
            .zip(&self.classifiers[..])
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
