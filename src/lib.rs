#![warn(missing_docs)]

pub mod prelude;

pub use miniboosts_core::{
    Sample,
    SampleReader,
    Booster,
    WeakLearner,
    Classifier,
    Regressor,
};

pub use optimization::*;
pub use logging::*;

pub use hypotheses::{
    WeightedMajority,
    NaiveAggregation,
};

/// The AdaBoost algorithm 
/// proposed by Robert E. Schapire and Yoav Freund.
/// 
/// This struct is based on the book: 
///
/// [
/// Boosting: Foundations and Algorithms
/// ](https://direct.mit.edu/books/oa-monograph/5342/BoostingFoundations-and-Algorithms)  
/// by Robert E. Schapire and Yoav Freund.
/// 
/// AdaBoost is a boosting algorithm for binary classification 
/// that minimizes exponential loss over a set of training examples.
///
/// # Convergence rate
/// Given a set of training examples of size `m > 0`
/// and an accuracy parameter `ε > 0`,
/// `AdaBoost` finds an convex combination of hypotheses
/// whose empirical loss is less than `ε`
/// in `O( ln(m) / ε² )` iterations.
/// 
/// # Related information
/// - As some papers proved, 
///   `AdaBoost` **approximately maximizes the hard margin.**
/// 
/// - [`AdaBoostV`], a successor of AdaBoost, maximizes the hard margin.
/// 
/// # Example
/// ```no_run
/// use miniboosts::prelude::*;
/// 
/// let file = "/path/to/training/dataset.csv";
/// // Read the training sample from the CSV file.
/// // We use the column named `class` as the label.
/// let sample = SampleReader::default()
///     .file(file)
///     .has_header(true)
///     .target_feature("class")
///     .read()
///     .unwrap();
/// 
/// // Initialize `AdaBoost` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis whose training error is
/// // less than `0.01` if the traing examples are linearly separable.
/// let mut booster = AdaBoost::init(&sample)
///     .tolerance(0.01);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .split_by(SplitBy::Entropy)
///     .build();
/// 
/// // Run `AdaBoost` and obtain the resulting hypothesis `f`.
/// let f = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions = f.predict_all(&sample);
/// 
/// // Get the number of training examples.
/// let n_sample = sample.shape().0 as f64;
/// 
/// // Calculate the training loss.
/// let target = sample.target();
/// let training_loss = target.into_iter()
///     .zip(predictions)
///     .map(|(&y, fx)| if y as i64 == fx { 0.0 } else { 1.0 })
///     .sum::<f64>()
///     / n_sample;
/// 
///
/// println!("Training Loss is: {training_loss}");
/// ```
pub use adaboost::AdaBoost;

/// The `AdaBoostV` algorithm, proposed by Rätsch and Warmuth.  
/// `AdaBoostV`, also known as `AdaBoost_{ν}^{★}`, 
/// is a boosting algorithm proposed in the following paper:
/// 
/// [Gunnar Rätsch and Manfred K. Warmuth - Efficient Margin Maximizing with Boosting](https://www.jmlr.org/papers/v6/ratsch05a.html)
/// 
/// Given a set `{(x_{1}, y_{1}), (x_{2}, y_{2}), ..., (x_{m}, y_{m})}`
/// of training examples,
/// [`AdaBoostV`] aims to find an optimal solution of
/// the hard-margin optimization problem:
///
/// ```txt
/// max ρ
/// ρ,w
/// s.t. y_{i} Σ_{h ∈ Δ_{H}} w_{h} h(x_{i}) ≥ ρ, for all i ∈ [m],
///      w ∈ Δ_{H}
/// ```
///
/// # Convergence rate
/// Assume that there exists a convex combination of hypotheses
/// that perfectly classifies the training examples:
///
/// ```txt
/// ∃ w ∈ Δ_{H},
/// ∀ (x, y) in training examples,
/// y Σ_{h ∈ H} w_{h} h( x ) > 0.
/// ```
///
/// Given a set of training examples of size `m > 0`
/// and an accuracy parameter `ε > 0`,
/// `AdaBoostV` finds an `ε`-approximate solution of
/// the hard-margin optimization problem
/// in `O( ln(m) / ε² )` iterations.
/// 
/// # Related information
/// 
/// - `AdaBoostV` does not use the weak learnability parameter.
/// 
/// # Example
/// The following code shows a small example 
/// for running [`AdaBoostV`].  
/// 
/// 
/// ```no_run
/// use miniboosts::prelude::*;
/// 
/// let file = "path/to/dataset.csv";
/// // Read the training sample from the CSV file.
/// // We use the column named `class` as the label.
/// let sample = SampleReader::default()
///     .file(file)
///     .has_header(true)
///     .target_feature("class")
///     .read()
///     .unwrap();
/// 
/// // Initialize `AdaBoostV` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis whose training error is
/// // less than `0.01` if the traing examples are linearly separable.
/// let mut booster = AdaBoostV::init(&sample)
///     .tolerance(0.01);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .split_by(SplitBy::Entropy)
///     .build();
/// 
/// // Run `AdaBoostV` and obtain the resulting hypothesis `f`.
/// let f = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions = f.predict_all(&sample);
/// 
/// // Get the number of training examples.
/// let n_examples = sample.shape().0 as f64;
/// 
/// // Calculate the training loss.
/// let target = sample.target();
/// let training_loss = target.into_iter()
///     .zip(predictions)
///     .map(|(&y, fx)| if y as i64 == fx { 0.0 } else { 1.0 })
///     .sum::<f64>()
///     / n_examples;
/// 
///
/// println!("Training Loss is: {training_loss}");
/// ```
pub use adaboostv::AdaBoostV;

/// The `LpBoost` algorithm 
/// proposed by Demiriz, Bennett, and Shawe-Taylor.  
/// `LpBoost` is originally proposed in the following paper:  
/// 
/// [Ayhan Demiriz, Kristin P. Bennett, and John Shawe-Taylor - Linear Programming Boosting via Column Generation](https://www.researchgate.net/publication/220343627_Linear_Programming_Boosting_via_Column_Generation)
/// 
/// My implementation of `LpBoost` is based on the following paper:
/// 
/// [Manfred K. Warmuth, Karen Glocer, and Gunnar Rätsch - Boosting algorithms for Maximizing the Soft Margin](https://proceedings.neurips.cc/paper/2007/file/cfbce4c1d7c425baf21d6b6f2babe6be-Paper.pdf)
/// 
/// Given a set `{(x_{1}, y_{1}), (x_{2}, y_{2}), ..., (x_{m}, y_{m})}`
/// of training examples,
/// a capping parameters `ν ∈ [1, m]`, and
/// an accuracy parameter `ε > 0`,
/// `LpBoost` aims to find an `ε`-approximate solution of
/// the soft-margin optimization problem:
/// ```txt
///  max  ρ - (1/ν) Σ_{i=1}^{m} ξ_{i}
/// ρ,w,ξ
/// s.t. y_{i} Σ_{h ∈ Δ_{H}} w_{h} h(x_{i}) ≥ ρ - ξ_{i},
///                                         for all i ∈ [m],
///      w ∈ Δ_{H},
///      ξ ≥ 0.
/// ```
/// 
/// # Convergence rate
/// There exists a training set of size `m > 0` such that
/// `LpBoost` takes `Ω( m )` iterations for the worst case.
///
///
/// # Related information
/// - Currently (2023), `LpBoost` has no convergence guarantee.
/// - [`ErlpBoost`], 
///   A stabilized version of [`LpBoost`] is 
///   proposed by Warmuth et al. (2008).
/// 
/// # Example
/// The following code shows a small example for running [`LpBoost`].  
/// 
/// 
/// ```no_run
/// use miniboosts::prelude::*;
/// 
/// let file = "path/to/dataset.csv";
/// // Read the training sample from the CSV file.
/// // We use the column named `class` as the label.
/// let sample = SampleReader::default()
///     .file(file)
///     .has_header(true)
///     .target_feature("class")
///     .read()
///     .unwrap();
/// 
/// 
/// // Get the number of training examples.
/// let n_sample = sample.shape().0 as f64;
/// 
/// // Set the upper-bound parameter of outliers in `sample`.
/// // Here we assume that the outliers are at most 10% of `sample`.
/// let nu = 0.1 * n_sample;
/// 
/// // Initialize `LpBoost` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis whose training error is
/// // less than `0.01` if the traing examples are linearly separable.
/// let mut booster = LpBoost::init(&sample)
///     .tolerance(0.01)
///     .nu(0.1 * n_sample);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .split_by(SplitBy::Entropy)
///     .build();
/// 
/// // Run `LpBoost` and obtain the resulting hypothesis `f`.
/// let f = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions = f.predict_all(&sample);
/// 
/// // Calculate the training loss.
/// let target = sample.target();
/// let training_loss = target.into_iter()
///     .zip(predictions)
///     .map(|(&y, fx)| if y as i64 == fx { 0.0 } else { 1.0 })
///     .sum::<f64>()
///     / n_sample;
/// 
///
/// println!("Training Loss is: {training_loss}");
/// ```
pub use lpboost::LpBoost;

/// The `ErlpBoost` algorithm proposed in the following paper: 
/// 
/// [Manfred K. Warmuth, Karen A. Glocer, and S. V. N. Vishwanathan - Entropy Regularized LPBoost](https://link.springer.com/chapter/10.1007/978-3-540-87987-9_23)
/// 
/// Given a set `{(x_{1}, y_{1}), (x_{2}, y_{2}), ..., (x_{m}, y_{m})}`
/// of training examples,
/// a capping parameters `ν ∈ [1, m]`, and
/// an accuracy parameter `ε > 0`,
/// `ErlpBoost` aims to find an `ε`-approximate solution of
/// the soft-margin optimization problem:
/// ```txt
///  max  ρ - (1/ν) Σ_{i=1}^{m} ξ_{i}
/// ρ,w,ξ
/// s.t. y_{i} Σ_{h ∈ Δ_{H}} w_{h} h(x_{i}) ≥ ρ - ξ_{i},
///                                         for all i ∈ [m],
///      w ∈ Δ_{H},
///      ξ ≥ 0.
/// ```
/// 
/// # Convergence rate
/// - `ErlpBoost` terminates in `O( ln(m/ν) / ε² )` iterations.
///
/// # Related information
/// - Every round, `ErlpBoost` solves a convex program
///   by the sequential quadratic minimization technique.
///   So, running time per round is slow 
///   compared to [`LpBoost`].
/// 
/// # Example
/// The following code shows a small example 
/// for running [`ErlpBoost`].  
/// 
/// 
/// ```no_run
/// use miniboosts::prelude::*;
///
/// let file = "path/to/dataset.csv";
/// // Read the training sample from the CSV file.
/// // We use the column named `class` as the label.
/// let sample = SampleReader::default()
///     .file(file)
///     .has_header(true)
///     .target_feature("class")
///     .read()
///     .unwrap();
/// 
/// // Get the number of training examples.
/// let n_sample = sample.shape().0 as f64;
/// 
/// // Set the upper-bound parameter of outliers in `sample`.
/// // Here we assume that the outliers are at most 10% of `sample`.
/// let nu = 0.1 * n_sample;
/// 
/// // Initialize `ErlpBoost` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis whose training error is
/// // less than `0.01` if the traing examples are linearly separable.
/// let mut booster = ErlpBoost::init(&sample)
///     .tolerance(0.01)
///     .nu(0.1 * n_sample);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .split_by(SplitBy::Entropy)
///     .build();
/// 
/// // Run `ErlpBoost` and obtain the resulting hypothesis `f`.
/// let f = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions = f.predict_all(&sample);
/// 
/// // Calculate the training loss.
/// let target = sample.target();
/// let training_loss = target.into_iter()
///     .zip(predictions)
///     .map(|(&y, fx)| if y as i64 == fx { 0.0 } else { 1.0 })
///     .sum::<f64>()
///     / n_sample;
/// 
///
/// println!("Training Loss is: {training_loss}");
/// ```
pub use erlpboost::ErlpBoost;

/// The Graph Separation Boosting algorithm proposed by Robert E. Schapire and Yoav Freund.
/// 
/// The algorithm is comes from the following paper: 
/// [Boosting Simple Learners](https://theoretics.episciences.org/10757/pdf)
/// by Noga Alon, Alon Gonen, Elad Hazan, and Shay Moran.
/// 
/// Given a `γ`-weak learner and a set `S` of training examples of size `m`,
/// `GraphSeparationBoosting` terminates in `O( ln(m) / γ)` rounds.
///
/// To guarantee the generalization ability,
/// one needs to use a **simple** weak-learner.
/// 
/// # Example
/// The following code shows a small example 
/// for running [`Graph Separation Boosting`](Graph Separation Boosting).  
/// See also:
/// - [`DecisionTree`]
/// - [`DecisionTreeClassifier`]
/// - [`NaiveAggregation<F>`]
/// - [`Sample`]
/// 
/// [`DecisionTree`]: crate::weak_learner::DecisionTree
/// [`DecisionTreeClassifier`]: crate::weak_learner::DecisionTreeClassifier
/// [`NaiveAggregation<F>`]: crate::hypothesis::NaiveAggregation
/// 
/// 
/// ```no_run
/// use miniboosts::prelude::*;
/// 
/// let file = "path/to/dataset.csv";
/// // Read the training sample from the CSV file.
/// // We use the column named `class` as the label.
/// let sample = SampleReader::default()
///     .file(file)
///     .has_header(true)
///     .target_feature("class")
///     .read()
///     .unwrap();
/// 
/// let mut booster = GraphSeparationBoosting::init(&sample);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTreeBuilder::new(&sample)
///     .max_depth(1)
///     .split_by(SplitBy::Entropy)
///     .build();
/// 
/// // Run `GraphSeparationBoosting` and obtain the resulting hypothesis `f`.
/// let f = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions = f.predict_all(&sample);
/// 
/// // Get the number of training examples.
/// let n_examples = sample.shape().0 as f64;
/// 
/// // Calculate the training loss.
/// let target = sample.target();
/// let training_loss = target.into_iter()
///     .zip(predictions)
///     .map(|(&y, fx)| if y as i64 == fx { 0.0 } else { 1.0 })
///     .sum::<f64>()
///     / n_examples;
/// 
///
/// println!("Training Loss is: {training_loss}");
/// ```
pub use graph_separation_boosting::GraphSeparationBoosting;

/// The Corrective ERLPBoost algorithm, 
/// proposed in the following paper:
/// 
/// [Shai Shalev-Shwartz and Yoram Singer - On the equivalence of weak learnability and linear separability: new relaxations and efficient boosting algorithms](https://link.springer.com/article/10.1007/s10994-010-5173-z)
/// 
///
/// Given a set `{(x_{1}, y_{1}), (x_{2}, y_{2}), ..., (x_{m}, y_{m})}`
/// of training examples,
/// a capping parameters `ν ∈ [1, m]`, and
/// an accuracy parameter `ε > 0`,
/// Corrective ERLPBoost aims to find an `ε`-approximate solution of
/// the soft margin optimization problem
/// ```txt
///  max  ρ - (1/ν) Σ_{i=1}^{m} ξ_{i}
/// ρ,w,ξ
/// s.t. y_{i} Σ_{h ∈ Δ_{H}} w_{h} h(x_{i}) ≥ ρ - ξ_{i},
///                                         for all i ∈ [m],
///      w ∈ Δ_{H},
///      ξ ≥ 0.
/// ```
/// without using LP/QP solver.
///
/// # Convergence rate
/// - `CorrectiveErlpBoost` terminates in `O( ln(m/ν) / ε² )` iterations.
///
/// # Related information
/// - Running time per round is 
///   the fastest among soft-margin boosting algorithms.
/// - The iteration bound is the same as the one to [`ERLPBoost`][erlpboost].
/// - Empirically, the number of rounds tend to huge compared to
///   totally corrective algorithms 
///   such as [`ERLPBoost`][erlpboost] and [`LPBoost`][lpboost].
/// - By default, CorrectiveErlpBoost uses short-step strategy,
///   which converges very slow.
/// - One can specify other step size strategies.
///   See [`CorrectiveErlpBoost::update_rule`].
///
/// [erlpboost]: crate::booster::ERLPBoost
/// [lpboost]: crate::booster::LPBoost
/// 
/// # Example
/// The following code shows a small example 
/// for running [`CorrectiveErlpBoost`].  
/// 
/// ```no_run
/// use miniboosts::prelude::*;
/// 
/// let file = "path/to/dataset.csv";
/// // Read the training sample from the CSV file.
/// // We use the column named `class` as the label.
/// let sample = SampleReader::default()
///     .file(file)
///     .has_header(true)
///     .target_feature("class")
///     .read()
///     .unwrap();
/// 
/// 
/// // Get the number of training examples.
/// let n_examples = sample.shape().0 as f64;
/// 
/// // Set the upper-bound parameter of outliers in `sample`.
/// // Here we assume that the outliers are at most 10% of `sample`.
/// let nu = 0.1 * n_examples;
/// 
/// // Initialize `CorrectiveErlpBoost` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis whose training error is
/// // less than `0.01` if the traing examples are linearly separable.
/// let mut booster = CorrectiveErlpBoost::init(&sample)
///     .tolerance(0.01)
///     .nu(0.1 * n_examples);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .split_by(SplitBy::Entropy)
///     .build();
/// 
/// // Run `CorrectiveErlpBoost` and obtain the resulting hypothesis `f`.
/// let f = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions = f.predict_all(&sample);
/// 
/// // Calculate the training loss.
/// let target = sample.target();
/// let training_loss = target.into_iter()
///     .zip(predictions)
///     .map(|(&y, fx)| if y as i64 == fx { 0.0 } else { 1.0 })
///     .sum::<f64>()
///     / n_examples;
/// 
///
/// println!("Training Loss is: {training_loss}");
/// ```
pub use corrective_erlpboost::CorrectiveErlpBoost;

/// The MadaBoost algorithm 
/// proposed by Carlos Domingo and Osamu Watanabe, 2000.
/// 
/// This algorithm is based on the paper: 
///
/// [
/// MadaBoost: A Modification of AdaBoost
/// ](https://www.learningtheory.org/colt2000/papers/DomingoWatanabe.pdf)
/// by Carlos Domingo and Osamu Watanabe.
/// 
/// MadaBoost is a boosting algorithm for binary classification 
/// that minimizes exponential loss over a set of training examples.
///
/// This struct provides the `MB:1/2` algorithm.
///
/// # Convergence rate
/// Given a set of training examples of size `m > 0`
/// and an accuracy parameter `ε > 0`,
/// `MadaBoost` finds an convex combination of hypotheses
/// whose empirical loss is less than `ε`
/// in `O( m / ε² )` iterations.
/// 
/// 
/// ```no_run
/// use miniboosts::prelude::*;
/// 
/// let file = "/path/to/training/dataset.csv";
/// // Read the training sample from the CSV file.
/// // We use the column named `class` as the label.
/// let sample = SampleReader::default()
///     .file(file)
///     .has_header(true)
///     .target_feature("class")
///     .read()
///     .unwrap();
/// 
/// // Initialize `MadaBoost` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis whose training error is
/// // less than `0.01` if the traing examples are linearly separable.
/// let mut booster = MadaBoost::init(&sample)
///     .tolerance(0.01);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .split_by(SplitBy::Entropy)
///     .build();
/// 
/// // Run `MadaBoost` and obtain the resulting hypothesis `f`.
/// let f = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions = f.predict_all(&sample);
/// 
/// // Get the number of training examples.
/// let n_sample = sample.shape().0 as f64;
/// 
/// // Calculate the training loss.
/// let target = sample.target();
/// let training_loss = target.into_iter()
///     .zip(predictions)
///     .map(|(&y, fx)| if y as i64 == fx { 0.0 } else { 1.0 })
///     .sum::<f64>()
///     / n_sample;
/// 
///
/// println!("Training Loss is: {training_loss}");
/// ```
pub use madaboost::MadaBoost;

/// The MlpBoost algorithm, shorthand of Modified LpBoost algorithm,
/// proposed in the following paper:
/// 
/// [Ryotaro Mitsuboshi, Kohei Hatano, and Eiji Takimoto - Boosting as Frank-Wolfe](https://arxiv.org/abs/2209.10831)
/// 
/// MlpBoost is an abstraction of 
/// some soft-margin boosting algorithms.
///
/// 
/// Given a set `{(x_{1}, y_{1}), (x_{2}, y_{2}), ..., (x_{m}, y_{m})}`
/// of training examples,
/// a capping parameters `ν ∈ [1, m]`, and
/// an accuracy parameter `ε > 0`,
/// `MlpBoost` aims to find an `ε`-approximate solution of
/// the soft-margin optimization problem:
/// ```txt
///  max  ρ - (1/ν) Σ_{i=1}^{m} ξ_{i}
/// ρ,w,ξ
/// s.t. y_{i} Σ_{h ∈ Δ_{H}} w_{h} h(x_{i}) ≥ ρ - ξ_{i},
///                                         for all i ∈ [m],
///      w ∈ Δ_{H},
///      ξ ≥ 0.
/// ```
/// 
/// `MlpBoost` can be seen as a mixture of 
/// [`LpBoost`] and [`CorrectiveErlpBoost`].
/// `MlpBoost` attains the fast convergence property 
/// of [`LpBoost`] while keeping the convergence guarantee 
/// for [`CorrectiveErlpBoost`].
/// 
/// # Convergence rate
/// - [`MlpBoost`] terminates in `O( ln(m/ν) / ε² )` iterations.
///
/// # Related information
/// - Every round, `MlpBoost` solves a linear program.
/// - By default, `MlpBoost` sets 
///   [`FwUpdateRule::ShortStep`] as the primary strategy.
///   This choice triggers 
///   [`LpBoost`] update more frequently.
///   Therefore, one can expect fast convergent for real-world dataset.
///   (For many real datasets, [`LpBoost`] is the fastest.
/// - One can choose different Frank-Wolfe algorithm.
///
/// # Example
/// The following code shows 
/// a small example for running [`MlpBoost`].  
/// 
/// 
/// ```no_run
/// use miniboosts::prelude::*;
/// 
/// let file = "path/to/dataset.csv";
/// // Read the training sample from the CSV file.
/// // We use the column named `class` as the label.
/// let sample = SampleReader::default()
///     .file(file)
///     .has_header(true)
///     .target_feature("class")
///     .read()
///     .unwrap();
/// 
/// 
/// // Get the number of training examples.
/// let n_examples = sample.shape().0 as f64;
/// 
/// // Set the upper-bound parameter of outliers in `sample`.
/// // Here we assume that the outliers are at most 10% of `sample`.
/// let nu = 0.1 * n_examples;
/// 
/// // Initialize `MlpBoost` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis whose training error is
/// // less than `0.01` if the traing examples are linearly separable.
/// // Note that the default tolerance parameter is set as `1 / n_examples`,
/// // where `n_examples = sample.shape().0` is 
/// // the number of training examples in `sample`.
/// let mut booster = MlpBoost::init(&sample)
///     .tolerance(0.01)
///     .nu(0.1 * n_examples);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .split_by(SplitBy::Entropy)
///     .build();
/// 
/// // Run `MlpBoost` and obtain the resulting hypothesis `f`.
/// let f = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions = f.predict_all(&sample);
/// 
/// // Calculate the training loss.
/// let target = sample.target();
/// let training_loss = target.into_iter()
///     .zip(predictions)
///     .map(|(&y, fx)| if y as i64 == fx { 0.0 } else { 1.0 })
///     .sum::<f64>()
///     / n_examples;
/// 
///
/// println!("Training Loss is: {training_loss}");
/// ```
pub use mlpboost::MlpBoost;

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
/// for running [`SmoothBoost`].  
/// See also:
/// - [`SmoothBoost::kappa`]
/// - [`SmoothBoost::gamma`]
/// - [`DecisionTree`]
/// - [`DecisionTreeClassifier`]
/// - [`WeightedMajority<F>`]
/// 
/// [`SmoothBoost::kappa`]: SmoothBoost::kappa
/// [`SmoothBoost::gamma`]: SmoothBoost::gamma
/// [`DecisionTree`]: crate::weak_learner::DecisionTree
/// [`DecisionTreeClassifier`]: crate::weak_learner::DecisionTreeClassifier
/// [`WeightedMajority<F>`]: crate::hypothesis::WeightedMajority
/// 
/// 
/// ```no_run
/// use miniboosts::prelude::*;
/// 
/// let file = "/path/to/training/dataset.csv";
/// // Read the training sample from the CSV file.
/// // We use the column named `class` as the label.
/// let sample = SampleReader::default()
///     .file(file)
///     .has_header(true)
///     .target_feature("class")
///     .read()
///     .unwrap();
/// 
/// // Get the number of training examples.
/// let n_examples = sample.shape().0 as f64;
/// 
/// // Initialize `SmoothBoost` and 
/// // set the weak learner guarantee `gamma` as `0.05`.
/// // For this case, weak learner returns a hypothesis
/// // that returns a hypothesis with weighted loss 
/// // at most `0.45 = 0.5 - 0.05`.
/// let mut booster = SmoothBoost::init(&sample)
///     .kappa(0.01)
///     .gamma(0.05);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .split_by(SplitBy::Entropy)
///     .build();
/// 
/// // Run `SmoothBoost` and obtain the resulting hypothesis `f`.
/// let f = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions: Vec<i64> = f.predict_all(&sample);
/// 
/// // Calculate the training loss.
/// let target = sample.target();
/// let training_loss = target.into_iter()
///     .zip(predictions)
///     .map(|(&y, fx)| if y as i64 == fx { 0.0 } else { 1.0 })
///     .sum::<f64>()
///     / n_examples;
///
///
/// println!("Training Loss is: {training_loss}");
/// ```
pub use smoothboost::SmoothBoost;

/// The SoftBoost algorithm proposed in the following paper:  
///
/// [Gunnar Rätsch, Manfred K. Warmuth, and Laren A. Glocer - Boosting Algorithms for Maximizing the Soft Margin](https://papers.nips.cc/paper/2007/hash/cfbce4c1d7c425baf21d6b6f2babe6be-Abstract.html) 
///
/// Given a set `{(x_{1}, y_{1}), (x_{2}, y_{2}), ..., (x_{m}, y_{m})}`
/// of training examples,
/// a capping parameters `ν ∈ [1, m]`, and
/// an accuracy parameter `ε > 0`,
/// `SoftBoost` aims to find an `ε`-approximate solution of
/// the soft-margin optimization problem:
/// ```txt
///  max  ρ - (1/ν) Σ_{i=1}^{m} ξ_{i}
/// ρ,w,ξ
/// s.t. y_{i} Σ_{h ∈ Δ_{H}} w_{h} h(x_{i}) ≥ ρ - ξ_{i},
///                                         for all i ∈ [m],
///      w ∈ Δ_{H},
///      ξ ≥ 0.
/// ```
/// 
/// 
/// # Convergence rate
/// - `SoftBoost` terminates in `O( ln(m/ν) / ε² )` iterations.
/// 
/// # Related information
/// - Every round, `ERLPBoost` solves a convex program
///   by the sequential quadratic minimization technique.
///   So, running time per round is slow 
///   compared to [`LpBoost`].
/// - [`SoftBoost`] is the extension 
///   of [`TotalBoost`].
/// 
/// # Example
/// The following code shows 
/// a small example for running [`SoftBoost`].  
/// 
/// 
/// ```no_run
/// use miniboosts::prelude::*;
/// 
/// let file = "path/to/dataset.csv";
/// // Read the training sample from the CSV file.
/// // We use the column named `class` as the label.
/// let sample = SampleReader::default()
///     .file(file)
///     .has_header(true)
///     .target_feature("class")
///     .read()
///     .unwrap();
/// 
/// // Get the number of training examples.
/// let n_examples = sample.shape().0 as f64;
/// 
/// // Set the upper-bound parameter of outliers in `sample`.
/// // Here we assume that the outliers are at most 10% of `sample`.
/// let nu = 0.1 * n_examples;
/// 
/// // Initialize `SoftBoost` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis 
/// // whose soft margin objective value is differs at most `0.01`
/// // from the optimal one.
/// // Further, at the end of this chain,
/// // SoftBoost calls `SoftBoost::nu` to set the capping parameter 
/// // as `0.1 * n_examples`, which means that, 
/// // at most, `0.1 * n_examples` examples are regarded as outliers.
/// let mut booster = SoftBoost::init(&sample)
///     .tolerance(0.01)
///     .nu(nu);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .split_by(SplitBy::Entropy)
///     .build();
/// 
/// // Run `SoftBoost` and obtain the resulting hypothesis `f`.
/// let f = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions = f.predict_all(&sample);
/// 
/// // Calculate the training loss.
/// let target = sample.target();
/// let training_loss = target.into_iter()
///     .zip(predictions)
///     .map(|(&y, fx)| if y as i64 == fx { 0.0 } else { 1.0 })
///     .sum::<f64>()
///     / n_examples;
/// 
/// 
/// println!("Training Loss is: {training_loss}");
/// ```
pub use softboost::SoftBoost;

/// The TotalBoost algorithm proposed in the following paper:
/// [Manfred K. Warmuth, Jun Liao, and Gunnar Rätsch - Totally corrective boosting algorithms that maximize the margin](https://dl.acm.org/doi/10.1145/1143844.1143970)
///
/// Given a set `{(x_{1}, y_{1}), (x_{2}, y_{2}), ..., (x_{m}, y_{m})}`
/// of training examples,
/// [`TotalBoost`] aims to find an optimal solution of
/// the hard-margin optimization problem:
///
/// ```txt
/// max ρ
/// ρ,w
/// s.t. y_{i} Σ_{h ∈ Δ_{H}} w_{h} h(x_{i}) ≥ ρ, for all i ∈ [m],
///      w ∈ Δ_{H}
/// ```
/// 
/// # Convergence rate
/// Assume that there exists a convex combination of hypotheses
/// that perfectly classifies the training examples:
///
/// ```txt
/// ∃ w ∈ Δ_{h},
/// ∀ (x, y) in training examples,
/// y Σ_{h ∈ H} w_{h} h( x ) > 0.
/// ```
///
/// Given a set of training examples of size `m > 0`
/// and an accuracy parameter `ε > 0`,
/// `TotalBoost` finds an `ε`-approximate solution of
/// the hard-margin optimization problem
/// in `o( ln(m) / ε² )` iterations.
/// 
/// # Related information
/// - [`TotalBoost`] is a special case of [`SoftBoost`].
///   That is, `TotalBoost` restricts [`SoftBoost::nu`] as `1.0`.  
///   For this reason, [`TotalBoost`] is 
///   just a wrapper of [`SoftBoost`].
///
/// 
/// # Example
/// The following code shows 
/// a small example for running [`TotalBoost`].  
/// 
/// 
/// ```no_run
/// use miniboosts::prelude::*;
/// 
/// let file = "path/to/dataset.csv";
/// // Read the training sample from the CSV file.
/// // We use the column named `class` as the label.
/// let sample = SampleReader::default()
///     .file(file)
///     .has_header(true)
///     .target_feature("class")
///     .read()
///     .unwrap();
/// 
/// 
/// // Get the number of training examples.
/// let n_examples = sample.shape().0 as f64;
/// 
/// // Initialize `TotalBoost` and set the tolerance parameter as `0.01`.
/// // This means `booster` returns a hypothesis whose training error is
/// // less than `0.01` if the traing examples are linearly separable.
/// let mut booster = TotalBoost::init(&sample)
///     .tolerance(0.01);
/// 
/// // Set the weak learner with setting parameters.
/// let weak_learner = DecisionTreeBuilder::new(&sample)
///     .max_depth(2)
///     .split_by(SplitBy::Entropy)
///     .build();
/// 
/// // Run `TotalBoost` and obtain the resulting hypothesis `f`.
/// let f = booster.run(&weak_learner);
/// 
/// // Get the predictions on the training set.
/// let predictions = f.predict_all(&sample);
/// 
/// // Calculate the training loss.
/// let target = sample.target();
/// let training_loss = target.into_iter()
///     .zip(predictions)
///     .map(|(&y, fx)| if y as i64 == fx { 0.0 } else { 1.0 })
///     .sum::<f64>()
///     / n_examples;
/// 
///
/// println!("Training Loss is: {training_loss}");
/// ```
pub use totalboost::TotalBoost;

pub use decision_tree::DecisionTreeBuilder;
pub use decision_tree::*;

