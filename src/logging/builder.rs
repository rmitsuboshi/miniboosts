use crate::core::Sample;
use crate::logging::Logger;

const DEFAULT_ROUND: usize = 100;
const DEFAULT_TIMELIMIT_MILLIS: u128 = u128::MAX;

/// Builder for [`Logger`].
///
/// Required components (missing ones make [`build`](Self::build) panic):
///
/// - booster (`B`), which must implement
///   [`CurrentHypothesis`](crate::CurrentHypothesis) for [`Logger::run`],
/// - weak learner (`W`),
/// - objective function (`F`, a [`LoggingObjective`](crate::LoggingObjective)),
/// - loss function (`G`, a closure `Fn(&Sample, &S) -> f64` where `S` is the
///   booster's current-hypothesis type),
/// - training sample, and
/// - test sample (used for reporting only).
///
/// Optional settings:
///
/// - time limit (default: unlimited), checked between boosting steps against
///   cumulative boosting time; it does not interrupt a running step;
/// - print interval (default: every `100` steps).
///
/// An iteration cap can be set on the built logger with
/// [`Logger::max_iterations`].
///
/// # Examples
/// A closure loss needs explicit parameter types because
/// [`loss_function`](Self::loss_function) does not bound `G`, so the
/// closure signature cannot be inferred from it. The
/// current-hypothesis type of [`AdaBoost`](crate::AdaBoost) with decision trees
/// is `WeightedMajority<DecisionTreeClassifier>`.
/// ```
/// # use std::io::BufReader;
/// # use miniboosts::{prelude::*, Sample};
/// # let csv = b"x,y,class\n\
/// #     -2.0,0.5,-1\n-1.5,-1.0,-1\n-1.0,0.8,-1\n-0.5,-1.5,-1\n\
/// #     0.5,-2.0,-1\n1.0,-1.8,-1\n-1.8,1.2,-1\n0.2,-0.9,-1\n\
/// #     2.0,-0.5,1\n1.5,1.0,1\n1.0,-0.2,1\n0.5,1.5,1\n\
/// #     -0.5,2.0,1\n-1.0,1.8,1\n1.8,0.4,1\n0.3,0.9,1\n";
/// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)?.set_target("class");
/// # let n = sample.shape().0;
/// # let ix: Vec<usize> = (0..n).filter(|i| i % 4 != 0).chain((0..n).filter(|i| i % 4 == 0)).collect();
/// # let (train, test) = sample.split(&ix, n - n / 4, n);
/// # fn error_rate<H: Classifier>(sample: &Sample, h: &H) -> f64 {
/// #     let wrong = h.predict_all(sample).iter().zip(sample.target())
/// #         .filter(|(p, y)| **p as f64 != **y).count();
/// #     wrong as f64 / sample.shape().0 as f64
/// # }
/// # let path = std::env::temp_dir()
/// #     .join(format!("miniboosts-doc-builder-type-{}.csv", std::process::id()));
/// use miniboosts::{DecisionTreeClassifier, LoggerBuilder, LoggingSoftMarginObjective, WeightedMajority};
///
/// type Ensemble = WeightedMajority<DecisionTreeClassifier>;
/// // Average exponential loss of the (normalized) ensemble.
/// let exp_loss = |sample: &Sample, f: &Ensemble| {
///     let m = sample.shape().0 as f64;
///     f.confidence_all(sample).iter().zip(sample.target())
///         .map(|(fx, y)| (-y * fx).exp()).sum::<f64>() / m
/// };
///
/// let mut logger = LoggerBuilder::new()
///     .booster(AdaBoost::init(&train).tolerance(0.1))
///     .weak_learner(DecisionTreeBuilder::new(&train).max_depth(1).build())
///     .objective_function(LoggingSoftMarginObjective::new(1.0))
///     .loss_function(exp_loss)
///     .train_sample(&train)
///     .test_sample(&test)
///     .print_every(usize::MAX)
///     .build()
///     .max_iterations(20);
/// let classifier = logger.run(&path)?;
/// let log = std::fs::read_to_string(&path)?;
/// std::fs::remove_file(&path)?;
/// // Header plus at most 20 steps.
/// assert!((2..=21).contains(&log.lines().count()));
/// assert_eq!(classifier.predict_all(&test).len(), test.shape().0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct LoggerBuilder<'a, B, W, F, G> {
    booster: Option<B>,
    weak_learner: Option<W>,
    objective_func: Option<F>,
    loss_func: Option<G>,
    train: Option<&'a Sample>,
    test: Option<&'a Sample>,
    time_limit: u128,
    round: usize,
}

impl<'a, B, W, F, G> LoggerBuilder<'a, B, W, F, G> {
    /// Creates a builder with no components, no time limit, and a print
    /// interval of `100`.
    /// # Examples
    /// All six components are required; the time limit and print interval are
    /// optional.
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{prelude::*, Sample};
    /// # let csv = b"x,y,class\n\
    /// #     -2.0,0.5,-1\n-1.5,-1.0,-1\n-1.0,0.8,-1\n-0.5,-1.5,-1\n\
    /// #     0.5,-2.0,-1\n1.0,-1.8,-1\n-1.8,1.2,-1\n0.2,-0.9,-1\n\
    /// #     2.0,-0.5,1\n1.5,1.0,1\n1.0,-0.2,1\n0.5,1.5,1\n\
    /// #     -0.5,2.0,1\n-1.0,1.8,1\n1.8,0.4,1\n0.3,0.9,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)?.set_target("class");
    /// # let n = sample.shape().0;
    /// # let ix: Vec<usize> = (0..n).filter(|i| i % 4 != 0).chain((0..n).filter(|i| i % 4 == 0)).collect();
    /// # let (train, test) = sample.split(&ix, n - n / 4, n);
    /// # fn error_rate<H: Classifier>(sample: &Sample, h: &H) -> f64 {
    /// #     let wrong = h.predict_all(sample).iter().zip(sample.target())
    /// #         .filter(|(p, y)| **p as f64 != **y).count();
    /// #     wrong as f64 / sample.shape().0 as f64
    /// # }
    /// # let path = std::env::temp_dir()
    /// #     .join(format!("miniboosts-doc-builder-new-{}.csv", std::process::id()));
    /// use miniboosts::{LoggerBuilder, LoggingSoftMarginObjective};
    ///
    /// let mut logger = LoggerBuilder::new()
    ///     .booster(LpBoost::init(&train).nu(1.0))
    ///     .weak_learner(DecisionTreeBuilder::new(&train).max_depth(1).build())
    ///     .objective_function(LoggingSoftMarginObjective::new(1.0))
    ///     .loss_function(error_rate)
    ///     .train_sample(&train)
    ///     .test_sample(&test)
    ///     .build();
    /// let classifier = logger.run(&path)?;
    /// std::fs::remove_file(&path)?;
    /// assert_eq!(classifier.predict_all(&test).len(), test.shape().0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new() -> Self {
        Self {
            booster: None,
            weak_learner: None,
            objective_func: None,
            loss_func: None,
            train: None,
            test: None,
            time_limit: DEFAULT_TIMELIMIT_MILLIS,
            round: DEFAULT_ROUND,
        }
    }

    /// Set the boosting algorithm.
    /// # Examples
    /// Any [`Booster`](crate::Booster) that also implements
    /// [`CurrentHypothesis`](crate::CurrentHypothesis), configured before it is
    /// passed in:
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{prelude::*, Sample};
    /// # let csv = b"x,y,class\n\
    /// #     -2.0,0.5,-1\n-1.5,-1.0,-1\n-1.0,0.8,-1\n-0.5,-1.5,-1\n\
    /// #     0.5,-2.0,-1\n1.0,-1.8,-1\n-1.8,1.2,-1\n0.2,-0.9,-1\n\
    /// #     2.0,-0.5,1\n1.5,1.0,1\n1.0,-0.2,1\n0.5,1.5,1\n\
    /// #     -0.5,2.0,1\n-1.0,1.8,1\n1.8,0.4,1\n0.3,0.9,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)?.set_target("class");
    /// # let n = sample.shape().0;
    /// # let ix: Vec<usize> = (0..n).filter(|i| i % 4 != 0).chain((0..n).filter(|i| i % 4 == 0)).collect();
    /// # let (train, test) = sample.split(&ix, n - n / 4, n);
    /// # fn error_rate<H: Classifier>(sample: &Sample, h: &H) -> f64 {
    /// #     let wrong = h.predict_all(sample).iter().zip(sample.target())
    /// #         .filter(|(p, y)| **p as f64 != **y).count();
    /// #     wrong as f64 / sample.shape().0 as f64
    /// # }
    /// # let path = std::env::temp_dir()
    /// #     .join(format!("miniboosts-doc-builder-booster-{}.csv", std::process::id()));
    /// use miniboosts::{LoggerBuilder, LoggingSoftMarginObjective};
    ///
    /// let booster = MlpBoost::init(&train).nu(2.0).tolerance(0.05);
    /// let mut logger = LoggerBuilder::new()
    ///     .booster(booster)
    /// #   .weak_learner(DecisionTreeBuilder::new(&train).max_depth(1).build())
    /// #   .objective_function(LoggingSoftMarginObjective::new(2.0))
    /// #   .loss_function(error_rate)
    /// #   .train_sample(&train)
    /// #   .test_sample(&test)
    /// #   .print_every(usize::MAX)
    ///     // ... remaining components ...
    ///     .build()
    ///     .max_iterations(10);
    /// logger.run(&path)?;
    /// let log = std::fs::read_to_string(&path)?;
    /// std::fs::remove_file(&path)?;
    /// assert!((2..=11).contains(&log.lines().count()));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn booster(mut self, booster: B) -> Self {
        self.booster = Some(booster);
        self
    }

    /// Set the weak learner.
    /// # Examples
    /// Build the weak learner on the training sample:
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{prelude::*, Sample};
    /// # let csv = b"x,y,class\n\
    /// #     -2.0,0.5,-1\n-1.5,-1.0,-1\n-1.0,0.8,-1\n-0.5,-1.5,-1\n\
    /// #     0.5,-2.0,-1\n1.0,-1.8,-1\n-1.8,1.2,-1\n0.2,-0.9,-1\n\
    /// #     2.0,-0.5,1\n1.5,1.0,1\n1.0,-0.2,1\n0.5,1.5,1\n\
    /// #     -0.5,2.0,1\n-1.0,1.8,1\n1.8,0.4,1\n0.3,0.9,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)?.set_target("class");
    /// # let n = sample.shape().0;
    /// # let ix: Vec<usize> = (0..n).filter(|i| i % 4 != 0).chain((0..n).filter(|i| i % 4 == 0)).collect();
    /// # let (train, test) = sample.split(&ix, n - n / 4, n);
    /// # fn error_rate<H: Classifier>(sample: &Sample, h: &H) -> f64 {
    /// #     let wrong = h.predict_all(sample).iter().zip(sample.target())
    /// #         .filter(|(p, y)| **p as f64 != **y).count();
    /// #     wrong as f64 / sample.shape().0 as f64
    /// # }
    /// # let path = std::env::temp_dir()
    /// #     .join(format!("miniboosts-doc-builder-weak-{}.csv", std::process::id()));
    /// use miniboosts::{LoggerBuilder, LoggingSoftMarginObjective};
    ///
    /// let tree = DecisionTreeBuilder::new(&train)
    ///     .max_depth(2)
    ///     .split_by(SplitBy::Entropy)
    ///     .build();
    /// let mut logger = LoggerBuilder::new()
    ///     .weak_learner(tree)
    /// #   .booster(LpBoost::init(&train).nu(1.0))
    /// #   .objective_function(LoggingSoftMarginObjective::new(1.0))
    /// #   .loss_function(error_rate)
    /// #   .train_sample(&train)
    /// #   .test_sample(&test)
    /// #   .print_every(usize::MAX)
    ///     // ... remaining components ...
    ///     .build();
    /// logger.run(&path)?;
    /// std::fs::remove_file(&path)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn weak_learner(mut self, weak_learner: W) -> Self {
        self.weak_learner = Some(weak_learner);
        self
    }

    /// Set the objective recorded in the `ObjectiveValue` column. It is
    /// evaluated on the training sample after every step.
    /// # Examples
    /// Use the same capping parameter `ν` as the booster so that the
    /// `ObjectiveValue` column is the objective the booster optimizes:
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{prelude::*, Sample};
    /// # let csv = b"x,y,class\n\
    /// #     -2.0,0.5,-1\n-1.5,-1.0,-1\n-1.0,0.8,-1\n-0.5,-1.5,-1\n\
    /// #     0.5,-2.0,-1\n1.0,-1.8,-1\n-1.8,1.2,-1\n0.2,-0.9,-1\n\
    /// #     2.0,-0.5,1\n1.5,1.0,1\n1.0,-0.2,1\n0.5,1.5,1\n\
    /// #     -0.5,2.0,1\n-1.0,1.8,1\n1.8,0.4,1\n0.3,0.9,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)?.set_target("class");
    /// # let n = sample.shape().0;
    /// # let ix: Vec<usize> = (0..n).filter(|i| i % 4 != 0).chain((0..n).filter(|i| i % 4 == 0)).collect();
    /// # let (train, test) = sample.split(&ix, n - n / 4, n);
    /// # fn error_rate<H: Classifier>(sample: &Sample, h: &H) -> f64 {
    /// #     let wrong = h.predict_all(sample).iter().zip(sample.target())
    /// #         .filter(|(p, y)| **p as f64 != **y).count();
    /// #     wrong as f64 / sample.shape().0 as f64
    /// # }
    /// # let path = std::env::temp_dir()
    /// #     .join(format!("miniboosts-doc-builder-objective-{}.csv", std::process::id()));
    /// use miniboosts::{LoggerBuilder, LoggingSoftMarginObjective};
    ///
    /// let nu = 2.0;
    /// let mut logger = LoggerBuilder::new()
    ///     .booster(LpBoost::init(&train).nu(nu))
    ///     .objective_function(LoggingSoftMarginObjective::new(nu))
    /// #   .weak_learner(DecisionTreeBuilder::new(&train).max_depth(1).build())
    /// #   .loss_function(error_rate)
    /// #   .train_sample(&train)
    /// #   .test_sample(&test)
    /// #   .print_every(usize::MAX)
    ///     // ... remaining components ...
    ///     .build();
    /// logger.run(&path)?;
    /// let log = std::fs::read_to_string(&path)?;
    /// std::fs::remove_file(&path)?;
    /// // The soft margin of an ensemble with confidences in [-1, 1] is in [-1, 1].
    /// for line in log.lines().skip(1) {
    ///     let objective: f64 = line.split(',').next().unwrap().parse()?;
    ///     assert!((-1.0..=1.0).contains(&objective));
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn objective_function(mut self, objective_func: F) -> Self {
        self.objective_func = Some(objective_func);
        self
    }

    /// Set the loss recorded in the `TrainLoss` and `TestLoss` columns.
    /// # Examples
    /// A generic function such as `error_rate` below works for any classifier;
    /// a closure needs explicit parameter types (see [`LoggerBuilder`]).
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{prelude::*, Sample};
    /// # let csv = b"x,y,class\n\
    /// #     -2.0,0.5,-1\n-1.5,-1.0,-1\n-1.0,0.8,-1\n-0.5,-1.5,-1\n\
    /// #     0.5,-2.0,-1\n1.0,-1.8,-1\n-1.8,1.2,-1\n0.2,-0.9,-1\n\
    /// #     2.0,-0.5,1\n1.5,1.0,1\n1.0,-0.2,1\n0.5,1.5,1\n\
    /// #     -0.5,2.0,1\n-1.0,1.8,1\n1.8,0.4,1\n0.3,0.9,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)?.set_target("class");
    /// # let n = sample.shape().0;
    /// # let ix: Vec<usize> = (0..n).filter(|i| i % 4 != 0).chain((0..n).filter(|i| i % 4 == 0)).collect();
    /// # let (train, test) = sample.split(&ix, n - n / 4, n);
    /// # fn error_rate<H: Classifier>(sample: &Sample, h: &H) -> f64 {
    /// #     let wrong = h.predict_all(sample).iter().zip(sample.target())
    /// #         .filter(|(p, y)| **p as f64 != **y).count();
    /// #     wrong as f64 / sample.shape().0 as f64
    /// # }
    /// # let path = std::env::temp_dir()
    /// #     .join(format!("miniboosts-doc-builder-loss-{}.csv", std::process::id()));
    /// use miniboosts::{LoggerBuilder, LoggingSoftMarginObjective};
    ///
    /// /// Fraction of misclassified examples.
    /// fn zero_one_loss<H: Classifier>(sample: &Sample, f: &H) -> f64 {
    ///     let predictions = f.predict_all(sample);
    ///     let wrong = predictions.iter().zip(sample.target())
    ///         .filter(|(p, y)| **p as f64 != **y)
    ///         .count();
    ///     wrong as f64 / sample.shape().0 as f64
    /// }
    ///
    /// let mut logger = LoggerBuilder::new()
    ///     .loss_function(zero_one_loss)
    /// #   .booster(LpBoost::init(&train).nu(1.0))
    /// #   .weak_learner(DecisionTreeBuilder::new(&train).max_depth(1).build())
    /// #   .objective_function(LoggingSoftMarginObjective::new(1.0))
    /// #   .train_sample(&train)
    /// #   .test_sample(&test)
    /// #   .print_every(usize::MAX)
    ///     // ... remaining components ...
    ///     .build();
    /// logger.run(&path)?;
    /// let log = std::fs::read_to_string(&path)?;
    /// std::fs::remove_file(&path)?;
    /// // TrainLoss and TestLoss are error rates.
    /// for line in log.lines().skip(1) {
    ///     let row: Vec<f64> = line.split(',').map(|v| v.parse().unwrap()).collect();
    ///     assert!((0.0..=1.0).contains(&row[1]) && (0.0..=1.0).contains(&row[2]));
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn loss_function(mut self, loss_func: G) -> Self {
        self.loss_func = Some(loss_func);
        self
    }

    /// Set the training sample.
    /// # Examples
    /// Pass the same training sample that the booster and the weak learner
    /// were built on; the logger evaluates the objective and `TrainLoss` on it.
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{prelude::*, Sample};
    /// # let csv = b"x,y,class\n\
    /// #     -2.0,0.5,-1\n-1.5,-1.0,-1\n-1.0,0.8,-1\n-0.5,-1.5,-1\n\
    /// #     0.5,-2.0,-1\n1.0,-1.8,-1\n-1.8,1.2,-1\n0.2,-0.9,-1\n\
    /// #     2.0,-0.5,1\n1.5,1.0,1\n1.0,-0.2,1\n0.5,1.5,1\n\
    /// #     -0.5,2.0,1\n-1.0,1.8,1\n1.8,0.4,1\n0.3,0.9,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)?.set_target("class");
    /// # let n = sample.shape().0;
    /// # let ix: Vec<usize> = (0..n).filter(|i| i % 4 != 0).chain((0..n).filter(|i| i % 4 == 0)).collect();
    /// # let (train, test) = sample.split(&ix, n - n / 4, n);
    /// # fn error_rate<H: Classifier>(sample: &Sample, h: &H) -> f64 {
    /// #     let wrong = h.predict_all(sample).iter().zip(sample.target())
    /// #         .filter(|(p, y)| **p as f64 != **y).count();
    /// #     wrong as f64 / sample.shape().0 as f64
    /// # }
    /// # let path = std::env::temp_dir()
    /// #     .join(format!("miniboosts-doc-builder-train-{}.csv", std::process::id()));
    /// use miniboosts::{LoggerBuilder, LoggingSoftMarginObjective};
    ///
    /// let mut logger = LoggerBuilder::new()
    ///     .booster(LpBoost::init(&train).nu(1.0))
    ///     .weak_learner(DecisionTreeBuilder::new(&train).max_depth(1).build())
    ///     .train_sample(&train)
    /// #   .objective_function(LoggingSoftMarginObjective::new(1.0))
    /// #   .loss_function(error_rate)
    /// #   .test_sample(&test)
    /// #   .print_every(usize::MAX)
    ///     // ... remaining components ...
    ///     .build();
    /// logger.run(&path)?;
    /// std::fs::remove_file(&path)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn train_sample(mut self, train: &'a Sample) -> Self {
        self.train = Some(train);
        self
    }

    /// Set the test sample. It is used only to compute `TestLoss`.
    /// # Examples
    /// Hold out the test rows with [`Sample::split`]; they never reach the
    /// booster or the weak learner.
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{prelude::*, Sample};
    /// # let csv = b"x,y,class\n\
    /// #     -2.0,0.5,-1\n-1.5,-1.0,-1\n-1.0,0.8,-1\n-0.5,-1.5,-1\n\
    /// #     0.5,-2.0,-1\n1.0,-1.8,-1\n-1.8,1.2,-1\n0.2,-0.9,-1\n\
    /// #     2.0,-0.5,1\n1.5,1.0,1\n1.0,-0.2,1\n0.5,1.5,1\n\
    /// #     -0.5,2.0,1\n-1.0,1.8,1\n1.8,0.4,1\n0.3,0.9,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)?.set_target("class");
    /// # let n = sample.shape().0;
    /// # let ix: Vec<usize> = (0..n).filter(|i| i % 4 != 0).chain((0..n).filter(|i| i % 4 == 0)).collect();
    /// # let (train, test) = sample.split(&ix, n - n / 4, n);
    /// # fn error_rate<H: Classifier>(sample: &Sample, h: &H) -> f64 {
    /// #     let wrong = h.predict_all(sample).iter().zip(sample.target())
    /// #         .filter(|(p, y)| **p as f64 != **y).count();
    /// #     wrong as f64 / sample.shape().0 as f64
    /// # }
    /// # let path = std::env::temp_dir()
    /// #     .join(format!("miniboosts-doc-builder-test-{}.csv", std::process::id()));
    /// use miniboosts::{LoggerBuilder, LoggingSoftMarginObjective};
    ///
    /// // A fixed permutation of the 16 rows; ix[12..16] become the test sample.
    /// let ix: Vec<usize> = (0..16).map(|i| (i * 5) % 16).collect();
    /// let (train, test) = sample.split(&ix, 12, 16);
    /// assert_eq!((train.shape().0, test.shape().0), (12, 4));
    /// let mut logger = LoggerBuilder::new()
    ///     .train_sample(&train)
    ///     .test_sample(&test)
    /// #   .booster(LpBoost::init(&train).nu(1.0))
    /// #   .weak_learner(DecisionTreeBuilder::new(&train).max_depth(1).build())
    /// #   .objective_function(LoggingSoftMarginObjective::new(1.0))
    /// #   .loss_function(error_rate)
    /// #   .print_every(usize::MAX)
    ///     // ... remaining components ...
    ///     .build();
    /// logger.run(&path)?;
    /// std::fs::remove_file(&path)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn test_sample(mut self, test: &'a Sample) -> Self {
        self.test = Some(test);
        self
    }

    /// Set the time limit for boosting algorithm as milliseconds.
    /// Checked between steps using cumulative boosting time; see [`Logger::run`].
    #[inline(always)]
    /// # Examples
    /// A limit of `0` ms is reached after the first step, which is still
    /// logged:
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{prelude::*, Sample};
    /// # let csv = b"x,y,class\n\
    /// #     -2.0,0.5,-1\n-1.5,-1.0,-1\n-1.0,0.8,-1\n-0.5,-1.5,-1\n\
    /// #     0.5,-2.0,-1\n1.0,-1.8,-1\n-1.8,1.2,-1\n0.2,-0.9,-1\n\
    /// #     2.0,-0.5,1\n1.5,1.0,1\n1.0,-0.2,1\n0.5,1.5,1\n\
    /// #     -0.5,2.0,1\n-1.0,1.8,1\n1.8,0.4,1\n0.3,0.9,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)?.set_target("class");
    /// # let n = sample.shape().0;
    /// # let ix: Vec<usize> = (0..n).filter(|i| i % 4 != 0).chain((0..n).filter(|i| i % 4 == 0)).collect();
    /// # let (train, test) = sample.split(&ix, n - n / 4, n);
    /// # fn error_rate<H: Classifier>(sample: &Sample, h: &H) -> f64 {
    /// #     let wrong = h.predict_all(sample).iter().zip(sample.target())
    /// #         .filter(|(p, y)| **p as f64 != **y).count();
    /// #     wrong as f64 / sample.shape().0 as f64
    /// # }
    /// # let path = std::env::temp_dir()
    /// #     .join(format!("miniboosts-doc-builder-millis-{}.csv", std::process::id()));
    /// use miniboosts::{LoggerBuilder, LoggingSoftMarginObjective};
    ///
    /// let mut logger = LoggerBuilder::new()
    ///     .time_limit_as_millis(0)
    /// #   .booster(LpBoost::init(&train).nu(1.0))
    /// #   .weak_learner(DecisionTreeBuilder::new(&train).max_depth(1).build())
    /// #   .objective_function(LoggingSoftMarginObjective::new(1.0))
    /// #   .loss_function(error_rate)
    /// #   .train_sample(&train)
    /// #   .test_sample(&test)
    /// #   .print_every(usize::MAX)
    ///     // ... remaining components ...
    ///     .build();
    /// logger.run(&path)?;
    /// let log = std::fs::read_to_string(&path)?;
    /// std::fs::remove_file(&path)?;
    /// assert_eq!(log.lines().count(), 2); // header + one step
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn time_limit_as_millis(mut self, time_limit: u128) -> Self {
        self.time_limit = time_limit;
        self
    }

    /// Set the time limit for boosting algorithm as seconds.
    /// Checked between steps using cumulative boosting time; see [`Logger::run`].
    #[inline(always)]
    /// # Examples
    /// Stop once the cumulative boosting time reaches 30 seconds (this small
    /// example finishes long before that):
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{prelude::*, Sample};
    /// # let csv = b"x,y,class\n\
    /// #     -2.0,0.5,-1\n-1.5,-1.0,-1\n-1.0,0.8,-1\n-0.5,-1.5,-1\n\
    /// #     0.5,-2.0,-1\n1.0,-1.8,-1\n-1.8,1.2,-1\n0.2,-0.9,-1\n\
    /// #     2.0,-0.5,1\n1.5,1.0,1\n1.0,-0.2,1\n0.5,1.5,1\n\
    /// #     -0.5,2.0,1\n-1.0,1.8,1\n1.8,0.4,1\n0.3,0.9,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)?.set_target("class");
    /// # let n = sample.shape().0;
    /// # let ix: Vec<usize> = (0..n).filter(|i| i % 4 != 0).chain((0..n).filter(|i| i % 4 == 0)).collect();
    /// # let (train, test) = sample.split(&ix, n - n / 4, n);
    /// # fn error_rate<H: Classifier>(sample: &Sample, h: &H) -> f64 {
    /// #     let wrong = h.predict_all(sample).iter().zip(sample.target())
    /// #         .filter(|(p, y)| **p as f64 != **y).count();
    /// #     wrong as f64 / sample.shape().0 as f64
    /// # }
    /// # let path = std::env::temp_dir()
    /// #     .join(format!("miniboosts-doc-builder-secs-{}.csv", std::process::id()));
    /// use miniboosts::{LoggerBuilder, LoggingSoftMarginObjective};
    ///
    /// let mut logger = LoggerBuilder::new()
    ///     .time_limit_as_secs(30)
    /// #   .booster(LpBoost::init(&train).nu(1.0))
    /// #   .weak_learner(DecisionTreeBuilder::new(&train).max_depth(1).build())
    /// #   .objective_function(LoggingSoftMarginObjective::new(1.0))
    /// #   .loss_function(error_rate)
    /// #   .train_sample(&train)
    /// #   .test_sample(&test)
    /// #   .print_every(usize::MAX)
    ///     // ... remaining components ...
    ///     .build();
    /// logger.run(&path)?;
    /// std::fs::remove_file(&path)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn time_limit_as_secs(mut self, time_limit: u64) -> Self {
        self.time_limit = (time_limit as u128)
            .checked_mul(1_000_u128)
            .expect("The time limit (ms) cannot be represented as u128");
        self
    }

    /// Set the time limit for boosting algorithm as minutes.
    /// Checked between steps using cumulative boosting time; see [`Logger::run`].
    #[inline(always)]
    /// # Examples
    /// Long experiments are usually budgeted in minutes:
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{prelude::*, Sample};
    /// # let csv = b"x,y,class\n\
    /// #     -2.0,0.5,-1\n-1.5,-1.0,-1\n-1.0,0.8,-1\n-0.5,-1.5,-1\n\
    /// #     0.5,-2.0,-1\n1.0,-1.8,-1\n-1.8,1.2,-1\n0.2,-0.9,-1\n\
    /// #     2.0,-0.5,1\n1.5,1.0,1\n1.0,-0.2,1\n0.5,1.5,1\n\
    /// #     -0.5,2.0,1\n-1.0,1.8,1\n1.8,0.4,1\n0.3,0.9,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)?.set_target("class");
    /// # let n = sample.shape().0;
    /// # let ix: Vec<usize> = (0..n).filter(|i| i % 4 != 0).chain((0..n).filter(|i| i % 4 == 0)).collect();
    /// # let (train, test) = sample.split(&ix, n - n / 4, n);
    /// # fn error_rate<H: Classifier>(sample: &Sample, h: &H) -> f64 {
    /// #     let wrong = h.predict_all(sample).iter().zip(sample.target())
    /// #         .filter(|(p, y)| **p as f64 != **y).count();
    /// #     wrong as f64 / sample.shape().0 as f64
    /// # }
    /// # let path = std::env::temp_dir()
    /// #     .join(format!("miniboosts-doc-builder-mins-{}.csv", std::process::id()));
    /// use miniboosts::{LoggerBuilder, LoggingSoftMarginObjective};
    ///
    /// let mut logger = LoggerBuilder::new()
    ///     .time_limit_as_mins(5)
    /// #   .booster(LpBoost::init(&train).nu(1.0))
    /// #   .weak_learner(DecisionTreeBuilder::new(&train).max_depth(1).build())
    /// #   .objective_function(LoggingSoftMarginObjective::new(1.0))
    /// #   .loss_function(error_rate)
    /// #   .train_sample(&train)
    /// #   .test_sample(&test)
    /// #   .print_every(usize::MAX)
    ///     // ... remaining components ...
    ///     .build();
    /// logger.run(&path)?;
    /// std::fs::remove_file(&path)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn time_limit_as_mins(mut self, time_limit: u64) -> Self {
        self.time_limit = (time_limit as u128)
            .checked_mul(60_u128)
            .expect("The time limit (s) cannot be represented as u128")
            .checked_mul(1_000u128)
            .expect("The time limit (ms) cannot be represented as u128");
        self
    }

    /// Set the interval to print the current status.
    /// By default, the method `run` prints its status every `100` rounds.
    /// `usize::MAX` suppresses the per-step table and the final `[FIN]` or
    /// `[LIMIT]` line; the settings panel and a `[TLE]` line are still printed.
    ///
    /// # Panics
    /// Panics if `round == 0`.
    #[inline(always)]
    /// # Examples
    /// Print a progress line every 10 steps, or pass `usize::MAX` to keep only
    /// the settings panel. The CSV file always gets every step.
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{prelude::*, Sample};
    /// # let csv = b"x,y,class\n\
    /// #     -2.0,0.5,-1\n-1.5,-1.0,-1\n-1.0,0.8,-1\n-0.5,-1.5,-1\n\
    /// #     0.5,-2.0,-1\n1.0,-1.8,-1\n-1.8,1.2,-1\n0.2,-0.9,-1\n\
    /// #     2.0,-0.5,1\n1.5,1.0,1\n1.0,-0.2,1\n0.5,1.5,1\n\
    /// #     -0.5,2.0,1\n-1.0,1.8,1\n1.8,0.4,1\n0.3,0.9,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)?.set_target("class");
    /// # let n = sample.shape().0;
    /// # let ix: Vec<usize> = (0..n).filter(|i| i % 4 != 0).chain((0..n).filter(|i| i % 4 == 0)).collect();
    /// # let (train, test) = sample.split(&ix, n - n / 4, n);
    /// # fn error_rate<H: Classifier>(sample: &Sample, h: &H) -> f64 {
    /// #     let wrong = h.predict_all(sample).iter().zip(sample.target())
    /// #         .filter(|(p, y)| **p as f64 != **y).count();
    /// #     wrong as f64 / sample.shape().0 as f64
    /// # }
    /// # let path = std::env::temp_dir()
    /// #     .join(format!("miniboosts-doc-builder-print-{}.csv", std::process::id()));
    /// use miniboosts::{LoggerBuilder, LoggingSoftMarginObjective};
    ///
    /// let mut logger = LoggerBuilder::new()
    ///     .print_every(10)
    /// #   .booster(LpBoost::init(&train).nu(1.0))
    /// #   .weak_learner(DecisionTreeBuilder::new(&train).max_depth(1).build())
    /// #   .objective_function(LoggingSoftMarginObjective::new(1.0))
    /// #   .loss_function(error_rate)
    /// #   .train_sample(&train)
    /// #   .test_sample(&test)
    ///     // ... remaining components ...
    ///     .build();
    /// logger.run(&path)?;
    /// std::fs::remove_file(&path)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn print_every(mut self, round: usize) -> Self {
        assert!(round > 0, "print interval must be positive");
        self.round = round;
        self
    }

    /// Build [`Logger`] from the given components.
    ///
    /// # Panics
    /// Panics if the booster, weak learner, objective function, loss
    /// function, training sample, or test sample has not been set.
    /// # Examples
    /// `build` returns a [`Logger`]; settings that need the booster's types,
    /// such as [`Logger::max_iterations`], are chained after it.
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{prelude::*, Sample};
    /// # let csv = b"x,y,class\n\
    /// #     -2.0,0.5,-1\n-1.5,-1.0,-1\n-1.0,0.8,-1\n-0.5,-1.5,-1\n\
    /// #     0.5,-2.0,-1\n1.0,-1.8,-1\n-1.8,1.2,-1\n0.2,-0.9,-1\n\
    /// #     2.0,-0.5,1\n1.5,1.0,1\n1.0,-0.2,1\n0.5,1.5,1\n\
    /// #     -0.5,2.0,1\n-1.0,1.8,1\n1.8,0.4,1\n0.3,0.9,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)?.set_target("class");
    /// # let n = sample.shape().0;
    /// # let ix: Vec<usize> = (0..n).filter(|i| i % 4 != 0).chain((0..n).filter(|i| i % 4 == 0)).collect();
    /// # let (train, test) = sample.split(&ix, n - n / 4, n);
    /// # fn error_rate<H: Classifier>(sample: &Sample, h: &H) -> f64 {
    /// #     let wrong = h.predict_all(sample).iter().zip(sample.target())
    /// #         .filter(|(p, y)| **p as f64 != **y).count();
    /// #     wrong as f64 / sample.shape().0 as f64
    /// # }
    /// # let path = std::env::temp_dir()
    /// #     .join(format!("miniboosts-doc-builder-build-{}.csv", std::process::id()));
    /// use miniboosts::{LoggerBuilder, LoggingSoftMarginObjective};
    ///
    /// let logger = LoggerBuilder::new()
    ///     .booster(LpBoost::init(&train).nu(1.0))
    ///     .weak_learner(DecisionTreeBuilder::new(&train).max_depth(1).build())
    ///     .objective_function(LoggingSoftMarginObjective::new(1.0))
    ///     .loss_function(error_rate)
    ///     .train_sample(&train)
    ///     .test_sample(&test)
    ///     .build();
    /// let mut logger = logger.max_iterations(1).print_every(usize::MAX);
    /// logger.run(&path)?;
    /// let log = std::fs::read_to_string(&path)?;
    /// std::fs::remove_file(&path)?;
    /// assert_eq!(log.lines().count(), 2); // header + the single allowed step
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn build(self) -> Logger<'a, B, W, F, G> {
        let booster = self.booster.expect("Boosting algorithm is not specified");
        let weak_learner = self.weak_learner.expect("Weak learner is not specified");
        let objective_func = self
            .objective_func
            .expect("Objective function is not specified");
        let loss_func = self.loss_func.expect("Loss function is not specified");
        let train = self.train.expect("Training sample is not specified");
        let test = self.test.expect("Test sample is not specified");
        let time_limit = self.time_limit;
        let round = self.round;

        Logger {
            booster,
            weak_learner,
            objective_func,
            loss_func,
            train,
            test,
            time_limit,
            round,
            max_iterations: usize::MAX,
        }
    }
}
