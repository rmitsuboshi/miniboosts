use crate::Logger;
use miniboosts_core::Sample;

const DEFAULT_ROUND: usize = 100;
const DEFAULT_TIMELIMIT_MILLIS: u128 = u128::MAX;

/// `LoggerBuilder` is a struct to construct `Logger.`
/// You need to specify the followings:
///
/// - Booster (Boosting algorithm),
/// - Weak Learner,
/// - Objective function,
/// - Loss function,
/// - Training examples,
/// - Test examples,
/// - Time limit for force quit, and
/// - Round (The log text is shown for every **round** you specified).
///
/// # Example
/// ```no_run
/// use logging::{LoggerBuilder, LoggingSoftMarginObjective};
/// use miniboosts_core::{Booster, Classifier, CurrentHypothesis, Sample, WeakLearner};
///
/// fn record<B, W, H, S, O>(booster: B, learner: W, train: &Sample, test: &Sample)
///     -> std::io::Result<O>
/// where
///     B: Booster<H, Output = O> + CurrentHypothesis<Output = S>,
///     W: WeakLearner<Hypothesis = H>,
///     S: Classifier,
///     O: Classifier,
/// {
///     let loss = |sample: &Sample, model: &S| {
///         model.predict_all(sample).iter().zip(sample.target())
///             .filter(|(prediction, label)| **prediction as f64 != **label).count()
///             as f64 / sample.shape().0 as f64
///     };
///     LoggerBuilder::new()
///         .booster(booster).weak_learner(learner)
///         .train_sample(train).test_sample(test)
///         .objective_function(LoggingSoftMarginObjective::new(1.0))
///         .loss_function(loss).time_limit_as_secs(300)
///         .build().run("output.csv")
/// }
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
    /// Construct a new instance of `LoggerBuilder.`
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
    pub fn booster(mut self, booster: B) -> Self {
        self.booster = Some(booster);
        self
    }

    /// Set the weak learner.
    pub fn weak_learner(mut self, weak_learner: W) -> Self {
        self.weak_learner = Some(weak_learner);
        self
    }

    /// Set the objective function for the boosting algorithm.
    pub fn objective_function(mut self, objective_func: F) -> Self {
        self.objective_func = Some(objective_func);
        self
    }

    /// Set the loss function.
    pub fn loss_function(mut self, loss_func: G) -> Self {
        self.loss_func = Some(loss_func);
        self
    }

    /// Set the training sample.
    pub fn train_sample(mut self, train: &'a Sample) -> Self {
        self.train = Some(train);
        self
    }

    /// Set the test sample.
    pub fn test_sample(mut self, test: &'a Sample) -> Self {
        self.test = Some(test);
        self
    }

    /// Set the time limit for boosting algorithm as milliseconds.
    /// Checked between steps using cumulative boosting time; see [`Logger::run`].
    #[inline(always)]
    pub fn time_limit_as_millis(mut self, time_limit: u128) -> Self {
        self.time_limit = time_limit;
        self
    }

    /// Set the time limit for boosting algorithm as seconds.
    /// Checked between steps using cumulative boosting time; see [`Logger::run`].
    #[inline(always)]
    pub fn time_limit_as_secs(mut self, time_limit: u64) -> Self {
        self.time_limit = (time_limit as u128)
            .checked_mul(1_000_u128)
            .expect("The time limit (ms) cannot be represented as u128");
        self
    }

    /// Set the time limit for boosting algorithm as minutes.
    /// Checked between steps using cumulative boosting time; see [`Logger::run`].
    #[inline(always)]
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
    /// If you don't want to print the log,
    /// set `usize::MAX`.
    #[inline(always)]
    pub fn print_every(mut self, round: usize) -> Self {
        assert!(round > 0, "print interval must be positive");
        self.round = round;
        self
    }

    /// Build [Logger] from the given components.
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
        }
    }
}
