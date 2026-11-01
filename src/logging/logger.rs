use colored::Colorize;

use crate::core::{Booster, Classifier, Sample, WeakLearner};
use crate::logging::objective::LoggingObjective;

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::time::{Duration, Instant};

const DEFAULT_ROUND: usize = 100;
const DEFAULT_TIMELIMIT_MILLIS: u128 = u128::MAX;
const WIDTH: usize = 8;
const PREC_WIDTH: usize = 5;
const FULL_WIDTH: usize = 60;
const HEADER: &str = "ObjectiveValue,TrainLoss,TestLoss,Time\n";

/// Runs a booster step by step and logs the objective value, the
/// train/test loss, and the cumulative running time after each step.
///
/// Construct it with [`LoggerBuilder`](crate::LoggerBuilder) or
/// [`Logger::new`], then call [`run`](Self::run). Type parameters: `B`
/// booster, `W` weak learner, `F` [`LoggingObjective`], and `G` loss function
/// `Fn(&Sample, &S) -> f64`, where `S` is the booster's
/// [`CurrentHypothesis::Output`].
///
/// See the crate-level documentation ("Logging learning curves") for the CSV
/// format and timing semantics.
pub struct Logger<'a, B, W, F, G> {
    pub(super) booster: B,
    pub(super) weak_learner: W,
    pub(super) objective_func: F,
    pub(super) loss_func: G,
    pub(super) train: &'a Sample,
    pub(super) test: &'a Sample,
    pub(super) time_limit: u128,
    pub(super) round: usize,
    pub(super) max_iterations: usize,
}

impl<'a, B, W, F, G> Logger<'a, B, W, F, G> {
    /// Creates a logger with no time limit, no iteration limit, and a print
    /// interval of `100`.
    /// # Examples
    /// The positional alternative to [`LoggerBuilder`](crate::LoggerBuilder):
    /// booster, weak learner, objective, loss, training sample, test sample.
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
    /// #     .join(format!("miniboosts-doc-logger-new-{}.csv", std::process::id()));
    /// use miniboosts::{Logger, LoggingSoftMarginObjective};
    ///
    /// let mut logger = Logger::new(
    ///     LpBoost::init(&train).nu(1.0),
    ///     DecisionTreeBuilder::new(&train).max_depth(1).build(),
    ///     LoggingSoftMarginObjective::new(1.0),
    ///     error_rate,
    ///     &train,
    ///     &test,
    /// )
    /// .print_every(usize::MAX);
    /// let classifier = logger.run(&path)?;
    /// std::fs::remove_file(&path)?;
    /// assert_eq!(classifier.predict_all(&test).len(), test.shape().0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(
        booster: B,
        weak_learner: W,
        objective_func: F,
        loss_func: G,
        train: &'a Sample,
        test: &'a Sample,
    ) -> Self {
        Self {
            booster,
            weak_learner,
            loss_func,
            objective_func,
            train,
            test,
            time_limit: DEFAULT_TIMELIMIT_MILLIS,
            round: DEFAULT_ROUND,
            max_iterations: usize::MAX,
        }
    }
}

impl<H, B, W, F, G, O, S> Logger<'_, B, W, F, G>
where
    B: Booster<H, Output = O> + CurrentHypothesis<Output = S>,
    O: Classifier,
    S: Classifier,
    W: WeakLearner<Hypothesis = H>,
    F: LoggingObjective,
    G: Fn(&Sample, &S) -> f64,
{
    /// Limit the number of boosting steps. The final step is logged before stopping.
    /// Defaults to no practical limit; algorithm and time limits still apply.
    ///
    /// # Panics
    /// Panics if `limit == 0`.
    /// # Examples
    /// The booster may stop earlier on its own, so at most `limit` rows are
    /// written:
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
    /// #     .join(format!("miniboosts-doc-logger-maxiter-{}.csv", std::process::id()));
    /// use miniboosts::{Logger, LoggingSoftMarginObjective};
    ///
    /// # let logger = Logger::new(
    /// #     LpBoost::init(&train).nu(1.0),
    /// #     DecisionTreeBuilder::new(&train).max_depth(1).build(),
    /// #     LoggingSoftMarginObjective::new(1.0),
    /// #     error_rate,
    /// #     &train,
    /// #     &test,
    /// # )
    /// # .print_every(usize::MAX);
    /// // `logger` was built with `Logger::new` (LPBoost, decision stumps).
    /// let mut logger = logger.max_iterations(3);
    /// logger.run(&path)?;
    /// let log = std::fs::read_to_string(&path)?;
    /// std::fs::remove_file(&path)?;
    /// let n_steps = log.lines().count() - 1; // minus the header
    /// assert!((1..=3).contains(&n_steps));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn max_iterations(mut self, limit: usize) -> Self {
        assert!(limit > 0, "iteration limit must be positive");
        self.max_iterations = limit;
        self
    }

    /// Set the time limit for boosting algorithm as milliseconds.
    /// Checked after each completed boosting step and evaluation; it does not interrupt a step.
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
    /// #     .join(format!("miniboosts-doc-logger-millis-{}.csv", std::process::id()));
    /// use miniboosts::{Logger, LoggingSoftMarginObjective};
    ///
    /// # let logger = Logger::new(
    /// #     LpBoost::init(&train).nu(1.0),
    /// #     DecisionTreeBuilder::new(&train).max_depth(1).build(),
    /// #     LoggingSoftMarginObjective::new(1.0),
    /// #     error_rate,
    /// #     &train,
    /// #     &test,
    /// # )
    /// # .print_every(usize::MAX);
    /// // `logger` was built with `Logger::new` (LPBoost, decision stumps).
    /// let mut logger = logger.time_limit_as_millis(0);
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
    /// Checked after each completed boosting step and evaluation; it does not interrupt a step.
    #[inline(always)]
    /// # Examples
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
    /// #     .join(format!("miniboosts-doc-logger-secs-{}.csv", std::process::id()));
    /// use miniboosts::{Logger, LoggingSoftMarginObjective};
    ///
    /// # let logger = Logger::new(
    /// #     LpBoost::init(&train).nu(1.0),
    /// #     DecisionTreeBuilder::new(&train).max_depth(1).build(),
    /// #     LoggingSoftMarginObjective::new(1.0),
    /// #     error_rate,
    /// #     &train,
    /// #     &test,
    /// # )
    /// # .print_every(usize::MAX);
    /// // `logger` was built with `Logger::new` (LPBoost, decision stumps).
    /// let mut logger = logger.time_limit_as_secs(30);
    /// // This small problem finishes long before the limit.
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
    /// Checked after each completed boosting step and evaluation; it does not interrupt a step.
    #[inline(always)]
    /// # Examples
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
    /// #     .join(format!("miniboosts-doc-logger-mins-{}.csv", std::process::id()));
    /// use miniboosts::{Logger, LoggingSoftMarginObjective};
    ///
    /// # let logger = Logger::new(
    /// #     LpBoost::init(&train).nu(1.0),
    /// #     DecisionTreeBuilder::new(&train).max_depth(1).build(),
    /// #     LoggingSoftMarginObjective::new(1.0),
    /// #     error_rate,
    /// #     &train,
    /// #     &test,
    /// # )
    /// # .print_every(usize::MAX);
    /// // `logger` was built with `Logger::new` (LPBoost, decision stumps).
    /// let mut logger = logger.time_limit_as_mins(5).max_iterations(100);
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

    #[inline(always)]
    fn print_log_header(&self) {
        println!(
            "      {:>WIDTH$}\t\t{:>WIDTH$}\t{:>WIDTH$}\t{:>WIDTH$}\t{:>WIDTH$}",
            "".bold().red(),
            "OBJ.".bold().blue(),
            "TRAIN".bold().green(),
            "TEST".bold().yellow(),
            "ACC.".bold().cyan(),
        );
        println!(
            "      {:>WIDTH$}\t\t{:>WIDTH$}\t{:>WIDTH$}\t{:>WIDTH$}\t{:>WIDTH$}\n",
            "ROUND".bold().red(),
            "VALUE".bold().blue(),
            "ERROR".bold().green(),
            "ERROR".bold().yellow(),
            "TIME".bold().cyan(),
        );
    }

    /// Print current settings in a rounded panel.
    fn print_stats(&self) {
        let limit = if self.time_limit != u128::MAX {
            time_format(self.time_limit)
        } else {
            "Unlimited".into()
        };
        let mut rows = vec![("Booster".to_owned(), self.booster.name().to_owned(), true)];
        if let Some(info) = self.booster.info() {
            rows.extend(
                info.into_iter()
                    .map(|(key, value)| (format!("  {key}"), value, false)),
            );
        }
        rows.push((String::new(), String::new(), false));
        rows.push(("Weak Learner".into(), self.weak_learner.name().into(), true));
        if let Some(info) = self.weak_learner.info() {
            rows.extend(
                info.into_iter()
                    .map(|(key, value)| (format!("  {key}"), value, false)),
            );
        }
        rows.push((String::new(), String::new(), false));
        rows.push(("Objective".into(), self.objective_func.name(), true));
        rows.push(("Time Limit".into(), limit, true));

        if self.max_iterations != usize::MAX {
            rows.push((
                "Iteration Limit".into(),
                self.max_iterations.to_string(),
                true,
            ));
        }

        // Compute padding before applying ANSI colors; avoid terminal-dependent tabs.
        let width = rows
            .iter()
            .map(|(key, value, _)| key.chars().count() + value.chars().count() + 2)
            .max()
            .unwrap_or(0)
            .max(FULL_WIDTH - 4);
        println!("\n╭─ {} {}╮", "STATS".bold(), "─".repeat(width - 6));
        for (key, value, heading) in rows {
            let padding = " ".repeat(width - key.chars().count() - value.chars().count());
            if heading {
                println!("│ {}{padding}{} │", key.bold(), value.green().bold());
            } else {
                println!("│ {key}{padding}{} │", value.yellow());
            }
        }
        println!("╰{}╯\n", "─".repeat(width + 2));
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
    /// #     .join(format!("miniboosts-doc-logger-print-{}.csv", std::process::id()));
    /// use miniboosts::{Logger, LoggingSoftMarginObjective};
    ///
    /// # let logger = Logger::new(
    /// #     LpBoost::init(&train).nu(1.0),
    /// #     DecisionTreeBuilder::new(&train).max_depth(1).build(),
    /// #     LoggingSoftMarginObjective::new(1.0),
    /// #     error_rate,
    /// #     &train,
    /// #     &test,
    /// # )
    /// # .print_every(usize::MAX);
    /// // `logger` was built with `Logger::new` (LPBoost, decision stumps).
    /// // A `[LOG]` line every 10 steps; `usize::MAX` keeps only the panel.
    /// let mut logger = logger.print_every(10);
    /// logger.run(&path)?;
    /// std::fs::remove_file(&path)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn print_every(mut self, round: usize) -> Self {
        assert!(round > 0, "print interval must be positive");
        self.round = round;
        self
    }

    /// Run the given boosting algorithm with logging and return the final
    /// classifier from [`Booster::postprocess`].
    ///
    /// This mirrors [`Booster::run`] with per-step logging:
    /// 1. creates (truncating) `filename` and writes the header
    ///    `ObjectiveValue,TrainLoss,TestLoss,Time`;
    /// 2. calls [`Booster::preprocess`] and prints the settings panel;
    /// 3. for `t = 1, 2, ...` up to the iteration limit: times
    ///    [`Booster::boost`], takes the current hypothesis, evaluates the
    ///    objective on the training sample and the loss on both samples, and
    ///    appends one CSV row; then stops if cumulative boosting time has
    ///    reached the time limit or the booster signaled a stop;
    /// 4. calls [`Booster::postprocess`].
    ///
    /// CSV time is cumulative boosting-step time in milliseconds (truncated).
    /// Setup, finalization, evaluation, and I/O are excluded from this budget.
    /// The step at which a limit is reached is logged, so the final time may
    /// exceed the limit.
    ///
    /// # Errors
    /// Returns an I/O error if the file cannot be created or written. An
    /// error after training has started returns early without calling
    /// [`Booster::postprocess`]. Panics from the booster, weak learner,
    /// objective, or loss propagate.
    #[inline(always)]
    /// # Examples
    /// Write the log under the system temporary directory, read it back, and
    /// remove it:
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
    /// #     .join(format!("miniboosts-doc-logger-run-unused-{}.csv", std::process::id()));
    /// use miniboosts::{Logger, LoggingSoftMarginObjective};
    ///
    /// # let logger = Logger::new(
    /// #     LpBoost::init(&train).nu(1.0),
    /// #     DecisionTreeBuilder::new(&train).max_depth(1).build(),
    /// #     LoggingSoftMarginObjective::new(1.0),
    /// #     error_rate,
    /// #     &train,
    /// #     &test,
    /// # )
    /// # .print_every(usize::MAX);
    /// // `logger` was built with `Logger::new` (LPBoost, decision stumps).
    /// # let mut logger = logger;
    /// let path = std::env::temp_dir()
    ///     .join(format!("miniboosts-doc-logger-run-{}.csv", std::process::id()));
    /// let classifier = logger.run(&path)?;
    /// let log = std::fs::read_to_string(&path)?;
    /// std::fs::remove_file(&path)?;
    ///
    /// let mut lines = log.lines();
    /// assert_eq!(lines.next(), Some("ObjectiveValue,TrainLoss,TestLoss,Time"));
    /// let times: Vec<u128> = lines
    ///     .map(|line| line.rsplit(',').next().unwrap().parse().unwrap())
    ///     .collect();
    /// assert!(!times.is_empty());
    /// // `Time` is cumulative milliseconds of `boost`, so it never decreases.
    /// assert!(times.windows(2).all(|t| t[0] <= t[1]));
    ///
    /// // The returned classifier is the booster's final output.
    /// let accuracy = 1.0 - error_rate(&test, &classifier);
    /// assert!((0.0..=1.0).contains(&accuracy));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn run<P: AsRef<Path>>(&mut self, filename: P) -> std::io::Result<O> {
        self.run_with_clock(filename, Instant::now)
    }

    fn run_with_clock<P: AsRef<Path>>(
        &mut self,
        filename: P,
        mut clock: impl FnMut() -> Instant,
    ) -> std::io::Result<O> {
        // Open file
        let mut file = File::create(filename)?;

        // Write header to the file
        file.write_all(HEADER.as_bytes())?;

        // ---------------------------------------------------------------------
        // Pre-processing
        self.booster.preprocess();
        self.print_stats();

        // Cumulative time
        let mut elapsed = Duration::ZERO;

        // ---------------------------------------------------------------------
        // Boosting step
        if self.round != usize::MAX {
            self.print_log_header();
        }
        for iter in 1..=self.max_iterations {
            // Start measuring time
            let now = clock();

            let flow = self.booster.boost(&self.weak_learner, iter);

            // Retain sub-millisecond precision across steps.
            elapsed += clock().duration_since(now);
            let time_acc = elapsed.as_millis();

            let f = self.booster.current_hypothesis();
            let obj = self.objective_func.objective_value(&self.train, &f);

            let train = (self.loss_func)(self.train, &f);
            let test = (self.loss_func)(self.test, &f);

            // Write the results to `file`.
            let line = format!("{obj},{train},{test},{time_acc}\n");
            file.write_all(line.as_bytes())?;

            if time_acc >= self.time_limit {
                println!(
                    "{} {}\t\t{}\t{}\t{}\t{}\n",
                    "[TLE]".bold().bright_red(),
                    format!("{:>WIDTH$}", iter).bold().red(),
                    format!("{:>WIDTH$.PREC_WIDTH$}", obj).bold().blue(),
                    format!("{:>WIDTH$.PREC_WIDTH$}", train).bold().green(),
                    format!("{:>WIDTH$.PREC_WIDTH$}", test).bold().yellow(),
                    time_format(time_acc).bold().cyan(),
                );
                break;
            }

            if self.round != usize::MAX && iter % self.round == 0 {
                println!(
                    "{} {}\t\t{}\t{}\t{}\t{}",
                    "[LOG]".bold().magenta(),
                    format!("{:>WIDTH$}", iter).red(),
                    format!("{:>WIDTH$.PREC_WIDTH$}", obj).blue(),
                    format!("{:>WIDTH$.PREC_WIDTH$}", train).green(),
                    format!("{:>WIDTH$.PREC_WIDTH$}", test).yellow(),
                    time_format(time_acc).bold().cyan(),
                );
            }

            let capped = iter == self.max_iterations && !flow.is_break();
            if (flow.is_break() || capped) && self.round != usize::MAX {
                println!(
                    "{} {}\t\t{}\t{}\t{}\t{}\n",
                    (if capped { "[LIMIT]" } else { "[FIN]" })
                        .bold()
                        .bright_green(),
                    format!("{:>WIDTH$}", iter).red(),
                    format!("{:>WIDTH$.PREC_WIDTH$}", obj).bold().blue(),
                    format!("{:>WIDTH$.PREC_WIDTH$}", train).bold().green(),
                    format!("{:>WIDTH$.PREC_WIDTH$}", test).bold().yellow(),
                    time_format(time_acc).bold().cyan(),
                );
            }
            if flow.is_break() {
                break;
            }
        }

        let f = self.booster.postprocess();
        Ok(f)
    }
}

fn time_format(millisec: u128) -> String {
    if millisec < 1_000 {
        return format!("  0.{:0>3}s", millisec);
    }
    let sec = millisec / 1_000;
    let millisec = millisec % 1_000;
    if sec < 60 {
        return format!(" {:0>2}.{:0>3}s", sec, millisec);
    }
    let min = sec / 60;
    let sec = sec % 60;
    if min < 60 {
        return format!(" {:0>2}m {:0>2}s", min, sec);
    }
    let hours = min / 60;
    let min = min % 60;
    format!(" {:0>2}h {:0>2}m", hours, min)
}

pub use crate::core::CurrentHypothesis;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::LoggingSoftMarginObjective;
    use std::ops::ControlFlow;

    #[derive(Clone)]
    struct Model;
    impl Classifier for Model {
        fn confidence(&self, sample: &Sample, row: usize) -> f64 {
            sample.target()[row]
        }
    }
    struct Learner;
    impl WeakLearner for Learner {
        type Hypothesis = Model;
        fn produce(&self, _: &Sample, _: &[f64]) -> Model {
            Model
        }
    }
    struct Steps(usize);
    impl Booster<Model> for Steps {
        type Output = Model;
        fn name(&self) -> &str {
            "test"
        }
        fn preprocess(&mut self) {
            self.0 = 0;
        }
        fn boost<W: WeakLearner<Hypothesis = Model>>(
            &mut self,
            _: &W,
            _: usize,
        ) -> ControlFlow<usize> {
            self.0 += 1;
            if self.0 == 5 {
                ControlFlow::Break(5)
            } else {
                ControlFlow::Continue(())
            }
        }
        fn postprocess(&mut self) -> Model {
            Model
        }
    }
    impl CurrentHypothesis for Steps {
        type Output = Model;
        fn current_hypothesis(&self) -> Model {
            Model
        }
    }

    #[test]
    fn iteration_limit_logs_last_step_and_preserves_early_stop() {
        let sample = Sample::dummy(2);
        for (limit, expected) in [(1, 1), (3, 3), (10, 5)] {
            let mut logger = Logger::new(
                Steps(0),
                Learner,
                LoggingSoftMarginObjective::new(1.0),
                |_: &Sample, _: &Model| 0.0,
                &sample,
                &sample,
            )
            .max_iterations(limit)
            .print_every(usize::MAX);
            let path = std::env::temp_dir().join(format!(
                "miniboosts-limit-{}-{limit}.csv",
                std::process::id()
            ));
            let now = Instant::now();
            let model = logger.run_with_clock(&path, || now).unwrap();
            assert_eq!(logger.booster.0, expected);
            assert_eq!(model.predict_all(&sample), vec![1, -1]);
            let csv = std::fs::read_to_string(&path).unwrap();
            std::fs::remove_file(path).unwrap();
            assert_eq!(csv.lines().count(), expected + 1);
        }
    }

    #[test]
    fn submillisecond_steps_accumulate_and_stop_at_budget() {
        let sample = Sample::dummy(2);
        let mut logger = Logger::new(
            Steps(0),
            Learner,
            LoggingSoftMarginObjective::new(1.0),
            |_: &Sample, _: &Model| 0.0,
            &sample,
            &sample,
        )
        .time_limit_as_millis(1)
        .print_every(usize::MAX);
        let path =
            std::env::temp_dir().join(format!("miniboosts-clock-{}.csv", std::process::id()));
        let mut now = Instant::now();
        let model = logger
            .run_with_clock(&path, || {
                let value = now;
                now += Duration::from_micros(600);
                value
            })
            .unwrap();
        assert_eq!(logger.booster.0, 2);
        assert_eq!(model.predict_all(&sample), vec![1, -1]);
        let csv = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        let rows: Vec<_> = csv.lines().collect();
        assert_eq!(rows[0], HEADER.trim());
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[1], "1,0,0,0");
        assert_eq!(rows[2], "1,0,0,1");
        // A directory is not a writable CSV file: propagate the error before training.
        assert!(logger.run(std::env::temp_dir()).is_err());
        assert_eq!(logger.booster.0, 2);
    }
}
