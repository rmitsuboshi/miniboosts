use colored::Colorize;

use crate::objective::LoggingObjective;
use miniboosts_core::{Booster, Classifier, Sample, WeakLearner};

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::time::{Duration, Instant};

const DEFAULT_ROUND: usize = 100;
const DEFAULT_TIMELIMIT_MILLIS: u128 = u128::MAX;
const WIDTH: usize = 8;
const PREC_WIDTH: usize = 5;
const FULL_WIDTH: usize = 60;
const STAT_WIDTH: usize = (FULL_WIDTH - 4) / 2;
const HEADER: &str = "ObjectiveValue,TrainLoss,TestLoss,Time\n";

/// Struct `Logger` provides a generic function that
/// logs objective value, train/test loss value, and running time
/// for each step of boosting.
pub struct Logger<'a, B, W, F, G> {
    pub(super) booster: B,
    pub(super) weak_learner: W,
    pub(super) objective_func: F,
    pub(super) loss_func: G,
    pub(super) train: &'a Sample,
    pub(super) test: &'a Sample,
    pub(super) time_limit: u128,
    pub(super) round: usize,
}

impl<'a, B, W, F, G> Logger<'a, B, W, F, G> {
    /// Create a new instance of `Logger`.
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
    /// Set the time limit for boosting algorithm as milliseconds.
    /// Checked after each completed boosting step and evaluation; it does not interrupt a step.
    #[inline(always)]
    pub fn time_limit_as_millis(mut self, time_limit: u128) -> Self {
        self.time_limit = time_limit;
        self
    }

    /// Set the time limit for boosting algorithm as seconds.
    /// Checked after each completed boosting step and evaluation; it does not interrupt a step.
    #[inline(always)]
    pub fn time_limit_as_secs(mut self, time_limit: u64) -> Self {
        self.time_limit = (time_limit as u128)
            .checked_mul(1_000_u128)
            .expect("The time limit (ms) cannot be represented as u128");
        self
    }

    /// Set the time limit for boosting algorithm as minutes.
    /// Checked after each completed boosting step and evaluation; it does not interrupt a step.
    #[inline(always)]
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

    /// print current settings.
    #[inline(always)]
    fn print_stats(&self) {
        let limit = if self.time_limit != u128::MAX {
            time_format(self.time_limit)
        } else {
            "Nothing".into()
        };
        let header = format!(
            "{:=>FULL_WIDTH$}\n{:^FULL_WIDTH$}\n{:->FULL_WIDTH$}",
            "",
            "STATS".bold(),
            "",
        );
        println!(
            "\n{header}\n\
            + {:<STAT_WIDTH$}\t{:>STAT_WIDTH$}",
            "Booster".bold(),
            self.booster.name().bold().green(),
        );

        if let Some(info) = self.booster.info() {
            let line = info
                .into_iter()
                .map(|(key, val)| {
                    format!(
                        "    + {:<STAT_WIDTH$}\t{:>width$}",
                        key,
                        val.bold().yellow(),
                        width = STAT_WIDTH - 8
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            println!("{line}");
        }

        println!(
            "+ {:<STAT_WIDTH$}\t{:>STAT_WIDTH$}",
            "Weak Learner".bold(),
            self.weak_learner.name().bold().green(),
        );
        if let Some(info) = self.weak_learner.info() {
            let line = info
                .into_iter()
                .map(|(key, val)| {
                    format!(
                        "    + {:<STAT_WIDTH$}\t{:>width$}",
                        key,
                        val.bold().yellow(),
                        width = STAT_WIDTH - 8
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            println!("{line}");
        }
        println!(
            "\
            + {:<STAT_WIDTH$}\t{:>STAT_WIDTH$}\n\
            + {:<STAT_WIDTH$}\t{:>STAT_WIDTH$}\n\
            {:=^FULL_WIDTH$}\n\
            ",
            "Objective".bold(),
            self.objective_func.name().bold().green(),
            "Time Limit".bold(),
            limit.bold().green(),
            "".bold(),
        );
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

    /// Run the given boosting algorithm with logging.
    /// Note that this method is almost the same as `Booster::run`.
    /// CSV time is cumulative boosting-step time in milliseconds.
    /// Setup, finalization, evaluation, and I/O are excluded from this budget.
    #[inline(always)]
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
        for iter in 1.. {
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

            if flow.is_break() && self.round != usize::MAX {
                println!(
                    "{} {}\t\t{}\t{}\t{}\t{}\n",
                    "[FIN]".bold().bright_green(),
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

pub use miniboosts_core::CurrentHypothesis;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LoggingSoftMarginObjective;
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
