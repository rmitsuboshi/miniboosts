use crate::{
    Sample,
    Booster,
    WeakLearner,
    CombinedHypothesis,
    common::ObjectiveFunction,
};

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::time::Instant;

const HEADER: &str = "ObjectiveValue,TrainLoss,TestLoss,Time\n";


/// Struct `Logger` provides a generic function that
/// logs objective value, train / test loss value
/// for each step of boosting.
pub struct Logger<'a, B, W, F, G> {
    booster: B,
    weak_learner: W,
    objective_func: F,
    loss_func: G,
    train: &'a Sample,
    test: &'a Sample,
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
    ) -> Self
    {
        Self { booster, weak_learner, loss_func, objective_func, train, test }
    }
}

impl<'a, H, B, W, F, G> Logger<'a, B, W, F, G>
    where B: Booster<H> + Research<H>,
          W: WeakLearner<Hypothesis = H>,
          F: ObjectiveFunction<H>,
          G: Fn(&Sample, &CombinedHypothesis<H>) -> f64,
{
    /// Run the given boosting algorithm with logging.
    /// Note that this method is almost the same as `Booster::run`.
    /// This method measures running time per iteration.
    pub fn run<P: AsRef<Path>>(&mut self, filename: P)
        -> std::io::Result<CombinedHypothesis<H>>
    {
        // Open file
        let mut file = File::create(filename)?;

        // Write header to the file
        file.write_all(HEADER.as_bytes())?;


        // ---------------------------------------------------------------------
        // Pre-processing
        self.booster.preprocess(&self.weak_learner);


        // Cumulative time
        let mut time_acc = 0;

        // ---------------------------------------------------------------------
        // Boosting step
        (1..).try_for_each(|iter| {
            // Start measuring time
            let now = Instant::now();

            let flow = self.booster.boost(&self.weak_learner, iter);

            // Stop measuring and convert `Duration` to Milliseconds.
            let time = now.elapsed().as_millis();

            // Update the cumulative time
            time_acc += time;

            let hypothesis = self.booster.current_hypothesis();

            let obj = self.objective_func.eval(self.train, &hypothesis);
            let train = (self.loss_func)(self.train, &hypothesis);
            let test = (self.loss_func)(self.test, &hypothesis);

            // Write the results to `file`.
            let line = format!("{obj},{train},{test},{time_acc}\n");
            file.write_all(line.as_bytes())
                .expect("Failed to writing {filename:?}");

            flow
        });

        let f = self.booster.postprocess(&self.weak_learner);
        Ok(f)
    }
}


/// Implementing this trait allows you to use `Logger` to
/// log algorithm's behavor.
pub trait Research<H> {
    /// Returns the combined hypothesis at current state.
    fn current_hypothesis(&self) -> CombinedHypothesis<H>;
}


