use polars::prelude::*;
use crate::{
    State,
    Booster,
    WeakLearner,
    CombinedHypothesis,
};
use super::logger::Logger;

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::time::Instant;

const HEADER: &str = "ObjectiveValue,TrainLoss,TestLoss,Time\n";


/// Run a boosting algorithm
/// with logging some informations to `file`.
/// Since each line of `file` corresponds to 
/// an iteration of boosting algorithm,
/// this function does not write the iteration.
pub fn with_log<B, W, H, F, P>(
    mut booster: B,
    weak_learner: W,
    loss_function: F,
    test_data: &DataFrame,
    test_target: &Series,
    file: P,
) -> std::io::Result<CombinedHypothesis<H>>
    where B: Booster<H> + Logger,
          W: WeakLearner<Hypothesis = H>,
          F: Fn(f64, f64) -> f64,
          P: AsRef<Path>,
{
    // Open file
    let mut file = File::create(file)?;

    // Write header to the file
    file.write_all(HEADER.as_bytes())?;


    // ---------------------------------------------------------------------
    // Pre-processing
    booster.preprocess(&weak_learner);


    // Cumulative time
    let mut time_acc = 0;

    // ---------------------------------------------------------------------
    // Boosting step
    for it in 1.. {
        // Start measuring time
        let now = Instant::now();

        let state = booster.boost(&weak_learner, it);

        // Stop measuring and convert `Duration` to Milliseconds.
        let time = now.elapsed().as_millis();

        // Update the cumulative time
        time_acc += time;

        booster.weights_on_hypotheses();

        // Get the objective value, train loss, and test loss
        let (obj, train, test) = booster.logging(
            &loss_function, test_data, test_target,
        );

        // Write the results to `file`.
        let line = format!("{obj},{train},{test},{time_acc}\n");
        file.write_all(line.as_bytes())?;


        if state == State::Terminate {
            break;
        }
    }


    // ---------------------------------------------------------------------
    // Post-process
    let f = booster.postprocess(&weak_learner);


    Ok(f)
}
