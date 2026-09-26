use miniboosts::prelude::*;
use miniboosts::{FwUpdateRule, Logger, LoggingSoftMarginObjective, Sample};
use std::{error::Error, path::PathBuf};

const ALGORITHMS: &[&str] = &[
    "adaboost",
    "adaboostv",
    "madaboost",
    "smoothboost",
    "totalboost",
    "softboost",
    "lpboost",
    "erlpboost",
    "mlpboost",
    "cerlpboost",
    "graphsepboost",
];
const USAGE: &str = "Usage: miniboosts-example ALGORITHM TRAIN.csv TEST.csv OUTPUT.csv [OPTIONS]
       miniboosts-example --list
CSV inputs must have a header and a binary class column (-1, +1).
Options:
  --tolerance VALUE     Tolerance, or SmoothBoost kappa (default: 0.001)
  --nu VALUE            Capping parameter (default: max(1, 0.01 * training rows))
  --time-limit-ms VALUE  Cumulative boosting time limit (default: 60000)
  --max-iterations N    Maximum boosting steps (default: 500)
  --gamma VALUE         SmoothBoost weak-learner advantage (default: 0.006)
  --help                Show this help";

fn zero_one_loss<H: Classifier>(sample: &Sample, model: &H) -> f64 {
    model
        .predict_all(sample)
        .iter()
        .zip(sample.target())
        .filter(|(prediction, target)| **prediction as f64 != **target)
        .count() as f64
        / sample.shape().0 as f64
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    if args == ["--list"] {
        println!("{}", ALGORITHMS.join("\n"));
        return Ok(());
    }
    if args.len() < 4 {
        return Err(USAGE.into());
    }
    let algorithm = args[0].as_str();
    if !ALGORITHMS.contains(&algorithm) {
        return Err(format!("Unknown algorithm {algorithm:?}; use --list").into());
    }
    let mut tolerance: f64 = 0.001;
    let mut gamma: f64 = 0.006;
    let mut nu: Option<f64> = None;
    let mut time_limit: u128 = 60_000;
    let mut max_iterations: usize = 500;
    let mut options = args[4..].iter();
    while let Some(option) = options.next() {
        let value = options
            .next()
            .ok_or_else(|| format!("Missing value for {option}"))?;
        match option.as_str() {
            "--tolerance" => tolerance = value.parse()?,
            "--gamma" => gamma = value.parse()?,
            "--nu" => nu = Some(value.parse()?),
            "--max-iterations" => max_iterations = value.parse()?,
            "--time-limit-ms" => time_limit = value.parse()?,
            _ => return Err(format!("Unknown option {option}").into()),
        }
    }
    if !tolerance.is_finite() || !(0.0 < tolerance && tolerance < 1.0) {
        return Err("tolerance must be finite and in (0, 1)".into());
    }
    if !gamma.is_finite() || !(0.0 < gamma && gamma < 0.5) {
        return Err("gamma must be finite and in (0, 0.5)".into());
    }
    if max_iterations == 0 {
        return Err("max iterations must be positive".into());
    }
    if time_limit == 0 {
        return Err("time limit must be positive".into());
    }
    let train = SampleReader::default()
        .file(args[1].as_str())
        .has_header(true)
        .target_feature("class")
        .read()?;
    let test = SampleReader::default()
        .file(args[2].as_str())
        .has_header(true)
        .target_feature("class")
        .read()?;
    train.is_valid_binary_instance();
    test.is_valid_binary_instance();
    if !train
        .features()
        .iter()
        .map(|f| f.name())
        .eq(test.features().iter().map(|f| f.name()))
    {
        return Err("train and test must have the same feature columns in the same order".into());
    }
    let nu = nu.unwrap_or((0.01 * train.shape().0 as f64).max(1.0));
    if !nu.is_finite() || nu < 1.0 || nu > train.shape().0 as f64 {
        return Err("nu must be finite and between 1 and the training row count".into());
    }
    let output = PathBuf::from(&args[3]);
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    // Use the same weak learner and evaluation metric for every run.
    // The logged soft margin is not necessarily the algorithm's training objective.
    macro_rules! run {
        ($booster:expr) => {{
            let tree = DecisionTreeBuilder::new(&train)
                .max_depth(1)
                .split_by(SplitBy::Entropy)
                .build();
            Logger::new(
                $booster,
                tree,
                LoggingSoftMarginObjective::new(nu),
                zero_one_loss,
                &train,
                &test,
            )
            .max_iterations(max_iterations)
            .time_limit_as_millis(time_limit)
            .print_every(100)
            .run(&output)?;
        }};
    }
    match algorithm {
        "adaboost" => run!(AdaBoost::init(&train).tolerance(tolerance)),
        "adaboostv" => run!(AdaBoostV::init(&train).tolerance(tolerance)),
        "madaboost" => run!(MadaBoost::init(&train).tolerance(tolerance)),
        "smoothboost" => run!(SmoothBoost::init(&train).kappa(tolerance).gamma(gamma)),
        "totalboost" => run!(TotalBoost::init(&train).tolerance(tolerance)),
        "softboost" => run!(SoftBoost::init(&train).tolerance(tolerance).nu(nu)),
        "lpboost" => run!(LpBoost::init(&train).tolerance(tolerance).nu(nu)),
        "erlpboost" => run!(ErlpBoost::init(&train).tolerance(tolerance).nu(nu)),
        "mlpboost" => run!(
            MlpBoost::init(&train)
                .tolerance(tolerance)
                .nu(nu)
                .update_rule(FwUpdateRule::Classic)
        ),
        "cerlpboost" => run!(
            CorrectiveErlpBoost::init(&train)
                .tolerance(tolerance)
                .nu(nu)
        ),
        "graphsepboost" => run!(GraphSeparationBoosting::init(&train)),
        _ => unreachable!(),
    }
    Ok(())
}
