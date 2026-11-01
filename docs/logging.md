## Logging learning curves

`LoggerBuilder` records the current ensemble after each boosting step. The
booster must implement `CurrentHypothesis`; a custom metric implements
`LoggingObjective`. `LoggingSoftMarginObjective` adapts the optimization
objective for logging (it is different from `SoftMarginObjective`).

The example below trains LPBoost with depth-1 decision trees on a small
two-dimensional data set, logs every step to a CSV file under the system
temporary directory, and reads the file back.

```rust
use std::io::BufReader;
use miniboosts::{
    prelude::*, Sample, LoggerBuilder, LoggingSoftMarginObjective,
};

/// Fraction of misclassified examples.
fn error_rate<H: Classifier>(sample: &Sample, classifier: &H) -> f64 {
    let n_errors = classifier.predict_all(sample).iter().zip(sample.target())
        .filter(|(prediction, target)| **prediction as f64 != **target)
        .count();
    n_errors as f64 / sample.shape().0 as f64
}

// 16 points labeled by the sign of `x + y`; no single stump separates them.
let csv = b"x,y,class\n\
    -2.0,0.5,-1\n-1.5,-1.0,-1\n-1.0,0.8,-1\n-0.5,-1.5,-1\n\
    0.5,-2.0,-1\n1.0,-1.8,-1\n-1.8,1.2,-1\n0.2,-0.9,-1\n\
    2.0,-0.5,1\n1.5,1.0,1\n1.0,-0.2,1\n0.5,1.5,1\n\
    -0.5,2.0,1\n-1.0,1.8,1\n1.8,0.4,1\n0.3,0.9,1\n";
let sample = Sample::from_reader(BufReader::new(&csv[..]), true)?
    .set_target("class");

// Hold out every fourth row as the test sample (used for reporting only).
let n = sample.shape().0;
let ix: Vec<usize> = (0..n).filter(|i| i % 4 != 0)
    .chain((0..n).filter(|i| i % 4 == 0))
    .collect();
let (train, test) = sample.split(&ix, n - n / 4, n);

let learner = DecisionTreeBuilder::new(&train)
    .max_depth(1).split_by(SplitBy::Entropy).build();
let booster = LpBoost::init(&train).nu(1.0).tolerance(0.01);
let mut logger = LoggerBuilder::new()
    .booster(booster)
    .weak_learner(learner)
    .objective_function(LoggingSoftMarginObjective::new(1.0))
    .loss_function(error_rate)
    .train_sample(&train)
    .test_sample(&test)
    .time_limit_as_secs(10)
    .print_every(5)
    .build()
    .max_iterations(50);

let path = std::env::temp_dir()
    .join(format!("miniboosts-doc-lpboost-{}.csv", std::process::id()));
let classifier = logger.run(&path)?;
let log = std::fs::read_to_string(&path)?;
std::fs::remove_file(&path)?;

// One header line, then one row per boosting step.
let mut lines = log.lines();
assert_eq!(lines.next(), Some("ObjectiveValue,TrainLoss,TestLoss,Time"));
let rows: Vec<Vec<f64>> = lines
    .map(|line| line.split(',').map(|v| v.parse().unwrap()).collect())
    .collect();
assert!(!rows.is_empty() && rows.len() <= 50);
for row in &rows {
    let (objective, train_loss, test_loss) = (row[0], row[1], row[2]);
    // The ensemble's confidences lie in [-1, 1], hence so does its margin.
    assert!((-1.0..=1.0).contains(&objective));
    assert!((0.0..=1.0).contains(&train_loss) && (0.0..=1.0).contains(&test_loss));
}
// `Time` is cumulative, so it never decreases.
assert!(rows.windows(2).all(|w| w[0][3] <= w[1][3]));

// `run` returns the final classifier from `postprocess`.
let test_error = error_rate(&test, &classifier);
assert!((0.0..=1.0).contains(&test_error));
# Ok::<(), Box<dyn std::error::Error>>(())
```

The output file is overwritten. CSV columns are `ObjectiveValue`, `TrainLoss`,
`TestLoss`, and `Time`. `Time` is cumulative boosting-step time in milliseconds,
including weak learning and optimization inside `boost`. It excludes preprocessing,
postprocessing, hypothesis snapshots, metric evaluation, printing, and file I/O.
The limit is checked between steps; it cannot interrupt a running learner or
solver. Measure end-to-end wall time separately when that is the comparison of
interest. Test data is used for reporting only.
