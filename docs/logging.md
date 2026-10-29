## Logging learning curves

`LoggerBuilder` records the current ensemble after each boosting step. The
booster must implement `CurrentHypothesis`; a custom metric implements
`LoggingObjective`. `LoggingSoftMarginObjective` adapts the optimization
objective for logging (it is different from `SoftMarginObjective`).

```no_run
use miniboosts::{
    prelude::*, Sample, LoggerBuilder, LoggingSoftMarginObjective,
};

fn error_rate<H: Classifier>(sample: &Sample, classifier: &H) -> f64 {
    classifier.predict_all(sample).iter().zip(sample.target())
        .filter(|(prediction, target)| **prediction != **target as i64)
        .count() as f64 / sample.shape().0 as f64
}

let train = SampleReader::default().file("train.csv")
    .has_header(true).target_feature("class").read()?;
let test = SampleReader::default().file("test.csv")
    .has_header(true).target_feature("class").read()?;
let learner = DecisionTreeBuilder::new(&train)
    .max_depth(2).split_by(SplitBy::Entropy).build();
let booster = LpBoost::init(&train).nu(1.0).tolerance(0.01);
let mut logger = LoggerBuilder::new()
    .booster(booster)
    .weak_learner(learner)
    .objective_function(LoggingSoftMarginObjective::new(1.0))
    .loss_function(error_rate)
    .train_sample(&train)
    .test_sample(&test)
    .time_limit_as_secs(120)
    .print_every(10)
    .build();
let classifier = logger.run("lpboost.csv")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The output file is overwritten. CSV columns are `ObjectiveValue`, `TrainLoss`,
`TestLoss`, and `Time`. `Time` is cumulative boosting-step time in milliseconds,
including weak learning and optimization inside `boost`. It excludes preprocessing,
postprocessing, hypothesis snapshots, metric evaluation, printing, and file I/O.
The limit is checked between steps; it cannot interrupt a running learner or
solver. Measure end-to-end wall time separately when that is the comparison of
interest. Test data is used for reporting only.
