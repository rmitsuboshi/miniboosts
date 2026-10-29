#[path = "support/mod.rs"]
mod support;

use miniboosts::CorrectiveErlpBoost;
use miniboosts::DecisionTreeBuilder;

#[test]
fn learns_separable_sample() {
    let sample = support::separable_sample();
    let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    let booster = CorrectiveErlpBoost::init(&sample).tolerance(0.01).nu(1.0);
    support::check_training(booster, &learner, &sample, 0.0);
}

#[test]
#[ignore = "larger dataset smoke test; run explicitly with --ignored"]
fn german() {
    let sample = miniboosts::SampleReader::default()
        .file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/german.csv"
        ))
        .has_header(true)
        .target_feature("class")
        .read()
        .unwrap();
    let learner = DecisionTreeBuilder::new(&sample).max_depth(2).build();
    let booster = CorrectiveErlpBoost::init(&sample).tolerance(0.01).nu(1.0);
    support::check_training(booster, &learner, &sample, 0.5);
}
