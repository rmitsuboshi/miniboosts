#[path = "../../../tests/support/mod.rs"]
mod support;

use adaboost::AdaBoost;
use decision_tree::DecisionTreeBuilder;

#[test]
fn learns_separable_sample() {
    let sample = support::separable_sample();
    let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    let booster = AdaBoost::init(&sample).tolerance(0.01);
    support::check_training(booster, &learner, &sample, 0.0);
}

#[test]
#[ignore = "larger dataset smoke test; run explicitly with --ignored"]
fn german() {
    let sample = miniboosts_core::SampleReader::default()
        .file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/dataset/german.csv"
        ))
        .has_header(true)
        .target_feature("class")
        .read()
        .unwrap();
    let learner = DecisionTreeBuilder::new(&sample).max_depth(2).build();
    let booster = AdaBoost::init(&sample).tolerance(0.01);
    support::check_training(booster, &learner, &sample, 0.5);
}
