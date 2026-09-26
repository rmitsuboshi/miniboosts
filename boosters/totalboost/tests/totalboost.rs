#[path = "../../../tests/support/mod.rs"]
mod support;

use decision_tree::DecisionTreeBuilder;
use totalboost::TotalBoost;

#[test]
fn learns_separable_sample() {
    let sample = support::separable_sample();
    let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    let booster = TotalBoost::init(&sample).tolerance(0.01);
    support::check_training(booster, &learner, &sample, 0.0);
}

#[test]
fn conflicting_labels_stop_with_a_finite_model() {
    use miniboosts_core::{Booster, Classifier, Sample};
    use std::io::{BufReader, Cursor};
    // Identical features with opposite labels cannot be separated.
    let sample = Sample::from_reader(BufReader::new(Cursor::new("x,class\n0,-1\n0,1\n")), true)
        .unwrap()
        .set_target("class");
    let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    let mut booster = TotalBoost::init(&sample).tolerance(0.01);
    booster.preprocess();
    assert!((1..=10).any(|i| booster.boost(&learner, i).is_break()));
    let model = booster.postprocess();
    assert!((0..2).all(|row| model.confidence(&sample, row).is_finite()));
    let errors = model
        .predict_all(&sample)
        .iter()
        .zip(sample.target())
        .filter(|(prediction, target)| **prediction as f64 != **target)
        .count();
    assert_eq!(errors, 1);
}
