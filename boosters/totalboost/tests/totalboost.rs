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
