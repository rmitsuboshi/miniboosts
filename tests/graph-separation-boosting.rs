#[path = "support/mod.rs"]
mod support;

use miniboosts::DecisionTreeBuilder;
use miniboosts::GraphSeparationBoosting;

#[test]
fn learns_separable_sample() {
    let sample = support::separable_sample();
    let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    let booster = GraphSeparationBoosting::init(&sample);
    support::check_training(booster, &learner, &sample, 0.0);
}
