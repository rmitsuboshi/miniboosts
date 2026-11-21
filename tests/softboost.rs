#[path = "support/mod.rs"]
mod support;

use miniboosts::DecisionTreeBuilder;
use miniboosts::SoftBoost;

#[test]
fn learns_separable_sample() {
    let sample = support::separable_sample();
    let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    let booster = SoftBoost::init(&sample).tolerance(0.01).nu(1.0);
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
    let booster = SoftBoost::init(&sample).tolerance(0.01).nu(1.0);
    support::check_training(booster, &learner, &sample, 0.5);
}

/// Regression: on the web demo's "Outliers" data with `ν = 10` and stumps,
/// the soft-margin LP behind the round-15 snapshot is degenerate (optimal
/// value 0, 86 examples tied at margin 0). Clarabel stalled one step short
/// of its tolerances and returned `AlmostSolved` with a dual residual of
/// about `1e-7`, and `soft_margin_optimization` panicked.
#[test]
fn outliers_with_nu_10_does_not_panic() {
    let sample = support::outliers_sample();
    let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    let booster = SoftBoost::init(&sample).tolerance(0.01).nu(10.0);
    support::check_snapshots(booster, &learner, &sample, 300);
}
