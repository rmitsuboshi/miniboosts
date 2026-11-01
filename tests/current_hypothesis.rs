//! `CurrentHypothesis` snapshots before the first round are empty ensembles.
use miniboosts::prelude::*;
use miniboosts::{Classifier, CurrentHypothesis, DecisionTreeClassifier, Sample, WeightedMajority};
use std::io::BufReader;

fn separable_sample() -> Sample {
    Sample::from_reader(
        BufReader::new(&b"x,class\n-2,-1\n-1,-1\n1,1\n2,1\n"[..]),
        true,
    )
    .unwrap()
    .set_target("class")
}

fn assert_empty_before_first_round<B>(mut booster: B, sample: &Sample)
where
    B: Booster<DecisionTreeClassifier>
        + CurrentHypothesis<Output = WeightedMajority<DecisionTreeClassifier>>,
{
    booster.preprocess();
    let f = booster.current_hypothesis();
    assert!(f.weights.is_empty() && f.hypotheses.is_empty());
    // The empty ensemble has confidence 0 and therefore predicts +1.
    assert_eq!(f.confidence_all(sample), vec![0.0; sample.shape().0]);
    assert_eq!(f.predict_all(sample), vec![1; sample.shape().0]);
}

#[test]
fn snapshots_before_the_first_round_are_empty() {
    let sample = separable_sample();
    assert_empty_before_first_round(AdaBoost::init(&sample), &sample);
    assert_empty_before_first_round(AdaBoostV::init(&sample), &sample);
    assert_empty_before_first_round(MadaBoost::init(&sample), &sample);
    assert_empty_before_first_round(LpBoost::init(&sample), &sample);
    assert_empty_before_first_round(ErlpBoost::init(&sample), &sample);
    assert_empty_before_first_round(CorrectiveErlpBoost::init(&sample), &sample);
    assert_empty_before_first_round(MlpBoost::init(&sample), &sample);
    assert_empty_before_first_round(SmoothBoost::init(&sample), &sample);
    assert_empty_before_first_round(SoftBoost::init(&sample), &sample);
    assert_empty_before_first_round(TotalBoost::init(&sample), &sample);
}

#[test]
fn zero_round_budget_returns_the_empty_ensemble() {
    let sample = separable_sample();
    let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    for f in [
        AdaBoost::init(&sample).force_quit_at(0).run(&learner),
        AdaBoostV::init(&sample).force_quit_at(0).run(&learner),
        MadaBoost::init(&sample).force_quit_at(0).run(&learner),
    ] {
        assert!(f.weights.is_empty());
        assert_eq!(f.predict_all(&sample), vec![1; 4]);
    }
}
