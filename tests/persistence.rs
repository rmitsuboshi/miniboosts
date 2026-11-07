//! A saved hypothesis must predict exactly as the original after loading.
//!
//! The comparison is bitwise: JSON with `float_roundtrip` restores every `f64`
//! exactly, and evaluation is deterministic, so no tolerance is needed.

use miniboosts::hypotheses::MajorityVote;
use miniboosts::prelude::*;
use miniboosts::{
    AdditiveModel, ClassMember, DecisionTreeClassifier, LinearClassifier, NaiveAggregation,
    NaiveBayesClassifier, RbfHypothesis, RegressionTreeRegressor, Regressor, Sample,
    TableHypothesis, WeightedMajority,
};
use serde::{Serialize, de::DeserializeOwned};

// Labels follow the sign of `x + y` except for two flipped examples, so
// boosting needs several rounds.
fn train() -> Sample {
    let rows = vec![
        vec![-2.0, 1.0],
        vec![-1.0, -2.0],
        vec![1.0, -3.0],
        vec![-3.0, 2.0],
        vec![0.0, -1.0],
        vec![-2.0, -1.0],
        vec![0.5, 0.25],
        vec![2.0, 1.0],
        vec![1.0, 2.0],
        vec![-1.0, 3.0],
        vec![3.0, -1.0],
        vec![0.0, 2.0],
        vec![2.0, 2.0],
        vec![-0.5, -0.25],
    ];
    let target = [
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    Sample::from_dense(&rows, Some(&target)).unwrap()
}

// Points that were not used for training.
fn test() -> Sample {
    let rows = (0..25)
        .map(|i| vec![(i % 5) as f64 - 1.875, (i / 5) as f64 - 2.125])
        .collect::<Vec<_>>();
    Sample::from_dense(&rows, None).unwrap()
}

fn round_trip<H: Serialize + DeserializeOwned>(h: &H) -> H {
    let json = serde_json::to_string(h).unwrap();
    let loaded: H = serde_json::from_str(&json).unwrap();
    // Everything is restored, including what these samples do not reach.
    assert_eq!(serde_json::to_string(&loaded).unwrap(), json);
    loaded
}

fn bits(values: Vec<f64>) -> Vec<u64> {
    values.into_iter().map(f64::to_bits).collect()
}

fn assert_same_classifier<H>(h: &H)
where
    H: Classifier + Serialize + DeserializeOwned,
{
    let loaded = round_trip(h);
    for sample in [train(), test()] {
        assert_eq!(
            bits(h.confidence_all(&sample)),
            bits(loaded.confidence_all(&sample))
        );
    }
}

#[test]
fn weighted_majority_of_trees() {
    let sample = train();
    let tree = DecisionTreeBuilder::new(&sample).max_depth(2).build();
    let f: WeightedMajority<DecisionTreeClassifier> =
        AdaBoost::init(&sample).force_quit_at(20).run(&tree);
    assert!(f.hypotheses.len() > 1);
    assert_same_classifier(&f);
}

#[test]
fn weighted_majority_of_linear_classifiers() {
    let sample = train();
    let f: WeightedMajority<LinearClassifier> = AdaBoost::init(&sample)
        .force_quit_at(5)
        .run(&LogisticRegression::new().output(LinearOutput::Tanh));
    assert_same_classifier(&f);
}

#[test]
fn majority_vote() {
    let sample = train();
    let tree = DecisionTreeBuilder::new(&sample).max_depth(2).build();
    let f: MajorityVote<DecisionTreeClassifier> =
        BoostByMajority::init(&sample).gamma(0.2).run(&tree);
    assert_same_classifier(&f);
}

#[test]
fn additive_model_as_classifier_and_regressor() {
    let sample = train();
    let tree = RegressionTreeBuilder::new(&sample).max_depth(2).build();
    let f: AdditiveModel<RegressionTreeRegressor> =
        LogitBoost::init(&sample).force_quit_at(10).run(&tree);
    assert_same_classifier(&f);
    let loaded = round_trip(&f);
    let test = test();
    assert_eq!(
        bits(Regressor::predict_all(&f, &test)),
        bits(Regressor::predict_all(&loaded, &test))
    );
}

#[test]
fn naive_aggregation() {
    let sample = train();
    let hypotheses = (0..2)
        .map(|t| {
            RbfHypothesis::new(
                vec!["feature_0".to_string(), "feature_1".to_string()],
                vec![t as f64, 1.0 - t as f64],
                1.0,
            )
        })
        .collect::<Vec<_>>();
    let f = NaiveAggregation::new(hypotheses, &sample);
    assert_same_classifier(&f);
}

#[test]
fn naive_aggregation_rejects_a_bit_out_of_range() {
    let json = r#"{"hypotheses": [], "prediction": [[1, [1]]]}"#;
    let error = serde_json::from_str::<NaiveAggregation<RbfHypothesis>>(json)
        .err()
        .expect("the bit is out of range");
    assert!(error.to_string().contains("out of range"));
}

#[test]
fn members_of_finite_classes() {
    let sample = train();
    let dist = vec![1.0 / sample.shape().0 as f64; sample.shape().0];

    let learner = RbfClassBuilder::new(&sample).bandwidth(1.5).build();
    let h: ClassMember<RbfHypothesis> = learner.produce(&sample, &dist);
    assert_same_classifier(&h);

    let column = sample.target().iter().map(|y| 0.5 * y).collect::<Vec<_>>();
    let learner = FiniteClass::from_table(&sample, vec![column]);
    let h: ClassMember<TableHypothesis> = learner.produce(&sample, &dist);
    let loaded = round_trip(&h);
    assert_eq!(
        bits(h.confidence_all(&sample)),
        bits(loaded.confidence_all(&sample))
    );
}

#[test]
fn naive_bayes() {
    let sample = train();
    let dist = vec![1.0 / sample.shape().0 as f64; sample.shape().0];
    let h = NaiveBayes::new().produce(&sample, &dist);
    assert_same_classifier(&h);
}

#[test]
fn naive_bayes_with_a_label_without_weight() {
    let sample = train();
    for (label, odds) in [(1.0, f64::INFINITY), (-1.0, f64::NEG_INFINITY)] {
        let weight = sample
            .target()
            .iter()
            .map(|y| if *y == label { 1.0 / 7.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let h: NaiveBayesClassifier = NaiveBayes::new().produce(&sample, &weight);
        let loaded = round_trip(&h);
        assert_eq!(loaded, h);
        assert_eq!(loaded.log_odds(&sample, 0), odds);
        assert_eq!(loaded.confidence(&sample, 0), label);
    }
}

// A file written by 0.5.2 has no `only_label`.
#[test]
fn naive_bayes_from_a_file_without_the_only_label_field() {
    let json = r#"{
        "feature_names": ["feature_0"],
        "log_prior_ratio": 0.0,
        "positive": [[1.0, 1.0]],
        "negative": [[-1.0, 1.0]]
    }"#;
    let h: NaiveBayesClassifier = serde_json::from_str(json).unwrap();
    let midpoint = Sample::from_dense(&[vec![0.0]], None).unwrap();
    assert_eq!(h.log_odds(&midpoint, 0), 0.0);
}
