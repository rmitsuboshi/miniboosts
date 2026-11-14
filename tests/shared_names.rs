//! Hypotheses that record the names of their features hold one list shared
//! with the sample, not a copy each: an ensemble of `T` hypotheses over `d`
//! features stores `d` names, not `T d`.

use miniboosts::weak_learners::linear::{BoundedLinear, LinearClassifier, LogisticRegression};
use miniboosts::weak_learners::naive_bayes::NaiveBayes;
use miniboosts::weak_learners::rbf::{RbfClassBuilder, RbfHypothesis};
use miniboosts::{Classifier, Sample, WeakLearner};

fn sample() -> Sample {
    let rows = vec![
        vec![-2.0, 1.0],
        vec![-1.0, -1.5],
        vec![0.5, 2.0],
        vec![1.5, -0.5],
        vec![2.5, 0.25],
    ];
    Sample::from_dense(&rows, Some(&[-1.0, -1.0, 1.0, 1.0, 1.0])).unwrap()
}

fn same_allocation(a: &[String], b: &[String]) -> bool {
    std::ptr::eq(a.as_ptr(), b.as_ptr())
}

#[test]
fn hypotheses_of_one_sample_share_the_names() {
    let sample = sample();
    let uniform = [0.2; 5];
    let skewed = [0.4, 0.3, 0.1, 0.1, 0.1];
    let linear = [
        BoundedLinear::new().produce(&sample, &uniform),
        BoundedLinear::new().produce(&sample, &skewed),
        LogisticRegression::new().produce(&sample, &uniform),
        // Regression boosters fit on copies with another target.
        BoundedLinear::new().produce(
            &sample.with_target(vec![1.0, -1.0, 1.0, -1.0, 1.0]),
            &uniform,
        ),
        BoundedLinear::new().produce(&sample.clone(), &uniform),
    ];
    for h in &linear {
        assert_eq!(h.feature_names(), ["feature_0", "feature_1"]);
        assert!(same_allocation(
            h.feature_names(),
            linear[0].feature_names()
        ));
    }

    let class = RbfClassBuilder::new(&sample).bandwidth(1.0).build();
    let bumps = class.hypotheses();
    assert_eq!(bumps.len(), 5);
    for bump in bumps {
        assert!(same_allocation(
            bump.feature_names(),
            linear[0].feature_names()
        ));
    }
}

#[test]
fn renaming_the_features_does_not_reach_earlier_hypotheses() {
    let mut sample = sample();
    let uniform = [0.2; 5];
    let before = BoundedLinear::new().produce(&sample, &uniform);
    sample.replace_names(["a", "b"]);
    let after = BoundedLinear::new().produce(&sample, &uniform);
    assert_eq!(before.feature_names(), ["feature_0", "feature_1"]);
    assert_eq!(after.feature_names(), ["a", "b"]);
    assert_eq!(
        after.confidence_all(&sample),
        before.confidence_all(&self::sample())
    );
}

#[test]
fn the_serialized_form_is_a_plain_list_of_names() {
    let sample = sample();
    let uniform = [0.2; 5];
    let h = BoundedLinear::new().produce(&sample, &uniform);
    let json = serde_json::to_string(&h).unwrap();
    assert!(json.contains(r#""feature_names":["feature_0","feature_1"]"#));
    let back: LinearClassifier = serde_json::from_str(&json).unwrap();
    assert_eq!(back, h);
    // Built from an owned list of names, as before.
    let rebuilt = LinearClassifier::new(
        h.feature_names().to_vec(),
        h.weights().to_vec(),
        h.bias(),
        h.output(),
    );
    assert_eq!(rebuilt, h);

    let bayes = NaiveBayes::new().produce(&sample, &uniform);
    let json = serde_json::to_string(&bayes).unwrap();
    assert_eq!(
        serde_json::from_str::<miniboosts::NaiveBayesClassifier>(&json).unwrap(),
        bayes
    );

    let bump = RbfHypothesis::new(
        vec!["feature_0".into(), "feature_1".into()],
        vec![0.5, 2.0],
        1.0,
    );
    let json = serde_json::to_string(&bump).unwrap();
    assert_eq!(serde_json::from_str::<RbfHypothesis>(&json).unwrap(), bump);
    assert_eq!(bump.confidence(&sample, 2), 1.0);
}
