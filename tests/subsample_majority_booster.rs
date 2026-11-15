//! The sub-sample majorities as boosters, and as weak learners through
//! `BoostedLearner`.
use miniboosts::core::CurrentHypothesis;
use miniboosts::prelude::*;
use miniboosts::{MajorityVote, Sample};

// Positive inside [15, 35), negative outside, on 60 rows.
fn interval() -> (Sample, Vec<i64>) {
    let rows: Vec<Vec<f64>> = (0..60).map(|i| vec![f64::from(i)]).collect();
    let y: Vec<f64> = (0..60)
        .map(|i| if (15..35).contains(&i) { 1.0 } else { -1.0 })
        .collect();
    let labels = y.iter().map(|&v| v as i64).collect();
    (Sample::from_dense(&rows, Some(&y)).unwrap(), labels)
}

fn accuracy(predictions: &[i64], labels: &[i64]) -> usize {
    predictions
        .iter()
        .zip(labels)
        .filter(|(p, t)| p == t)
        .count()
}

#[test]
fn majority_of_five_as_a_booster_agrees_with_run_with() {
    let (sample, labels) = interval();
    // Naive Bayes keeps nothing from the sample it is built for, so both
    // ways train the same voters; the confidences are compared exactly.
    let learner = NaiveBayes::new();
    let mut booster = MajorityOfFive::init(&sample)
        .seed(3)
        .booster(|part, w| AdaBoost::init(part).force_quit_at(8).run(w));
    let f = booster.run(&learner);
    let g = MajorityOfFive::init(&sample)
        .seed(3)
        .run_with(|part: &Sample| {
            AdaBoost::init(part)
                .force_quit_at(8)
                .run(&NaiveBayes::new())
        });
    assert_eq!(f.hypotheses.len(), 5);
    assert_eq!(f.confidence_all(&sample), g.confidence_all(&sample));
    assert!(accuracy(&f.predict_all(&sample), &labels) >= 50);

    // A second run starts afresh.
    let again = booster.run(&learner);
    assert_eq!(again.hypotheses.len(), 5);
    assert_eq!(again.confidence_all(&sample), f.confidence_all(&sample));
}

#[test]
fn rounds_are_the_parts() {
    let (sample, _) = interval();
    let stump = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    let mut booster = MajorityOfFive::init(&sample)
        .n_parts(3)
        .booster(|part, w| AdaBoost::init(part).force_quit_at(4).run(w));
    assert_eq!(booster.name(), "Majority-of-5");
    booster.preprocess();
    assert!(booster.current_hypothesis().hypotheses.is_empty());
    assert!(booster.boost(&stump, 1).is_continue());
    assert_eq!(booster.current_hypothesis().hypotheses.len(), 1);
    assert!(booster.boost(&stump, 2).is_continue());
    // The last part breaks with the number of parts.
    assert_eq!(booster.boost(&stump, 3).break_value(), Some(3));
    let snapshot = booster.current_hypothesis();
    let f: MajorityVote<_> = booster.postprocess();
    assert_eq!(f.hypotheses.len(), 3);
    assert_eq!(snapshot.confidence_all(&sample), f.confidence_all(&sample));
}

#[test]
fn larsen_ritzert_as_a_booster_agrees_with_run_with() {
    let (sample, labels) = interval();
    let learner = NaiveBayes::new();
    let mut booster = LarsenRitzert::init(&sample)
        .seed(2)
        .max_subsamples(7)
        .booster(|sub, w| AdaBoost::init(sub).force_quit_at(8).run(w));
    assert_eq!(booster.name(), "Larsen-Ritzert");
    let f = booster.run(&learner);
    let g = LarsenRitzert::init(&sample)
        .seed(2)
        .max_subsamples(7)
        .run_with(|sub: &Sample| AdaBoost::init(sub).force_quit_at(8).run(&NaiveBayes::new()));
    assert_eq!(f.hypotheses.len(), 7);
    assert_eq!(f.confidence_all(&sample), g.confidence_all(&sample));
    assert!(accuracy(&f.predict_all(&sample), &labels) >= 50);
}
