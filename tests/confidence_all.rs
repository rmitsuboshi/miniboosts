//! `Classifier::confidence_all` must return exactly the row-wise
//! `Classifier::confidence` values: boosters compute margins with it.

use miniboosts::hypotheses::MajorityVote;
use miniboosts::weak_learners::finite_class::ClassMember;
use miniboosts::weak_learners::linear::{LinearClassifier, LinearOutput};
use miniboosts::weak_learners::naive_bayes::NaiveBayes;
use miniboosts::weak_learners::rbf::RbfHypothesis;
use miniboosts::{Classifier, Sample, WeakLearner};

/// 12 examples with 3 features; the values are exact in binary so that the
/// sample is the same on every platform.
fn sample() -> Sample {
    let rows: Vec<Vec<f64>> = (0..12)
        .map(|i| {
            let i = f64::from(i);
            vec![i * 0.25 - 1.5, (i * 3.0) % 5.0 - 2.0, 0.5 * i * i - 20.0]
        })
        .collect();
    let target: Vec<f64> = (0..12)
        .map(|i| if i % 3 == 0 { 1.0 } else { -1.0 })
        .collect();
    Sample::from_dense(&rows, Some(&target)).unwrap()
}

/// Bitwise equality: the batch path must not reorder any floating-point sum.
fn assert_matches_rowwise<H: Classifier>(h: &H, sample: &Sample) {
    let all = h.confidence_all(sample);
    assert_eq!(all.len(), sample.shape().0);
    for (row, batch) in all.iter().enumerate() {
        let single = h.confidence(sample, row);
        assert_eq!(batch.to_bits(), single.to_bits(), "row {row}");
    }
}

fn names(sample: &Sample) -> Vec<String> {
    sample
        .features()
        .iter()
        .map(|f| f.name().to_string())
        .collect()
}

#[test]
fn linear_classifier() {
    let sample = sample();
    for output in [LinearOutput::Sign, LinearOutput::Tanh, LinearOutput::Clamp] {
        // The features are given in a different order than the sample's.
        let mut names = names(&sample);
        names.reverse();
        let h = LinearClassifier::new(names, vec![0.3, -0.7, 0.05], 0.1, output);
        assert_matches_rowwise(&h, &sample);
    }
}

#[test]
fn rbf_hypothesis_and_class_member() {
    let sample = sample();
    let h = RbfHypothesis::new(names(&sample), vec![0.25, -1.0, 3.0], 2.0);
    assert_matches_rowwise(&h, &sample);
    let member = ClassMember {
        index: 0,
        sign: -0.75,
        hypothesis: h,
    };
    assert_matches_rowwise(&member, &sample);
}

#[test]
fn naive_bayes() {
    let sample = sample();
    let n = sample.shape().0;
    let dist: Vec<f64> = (1..=n)
        .map(|i| i as f64 / (n * (n + 1) / 2) as f64)
        .collect();
    let h = NaiveBayes::new().produce(&sample, &dist);
    assert_matches_rowwise(&h, &sample);

    // All weight on one label: the features are not read.
    let mut one_label = vec![0.0; n];
    one_label[0] = 1.0;
    let h = NaiveBayes::new().produce(&sample, &one_label);
    assert_matches_rowwise(&h, &sample);
}

#[test]
fn majority_vote() {
    let sample = sample();
    let names = names(&sample);
    let voters: Vec<LinearClassifier> = [(0.3, 0.1), (-0.2, 0.4), (1.0, -0.6)]
        .into_iter()
        .map(|(w, b)| {
            LinearClassifier::new(names.clone(), vec![w, 0.5, -0.01], b, LinearOutput::Tanh)
        })
        .collect();
    assert_matches_rowwise(&MajorityVote::new(voters), &sample);
    assert_matches_rowwise(&MajorityVote::<LinearClassifier>::new(Vec::new()), &sample);
}
