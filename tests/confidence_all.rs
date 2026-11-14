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

/// 40 examples with 6 features, about a third of the entries nonzero and
/// feature 4 all zero, as a dense sample and as a sparse one (from CSR).
fn dense_and_sparse() -> (Sample, Sample) {
    let rows: Vec<Vec<f64>> = (0..40_u32)
        .map(|i| {
            (0..6_u32)
                .map(|k| {
                    if k != 4 && (i * 7 + k * 5) % 3 == 0 {
                        f64::from((i * 11 + k * 13) % 17) * 0.25 - 2.0
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    let target: Vec<f64> = (0..40)
        .map(|i| if i % 3 == 0 { 1.0 } else { -1.0 })
        .collect();
    let (mut indptr, mut indices, mut data) = (vec![0], Vec::new(), Vec::new());
    for row in &rows {
        for (k, &x) in row.iter().enumerate() {
            if x != 0.0 {
                indices.push(k);
                data.push(x);
            }
        }
        indptr.push(data.len());
    }
    let dense = Sample::from_dense(&rows, Some(&target)).unwrap();
    let sparse = Sample::from_csr(6, &indptr, &indices, &data, Some(&target)).unwrap();
    (dense, sparse)
}

/// On sparse features the terms at zero are added once for all rows, which
/// reorders the sums: the batch and row-wise values still agree bitwise, and
/// both agree with the dense sample up to rounding.
fn assert_sparse_matches_dense<H: Classifier>(h: &H, dense: &Sample, sparse: &Sample) {
    assert!(sparse.features().iter().all(|f| f.is_sparse()));
    assert_matches_rowwise(h, sparse);
    let expected = h.confidence_all(dense);
    for (row, value) in h.confidence_all(sparse).into_iter().enumerate() {
        // Sums of at most 7 terms of magnitude below 100.
        assert!((value - expected[row]).abs() < 1e-13, "row {row}");
        assert!(value.abs() <= 1.0, "row {row}");
    }
}

#[test]
fn naive_bayes_on_sparse_features() {
    let (dense, sparse) = dense_and_sparse();
    let dist = vec![1.0 / 40.0; 40];
    let from_dense = NaiveBayes::new().produce(&dense, &dist);
    assert_sparse_matches_dense(&from_dense, &dense, &sparse);
    // Trained on the sparse sample: the same model up to rounding.
    let from_sparse = NaiveBayes::new().produce(&sparse, &dist);
    assert_sparse_matches_dense(&from_sparse, &dense, &sparse);
    let expected = from_dense.confidence_all(&dense);
    for (row, value) in from_sparse.confidence_all(&dense).into_iter().enumerate() {
        assert!((value - expected[row]).abs() < 1e-13, "row {row}");
    }
}

#[test]
fn rbf_on_sparse_features() {
    let (dense, sparse) = dense_and_sparse();
    let h = RbfHypothesis::new(names(&dense), vec![0.25, -1.0, 0.0, 3.0, 0.0, -0.5], 2.0);
    assert_sparse_matches_dense(&h, &dense, &sparse);
    // Centered at an example: the squared distance cancels to about 0 there
    // and must not become negative.
    let (center, _) = dense.at(3);
    let h = RbfHypothesis::new(names(&dense), center, 0.5);
    assert_sparse_matches_dense(&h, &dense, &sparse);
    assert!((h.confidence(&sparse, 3) - 1.0).abs() < 1e-13);
}
