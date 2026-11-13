//! A sample read from an SVMlight file stores its features sparsely. The
//! weak learners take shortcuts on sparse features, which must not change
//! what they learn or predict.

use miniboosts::weak_learners::linear::{
    BoundedLinear, LinearClassifier, LinearOutput, LogisticRegression,
};
use miniboosts::{Classifier, DecisionTreeBuilder, Sample, SampleReader, WeakLearner};

const N_ROWS: usize = 60;
const N_FEATURES: usize = 4;

/// About a quarter of the entries are nonzero; all values are exact in
/// binary.
fn rows() -> Vec<Vec<f64>> {
    (0..N_ROWS)
        .map(|i| {
            (0..N_FEATURES)
                .map(|j| {
                    if (i * 7 + j * 3) % 4 == 0 {
                        let magnitude = ((i + j) % 4 + 1) as f64 * 0.5;
                        if (i + 2 * j) % 3 == 0 {
                            -magnitude
                        } else {
                            magnitude
                        }
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect()
}

fn target() -> Vec<f64> {
    rows()
        .iter()
        .map(|x| {
            if x[0] - x[1] + 0.5 * x[3] > 0.0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

fn dist() -> Vec<f64> {
    let total = (N_ROWS * (N_ROWS + 1) / 2) as f64;
    (1..=N_ROWS).map(|i| i as f64 / total).collect()
}

fn dense_sample() -> Sample {
    Sample::from_dense(&rows(), Some(&target())).unwrap()
}

/// The same data as [`dense_sample`], written as SVMlight and read back.
fn sparse_sample(test: &str) -> Sample {
    let mut text = String::new();
    for (x, y) in rows().iter().zip(target()) {
        text.push_str(&format!("{y}"));
        for (j, v) in x.iter().enumerate() {
            if *v != 0.0 {
                text.push_str(&format!(" {}:{v}", j + 1));
            }
        }
        text.push('\n');
    }
    let path = std::env::temp_dir().join(format!(
        "miniboosts-sparse-{}-{test}.svmlight",
        std::process::id()
    ));
    std::fs::write(&path, text).unwrap();
    let sample = SampleReader::<_, &str>::default()
        .file(&path)
        .read()
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(sample.shape(), (N_ROWS, N_FEATURES));
    assert!(sample.features().iter().all(|f| f.is_sparse()));
    sample
}

fn names(sample: &Sample) -> Vec<String> {
    sample
        .features()
        .iter()
        .map(|f| f.name().to_string())
        .collect()
}

#[test]
fn linear_classifier_predicts_the_same_on_sparse_features() {
    let (dense, sparse) = (dense_sample(), sparse_sample("predict"));
    for output in [LinearOutput::Sign, LinearOutput::Tanh, LinearOutput::Clamp] {
        let weights = vec![0.3, -0.7, 0.0, 0.05];
        let on_dense = LinearClassifier::new(names(&dense), weights.clone(), -0.1, output);
        let on_sparse = LinearClassifier::new(names(&sparse), weights, -0.1, output);
        let expected = on_dense.confidence_all(&dense);
        let all = on_sparse.confidence_all(&sparse);
        for row in 0..N_ROWS {
            // `==`: leaving out zero terms may only change the sign of a zero.
            assert_eq!(all[row], expected[row], "row {row}");
            assert_eq!(on_sparse.confidence(&sparse, row), expected[row]);
        }
    }
}

#[test]
fn linear_learners_fit_the_same_on_sparse_features() {
    let (dense, sparse) = (dense_sample(), sparse_sample("fit"));
    let dist = dist();

    let on_dense = BoundedLinear::new().produce(&dense, &dist);
    let on_sparse = BoundedLinear::new().produce(&sparse, &dist);
    assert_eq!(on_sparse.weights(), on_dense.weights());
    assert_eq!(on_sparse.bias(), on_dense.bias());

    let learner = LogisticRegression::new().regularization(0.01);
    let on_dense = learner.produce(&dense, &dist);
    let on_sparse = learner.produce(&sparse, &dist);
    assert_eq!(on_sparse.weights(), on_dense.weights());
    assert_eq!(on_sparse.bias(), on_dense.bias());
}

#[test]
fn decision_tree_predicts_the_same_on_sparse_features() {
    let (dense, sparse) = (dense_sample(), sparse_sample("tree"));
    let dist = dist();
    for depth in [1, 3] {
        let on_dense = DecisionTreeBuilder::new(&dense)
            .max_depth(depth)
            .build()
            .produce(&dense, &dist);
        let on_sparse = DecisionTreeBuilder::new(&sparse)
            .max_depth(depth)
            .build()
            .produce(&sparse, &dist);
        let expected = on_dense.confidence_all(&dense);
        let all = on_sparse.confidence_all(&sparse);
        for row in 0..N_ROWS {
            // The split search rounds the weights of the bin containing 0.0
            // differently on sparse features; the trees must still agree.
            assert!((all[row] - expected[row]).abs() < 1e-13, "row {row}");
            assert_eq!(on_sparse.confidence(&sparse, row), all[row]);
        }
    }
}
