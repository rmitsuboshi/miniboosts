//! The regression tree's cached bins and batch prediction must not change
//! any value: gradient boosting and LogitBoost rely on both.

use miniboosts::hypotheses::{AdditiveModel, ScoreLink};
use miniboosts::{Classifier, RegressionTreeBuilder, Regressor, Sample, WeakLearner};

const N: u32 = 60;

/// 60 examples with 5 features: feature 0 has more distinct values than the
/// 8 bins used below (equal-width bins), the others at most 6 (one bin per
/// value). About half of the entries are zero.
fn rows() -> Vec<Vec<f64>> {
    (0..N)
        .map(|i| {
            (0..5_u32)
                .map(|k| {
                    if k == 0 {
                        f64::from(i * 37 % 61) * 0.125 - 3.0
                    } else if (i + k) % 2 == 0 {
                        f64::from((i * 7 + k * 3) % 5) * 0.5 + 0.5
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect()
}

fn responses() -> Vec<f64> {
    (0..N)
        .map(|i| f64::from(i * 13 % 17) * 0.25 - 2.0)
        .collect()
}

fn weights() -> Vec<f64> {
    let raw: Vec<f64> = (0..N).map(|i| f64::from(i % 7 + 1)).collect();
    let total: f64 = raw.iter().sum();
    raw.into_iter().map(|w| w / total).collect()
}

fn dense() -> Sample {
    Sample::from_dense(&rows(), Some(&vec![0.0; N as usize])).unwrap()
}

fn sparse() -> Sample {
    let (mut indptr, mut indices, mut data) = (vec![0], Vec::new(), Vec::new());
    for row in rows() {
        for (k, x) in row.into_iter().enumerate() {
            if x != 0.0 {
                indices.push(k);
                data.push(x);
            }
        }
        indptr.push(data.len());
    }
    Sample::from_csr(5, &indptr, &indices, &data, Some(&vec![0.0; N as usize])).unwrap()
}

#[test]
fn cached_bins_give_the_same_tree() {
    for (sample, rebuilt) in [(dense(), dense()), (sparse(), sparse())] {
        let tree = RegressionTreeBuilder::new(&sample)
            .max_depth(4)
            .n_bins(8)
            .build();
        // `with_target` shares the features of `sample`, so the bins stored
        // at `build` are used; `rebuilt` has equal features in another
        // allocation, whose values are binned again.
        let cached = tree.produce(&sample.with_target(responses()), &weights());
        let searched = tree.produce(&rebuilt.with_target(responses()), &weights());
        assert_eq!(cached, searched);
        // A tree with 4 levels on 60 distinct responses is not a single leaf.
        assert!(
            cached
                .predict_all(&sample)
                .iter()
                .any(|&v| v != cached.predict(&sample, 0))
        );
    }
}

#[test]
fn dense_and_sparse_features_give_the_same_tree() {
    let (dense, sparse) = (dense(), sparse());
    assert!(sparse.features().iter().all(|f| f.is_sparse()));
    let fit = |sample: &Sample| {
        RegressionTreeBuilder::new(sample)
            .max_depth(4)
            .n_bins(8)
            .build()
            .produce(&sample.with_target(responses()), &weights())
    };
    // Every sum runs over the same examples in the same order.
    assert_eq!(fit(&dense), fit(&sparse));
}

#[test]
fn batch_prediction_matches_rowwise() {
    for sample in [dense(), sparse()] {
        let fit = |depth: usize, scale: f64| {
            let target = responses().into_iter().map(|z| z * scale).collect();
            RegressionTreeBuilder::new(&sample)
                .max_depth(depth)
                .build()
                .produce(&sample.with_target(target), &weights())
        };
        let mut model = AdditiveModel::new(0.125, ScoreLink::Tanh);
        for (depth, scale) in [(1, 1.0), (3, -0.5), (0, 0.25), (4, 0.3)] {
            let h = fit(depth, scale);
            for (row, value) in h.predict_all(&sample).into_iter().enumerate() {
                assert_eq!(value.to_bits(), h.predict(&sample, row).to_bits());
            }
            model.push(0.7 / (depth as f64 + 1.0), h);
        }
        for link in [ScoreLink::Tanh, ScoreLink::Clamp] {
            model.link = link;
            let scores = Regressor::predict_all(&model, &sample);
            let confidences = model.confidence_all(&sample);
            for row in 0..N as usize {
                let score = Regressor::predict(&model, &sample, row);
                assert_eq!(scores[row].to_bits(), score.to_bits(), "row {row}");
                let confidence = model.confidence(&sample, row);
                assert_eq!(
                    confidences[row].to_bits(),
                    confidence.to_bits(),
                    "row {row}"
                );
            }
        }
    }
}
