//! Boosters accept hypothesis types that implement only `Classifier + Clone`.
use miniboosts::prelude::*;
use miniboosts::{Classifier, Sample};
use std::io::BufReader;

/// A threshold rule on the first feature; deliberately not `PartialEq`/`Debug`.
#[derive(Clone)]
struct Threshold(f64);

impl Classifier for Threshold {
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        if sample["x"][row] < self.0 { -1.0 } else { 1.0 }
    }
}

struct ThresholdLearner;

impl WeakLearner for ThresholdLearner {
    type Hypothesis = Threshold;
    fn name(&self) -> &str {
        "Threshold"
    }
    fn produce(&self, _: &Sample, _: &[f64]) -> Threshold {
        Threshold(0.0)
    }
}

#[test]
fn corrective_erlpboost_and_mlpboost_need_only_clone() {
    let sample = Sample::from_reader(
        BufReader::new(&b"x,class\n-2,-1\n-1,-1\n1,1\n2,1\n"[..]),
        true,
    )
    .unwrap()
    .set_target("class");
    let f = CorrectiveErlpBoost::init(&sample)
        .tolerance(0.1)
        .run(&ThresholdLearner);
    assert_eq!(f.predict_all(&sample), vec![-1, -1, 1, 1]);
    let f = MlpBoost::init(&sample)
        .tolerance(0.1)
        .run(&ThresholdLearner);
    assert_eq!(f.predict_all(&sample), vec![-1, -1, 1, 1]);
}
