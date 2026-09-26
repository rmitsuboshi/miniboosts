use miniboosts_core::{Booster, Classifier, Sample, WeakLearner};
use std::io::{BufReader, Cursor};

pub fn separable_sample() -> Sample {
    Sample::from_reader(
        BufReader::new(Cursor::new("x,class\n-2,-1\n-1,-1\n1,1\n2,1\n")),
        true,
    )
    .unwrap()
    .set_target("class")
}

pub fn check_training<B, W>(mut booster: B, learner: &W, sample: &Sample, max_loss: f64)
where
    W: WeakLearner,
    B: Booster<W::Hypothesis>,
    B::Output: Classifier,
{
    booster.preprocess();
    let stopped = (1..=1000).any(|iteration| booster.boost(learner, iteration).is_break());
    assert!(stopped, "booster exceeded the test's 1000-round budget");
    let model = booster.postprocess();
    let predictions = model.predict_all(sample);
    assert_eq!(predictions.len(), sample.shape().0);
    for row in 0..sample.shape().0 {
        assert!(model.confidence(sample, row).is_finite());
    }
    let errors = predictions
        .iter()
        .zip(sample.target())
        .filter(|(prediction, target)| **prediction as f64 != **target)
        .count();
    let loss = errors as f64 / sample.shape().0 as f64;
    assert!(loss <= max_loss, "training loss {loss} exceeds {max_loss}");
}
