use miniboosts::{Booster, Classifier, CurrentHypothesis, Sample, WeakLearner};
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

/// The "Outliers" preset of the web demo (`main.js` in the homepage
/// repository): 100 points drawn with `mulberry32(20261101)`. Example `i` has
/// label `+1` if `i` is odd and `-1` otherwise, `x = -0.5·label + 0.2·g`,
/// `y = 0.35·g'` for standard normals `g, g'` from Box–Muller, and the label
/// is flipped with probability 0.08. Both coordinates are clipped to
/// `[-0.97, 0.97]`.
///
/// The generator is a port of the demo's JavaScript. `ln` and `cos` come from
/// the platform's libm, which may differ from V8's in the last bit; this
/// cannot change which side of a stump threshold a point falls on, so the
/// stumps' margins (all ±1) and the LPs built from them are the same.
#[allow(dead_code)]
pub fn outliers_sample() -> Sample {
    let mut seed: u32 = 20261101;
    let mut rand = move || {
        seed = seed.wrapping_add(0x6d2b_79f5);
        let mut t = (seed ^ (seed >> 15)).wrapping_mul(1 | seed);
        t = t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t)) ^ t;
        f64::from(t ^ (t >> 14)) / 4_294_967_296.0
    };
    let gaussian = |rand: &mut dyn FnMut() -> f64| {
        let u = 1.0 - rand();
        let v = rand();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    };
    let clip = |v: f64| v.clamp(-0.97, 0.97);
    let mut rows = Vec::new();
    let mut labels = Vec::new();
    for i in 0..100 {
        let label = if i % 2 == 1 { 1.0 } else { -1.0 };
        let x = -0.5 * label + 0.2 * gaussian(&mut rand);
        let y = 0.35 * gaussian(&mut rand);
        let flip = rand() < 0.08;
        rows.push(vec![clip(x), clip(y)]);
        labels.push(if flip { -label } else { label });
    }
    Sample::from_dense(&rows, Some(&labels)).unwrap()
}

/// Runs `booster` for at most `rounds` rounds and builds the current
/// ensemble after every round, as the web demo does, then the final one.
/// Every snapshot must give finite confidences.
#[allow(dead_code)]
pub fn check_snapshots<B, W>(mut booster: B, learner: &W, sample: &Sample, rounds: usize)
where
    W: WeakLearner,
    B: Booster<W::Hypothesis> + CurrentHypothesis,
    <B as Booster<W::Hypothesis>>::Output: Classifier,
    <B as CurrentHypothesis>::Output: Classifier,
{
    let finite = |model: &dyn Classifier| {
        (0..sample.shape().0).all(|row| model.confidence(sample, row).is_finite())
    };
    booster.preprocess();
    for round in 1..=rounds {
        let stop = booster.boost(learner, round).is_break();
        assert!(finite(&booster.current_hypothesis()), "round {round}");
        if stop {
            break;
        }
    }
    assert!(finite(&booster.postprocess()));
}
