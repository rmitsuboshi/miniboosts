//! Prediction traits: [`Classifier`] for confidence-rated binary classifiers
//! and [`Regressor`] for real-valued predictors.
use crate::core::Sample;
use std::sync::Arc;

/// A confidence-rated binary classifier.
///
/// Implementors only need to provide [`Classifier::confidence`]; the other
/// methods are derived from it.
///
/// Boosters in this crate assume that confidences lie in `[-1, 1]`
/// (**confidence-rated hypotheses**); the sign is the predicted label and the
/// magnitude is the confidence. This range is not checked by the trait.
/// For a fixed sample and row, `confidence` must return the same value on
/// every call; some boosters cache margins.
///
/// # Examples
/// A hand-written confidence-rated classifier: `clamp(x / scale, -1, 1)`.
/// ```
/// use std::io::BufReader;
/// use miniboosts::{Classifier, Sample};
///
/// struct Ramp {
///     scale: f64,
/// }
///
/// impl Classifier for Ramp {
///     fn confidence(&self, sample: &Sample, row: usize) -> f64 {
///         (sample["x"][row] / self.scale).clamp(-1.0, 1.0)
///     }
/// }
///
/// let csv = b"x,class\n-4,-1\n-1,-1\n0,1\n1,1\n4,1\n";
/// let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
///     .unwrap()
///     .set_target("class");
/// let h = Ramp { scale: 2.0 };
///
/// // One example at a time ...
/// assert_eq!(h.confidence(&sample, 1), -0.5);
/// assert_eq!(h.predict(&sample, 1), -1);
/// // ... or every example in row order.
/// assert_eq!(h.confidence_all(&sample), vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
/// // A confidence of exactly 0 predicts +1.
/// assert_eq!(h.predict_all(&sample), vec![-1, -1, 1, 1, 1]);
///
/// // Training accuracy from `predict_all`.
/// let correct = h
///     .predict_all(&sample)
///     .into_iter()
///     .zip(sample.target())
///     .filter(|&(p, &y)| p as f64 == y)
///     .count();
/// assert_eq!(correct, 5);
/// ```
pub trait Classifier {
    /// Returns the confidence value, in `[-1, 1]`, for the example at index
    /// `row` of `sample`.
    ///
    /// Implementations typically panic if `row` is out of bounds or if
    /// `sample` lacks a feature the hypothesis uses.
    fn confidence(&self, sample: &Sample, row: usize) -> f64;

    /// Predicts the label (`-1` or `+1`) of the example at index `row` of
    /// `sample`: `+1` if `confidence >= 0` and `-1` otherwise.
    /// A confidence of exactly `0` (or `-0.0`) is mapped to `+1`;
    /// a NaN confidence is mapped to `-1`.
    fn predict(&self, sample: &Sample, row: usize) -> i64 {
        let conf = self.confidence(sample, row);
        if conf >= 0.0 { 1 } else { -1 }
    }

    /// Returns [`Classifier::confidence`] for every example of `sample`,
    /// in row order.
    fn confidence_all(&self, sample: &Sample) -> Vec<f64> {
        let n_sample = sample.shape().0;
        (0..n_sample)
            .map(|row| self.confidence(sample, row))
            .collect::<Vec<_>>()
    }

    /// Returns [`Classifier::predict`] for every example of `sample`,
    /// in row order.
    fn predict_all(&self, sample: &Sample) -> Vec<i64> {
        let n_sample = sample.shape().0;
        (0..n_sample)
            .map(|row| self.predict(sample, row))
            .collect::<Vec<_>>()
    }
}

/// Forwards to the shared classifier, so that type-erased hypotheses such as
/// [`DynClassifier`](crate::weak_learners::union::DynClassifier) can be
/// combined by boosters.
///
/// # Examples
/// ```
/// use std::sync::Arc;
/// use miniboosts::{Classifier, Sample};
///
/// struct Positive;
/// impl Classifier for Positive {
///     fn confidence(&self, _: &Sample, _: usize) -> f64 {
///         0.5
///     }
/// }
///
/// let sample = Sample::dummy(1);
/// let h: Arc<dyn Classifier> = Arc::new(Positive);
/// assert_eq!(h.confidence(&sample, 0), 0.5);
/// assert_eq!(h.predict(&sample, 0), 1);
/// ```
impl<H: Classifier + ?Sized> Classifier for Arc<H> {
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        (**self).confidence(sample, row)
    }

    fn predict(&self, sample: &Sample, row: usize) -> i64 {
        (**self).predict(sample, row)
    }

    fn confidence_all(&self, sample: &Sample) -> Vec<f64> {
        (**self).confidence_all(sample)
    }

    fn predict_all(&self, sample: &Sample) -> Vec<i64> {
        (**self).predict_all(sample)
    }
}

/// A real-valued predictor.
///
/// Implementors only need to provide [`Regressor::predict`].
///
/// No booster in this crate trains regressors; the trait is implemented by
/// the combined hypotheses ([`crate::WeightedMajority`]) and by tree nodes
/// so that user-defined regression hypotheses can be combined.
///
/// # Examples
/// ```
/// use miniboosts::{Regressor, Sample, WeightedMajority};
///
/// // A linear model `w * x_0 + b`.
/// struct Linear {
///     w: f64,
///     b: f64,
/// }
///
/// impl Regressor for Linear {
///     fn predict(&self, sample: &Sample, row: usize) -> f64 {
///         self.w * sample["feature_0"][row] + self.b
///     }
/// }
///
/// let rows = vec![vec![0.0], vec![1.0], vec![2.0]];
/// let sample = Sample::from_dense(&rows, None).unwrap();
/// let f = Linear { w: 2.0, b: 1.0 };
/// assert_eq!(f.predict(&sample, 2), 5.0);
/// assert_eq!(f.predict_all(&sample), vec![1.0, 3.0, 5.0]);
///
/// // A weighted sum of regressors is again a regressor:
/// // 0.5 * (2x + 1) + 0.5 * (0x + 3) = x + 2.
/// let mut sum = WeightedMajority { weights: vec![], hypotheses: vec![] };
/// sum.push(0.5, f);
/// sum.push(0.5, Linear { w: 0.0, b: 3.0 });
/// assert_eq!(sum.predict_all(&sample), vec![2.0, 3.0, 4.0]);
/// ```
pub trait Regressor {
    /// Predicts the target value of the example at index `row` of `sample`.
    fn predict(&self, sample: &Sample, row: usize) -> f64;

    /// Returns [`Regressor::predict`] for every example of `sample`,
    /// in row order.
    fn predict_all(&self, sample: &Sample) -> Vec<f64> {
        let n_sample = sample.shape().0;
        (0..n_sample)
            .map(|row| self.predict(sample, row))
            .collect::<Vec<_>>()
    }
}
