//! An unweighted majority vote over the hypotheses' predicted labels.

use crate::core::{Classifier, Sample};
use serde::{Deserialize, Serialize};

/// The unweighted majority vote `x ↦ (1/T) Σ_t pred_t(x)`, where
/// `pred_t(x) ∈ {-1, +1}` is [`Classifier::predict`] of hypothesis `t`
/// (which maps a confidence of `0` to `+1`).
///
/// Its confidence is the average vote, in `[-1, 1]`; a tie gives `0`, which
/// `predict` maps to `+1`. An empty vote has confidence `0`. Returned by
/// [`BoostByMajority`](crate::BoostByMajority).
///
/// # Examples
/// ```
/// use miniboosts::{Classifier, MajorityVote, Sample};
///
/// struct Constant(f64);
/// impl Classifier for Constant {
///     fn confidence(&self, _: &Sample, _: usize) -> f64 {
///         self.0
///     }
/// }
///
/// let sample = Sample::dummy(1);
/// // Votes +1, +1, -1: confidences enter only through their signs.
/// let f = MajorityVote::new(vec![Constant(0.1), Constant(0.9), Constant(-0.2)]);
/// assert!((f.confidence(&sample, 0) - 1.0 / 3.0).abs() < 1e-15);
/// assert_eq!(f.predict(&sample, 0), 1);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MajorityVote<H> {
    /// The voting hypotheses.
    pub hypotheses: Vec<H>,
}

impl<H> MajorityVote<H> {
    /// Constructs the vote over `hypotheses`.
    ///
    /// # Examples
    /// See [`MajorityVote`].
    pub fn new(hypotheses: Vec<H>) -> Self {
        Self { hypotheses }
    }
}

impl<H: Classifier> Classifier for MajorityVote<H> {
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        if self.hypotheses.is_empty() {
            return 0.0;
        }
        let votes: i64 = self.hypotheses.iter().map(|h| h.predict(sample, row)).sum();
        votes as f64 / self.hypotheses.len() as f64
    }
}
