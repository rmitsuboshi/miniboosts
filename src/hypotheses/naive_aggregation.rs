//! The lookup-table aggregation rule [`NaiveAggregation`].

use crate::core::{Classifier, Sample};
use fixedbitset::FixedBitSet;
use serde::{Deserialize, Serialize};

/// A lookup-table aggregation of binary hypotheses that is consistent with
/// the training sample whenever the hypotheses separate it.
///
/// This is the aggregation rule used in the proof of Lemma 4.2 of
/// Alon, Gonen, Hazan, and Moran,
/// [Boosting Simple Learners](https://theoretics.episciences.org/10757)
/// (TheoretiCS, 2023), and the output of
/// [`GraphSeparationBoosting`](crate::boosters::graph_separation_boosting::GraphSeparationBoosting).
///
/// # Description
///
/// Given hypotheses `h_1, ..., h_T` and training examples
/// `(x_1, y_1), ..., (x_m, y_m)`, record the prediction vector of every
/// example:
///
/// ```txt
///              | h1(x)   h2(x)   h3(x) ... hT(x) | y
///     (x1, y1) |   +       -       +   ...   +   | -
///     (x2, y2) |   -       +       +   ...   -   | +
///        ...   |                       ...       |
///     (xm, ym) |   -       -       +   ...   -   | -
/// ```
///
/// For a new instance `x`, compute `B = (h_1(x), ..., h_T(x))`. Then:
///
/// 1. If some training example `(x_k, y_k)` has `y_k = +1` and
///    `(h_1(x_k), ..., h_T(x_k)) == B`, predict `+1`.
/// 2. Otherwise, predict `-1`.
///
/// Each `h_t(x)` is [`Classifier::predict`], i.e., the sign of the confidence
/// with `0` mapped to `+1`. [`Classifier::confidence`] of this type returns
/// exactly `1.0` or `-1.0`.
///
/// If the hypotheses separate the sample (every pair with opposite labels
/// has different prediction vectors), the rule labels every training example
/// correctly. Otherwise, a negative example sharing a prediction vector with
/// a positive example is predicted `+1`. Unseen prediction vectors are always
/// predicted `-1`, so the rule is asymmetric in the labels.
///
/// The rule stores one bit set of length `T` per positive training example;
/// predicting one example costs `T` hypothesis evaluations plus a linear
/// scan over the stored bit sets.
///
/// # Examples
/// Two threshold rules jointly separate an interval of positives that no
/// single threshold separates.
/// ```
/// use miniboosts::{Classifier, NaiveAggregation, Sample};
///
/// // h(x) = +1 if x >= t, else -1.
/// #[derive(Clone)]
/// struct Above(f64);
/// impl Classifier for Above {
///     fn confidence(&self, sample: &Sample, row: usize) -> f64 {
///         if sample["feature_0"][row] >= self.0 { 1.0 } else { -1.0 }
///     }
/// }
///
/// // Positives exactly on [0, 2).
/// let rows = vec![vec![-2.0], vec![-1.0], vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
/// let y = [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0];
/// let sample = Sample::from_dense(&rows, Some(&y)).unwrap();
///
/// let f = NaiveAggregation::new(vec![Above(0.0), Above(2.0)], &sample);
/// // Positives have prediction vector (+1, -1); negatives (-1, -1) or
/// // (+1, +1). The hypotheses separate the sample, so f is consistent.
/// assert_eq!(f.predict_all(&sample), vec![-1, -1, 1, 1, -1, -1]);
/// // Confidences are exactly ±1.
/// assert!(f.confidence_all(&sample).iter().all(|c| c.abs() == 1.0));
/// ```
#[derive(Serialize, Deserialize)]
pub struct NaiveAggregation<H> {
    /// `hypotheses` stores the functions from `X` to `{-1, +1}`
    /// collected by the boosting algorithm.
    hypotheses: Vec<H>,
    /// `prediction` stores the bit sequence `h1(xk) h2(xk) ... hT(xk)`
    /// (bit `t` set iff `ht(xk) = +1`) of each example with `yk = +1`.
    #[serde(with = "bit_sequences")]
    prediction: Vec<FixedBitSet>,
}

// Stores each bit sequence as its length and the indices of its set bits.
// `FixedBitSet`'s own serde support cannot be read back from JSON.
mod bit_sequences {
    use fixedbitset::FixedBitSet;
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error};

    pub(super) fn serialize<S>(sequences: &[FixedBitSet], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        sequences
            .iter()
            .map(|bits| (bits.len(), bits.ones().collect::<Vec<_>>()))
            .collect::<Vec<_>>()
            .serialize(serializer)
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Vec<FixedBitSet>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Vec::<(usize, Vec<usize>)>::deserialize(deserializer)?
            .into_iter()
            .map(|(len, ones)| {
                let mut bits = FixedBitSet::with_capacity(len);
                for one in ones {
                    if one >= len {
                        return Err(D::Error::custom(format!(
                            "bit {one} is out of range for a sequence of length {len}"
                        )));
                    }
                    bits.put(one);
                }
                Ok(bits)
            })
            .collect()
    }
}

impl<H> NaiveAggregation<H>
where
    H: Classifier,
{
    /// Constructs the rule from `hypotheses` and the training `sample`.
    ///
    /// Examples with a positive target (`y > 0`) are treated as positive and
    /// their prediction vectors are stored; all others are treated as
    /// negative. Time complexity: `O(m T)` hypothesis evaluations for `m`
    /// examples and `T` hypotheses.
    ///
    /// # Examples
    /// When the hypotheses do **not** separate the sample, a negative example
    /// sharing a positive example's prediction vector is predicted `+1`.
    /// ```
    /// use miniboosts::{Classifier, NaiveAggregation, Sample};
    ///
    /// // One constant hypothesis: every example has the same vector.
    /// struct AlwaysPositive;
    /// impl Classifier for AlwaysPositive {
    ///     fn confidence(&self, _: &Sample, _: usize) -> f64 {
    ///         1.0
    ///     }
    /// }
    ///
    /// let sample = Sample::dummy(4); // labels +1, +1, -1, -1
    /// let f = NaiveAggregation::new(vec![AlwaysPositive], &sample);
    /// assert_eq!(f.predict_all(&sample), vec![1, 1, 1, 1]);
    /// ```
    #[inline(always)]
    pub fn new(hypotheses: Vec<H>, sample: &Sample) -> Self {
        let targets = sample.target();
        let n_hypotheses = hypotheses.len();
        let n_pos = targets.iter().copied().filter(|y| *y > 0.0).count();
        let mut prediction = Vec::with_capacity(n_pos);

        let iter = targets
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(i, y)| if y > 0.0 { Some(i) } else { None });
        for i in iter {
            let mut bits = FixedBitSet::with_capacity(n_hypotheses);
            hypotheses.iter().enumerate().for_each(|(t, h)| {
                if h.predict(sample, i) > 0 {
                    bits.put(t);
                }
            });
            prediction.push(bits);
        }
        Self {
            hypotheses,
            prediction,
        }
    }
}

impl<H> NaiveAggregation<H>
where
    H: Classifier + Clone,
{
    /// Constructs the rule from a slice of hypotheses (cloned) and the
    /// training `sample`. See [`NaiveAggregation::new`].
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{Classifier, NaiveAggregation, Sample};
    ///
    /// #[derive(Clone)]
    /// struct SignOfX;
    /// impl Classifier for SignOfX {
    ///     fn confidence(&self, sample: &Sample, row: usize) -> f64 {
    ///         sample["feature_0"][row].signum()
    ///     }
    /// }
    ///
    /// let rows = vec![vec![-1.0], vec![2.0]];
    /// let sample = Sample::from_dense(&rows, Some(&[-1.0, 1.0])).unwrap();
    /// let hypotheses = vec![SignOfX];
    /// let f = NaiveAggregation::from_slice(&hypotheses, &sample);
    /// assert_eq!(f.predict_all(&sample), vec![-1, 1]);
    /// assert_eq!(hypotheses.len(), 1); // the slice is still usable
    /// ```
    #[inline(always)]
    pub fn from_slice(hypotheses: &[H], sample: &Sample) -> Self {
        let hypotheses = hypotheses.to_vec();
        Self::new(hypotheses, sample)
    }
}

impl<H> Classifier for NaiveAggregation<H>
where
    H: Classifier,
{
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        let n_hypotheses = self.hypotheses.len();
        let mut bits = FixedBitSet::with_capacity(n_hypotheses);
        self.hypotheses.iter().enumerate().for_each(|(t, h)| {
            if h.predict(sample, row) == 1 {
                bits.put(t);
            }
        });

        if self.prediction.iter().any(|p| p.eq(&bits)) {
            1.0
        } else {
            -1.0
        }
    }
}
