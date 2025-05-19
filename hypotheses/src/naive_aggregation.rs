use fixedbitset::FixedBitSet;
use miniboosts_core::{
    Classifier,
    Sample,
};

/// The naive aggregation rule.
/// See the following paper for example:
///
/// [Noga Alon, Alon Gonen, Elad Hazan, and Shay Moran. Boosting Simple Learners](https://theoretics.episciences.org/10757)
///
///
/// # Description
/// Given a set of hypotheses `{ h1, h2, ..., hT } \subset {-1, +1}^X` 
/// and training instances `(x1, y1), (x2, y2), ..., (xm, ym)`,
/// one can construct the following table:
///
/// ```txt
///              | h1(x)   h2(x)   h3(x) ... hT(x) | y
///     (x1, y1) |   +       -       +   ...   +   | -
///     (x2, y2) |   -       +       +   ...   -   | +
///        ...   |                       ...       |
///     (xm, ym) |   -       -       +   ...   -   | -
/// ```
///
/// Given a new instance `x`, 
/// we can get the following binary sequence of length `T`.
///
/// ```txt
/// B := h1(x) h2(x) ... hT(x) = +-+-....-+--+
/// ```
/// The following is the behavior of `NaiveAggregation`:
/// 1. If there exists an instance (xk, yk) such that
///    `h1(xk) h2(xk) ... hT(xk) == B` and `yk == +`,
///    the predict `+`.
/// 2. If there is no instance `(xk, yk)` satisfying the above condition,
///    the predict `-`.
/// 
pub struct NaiveAggregation<H> {
    /// `hypotheses` stores the functions from `X` to `{-1, +1}`
    /// collected by the boosting algorithm.
    hypotheses: Vec<H>,
    /// `prediction` stores a bit sequence
    /// `h1(xk) h2(xk) ... hT(xk)` whose label `yk` is `+1`.
    prediction: Vec<FixedBitSet>,
}

impl<H> NaiveAggregation<H>
    where H: Classifier
{
    /// Construct a new instance of `NaiveAggregation<H>`.
    #[inline(always)]
    pub fn new(
        hypotheses: Vec<H>,
        sample: &Sample,
    ) -> Self
    {
        let targets = sample.target();
        let n_hypotheses = hypotheses.len();
        let n_pos = targets.iter()
            .copied()
            .filter(|y| *y > 0.0)
            .count();
        let mut prediction = Vec::with_capacity(n_pos);

        let iter = targets.iter()
            .copied()
            .enumerate()
            .filter_map(|(i, y)| if y > 0.0 { Some(i) } else { None });
        for i in iter {
            let mut bits = FixedBitSet::with_capacity(n_hypotheses);
            hypotheses.iter()
                .enumerate()
                .for_each(|(t, h)| {
                    if h.predict(sample, i) > 0 {
                        bits.put(t);
                    }
                });
            prediction.push(bits);
        }
        Self { hypotheses, prediction }
    }
}

impl<H> NaiveAggregation<H>
    where H: Classifier + Clone
{
    /// Construct a new instance of `NaiveAggregation<H>`
    /// from a slice of hypotheses and `sample`.
    #[inline(always)]
    pub fn from_slice(
        hypotheses: &[H],
        sample: &Sample,
    ) -> Self
    {
        let hypotheses = hypotheses.to_vec();
        Self::new(hypotheses, sample)
    }
}

impl<H> Classifier for NaiveAggregation<H>
    where H: Classifier
{
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        let n_hypotheses = self.hypotheses.len();
        let mut bits = FixedBitSet::with_capacity(n_hypotheses);
        self.hypotheses.iter()
            .enumerate()
            .for_each(|(t, h)| {
                if h.predict(sample, row) == 1 {
                    bits.put(t);
                }
            });

        if self.prediction.iter().any(|p| p.eq(&bits)) { 1.0 } else { -1.0 }
    }
}

