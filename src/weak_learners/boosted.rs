//! A weak learner that runs a booster: [`BoostedLearner`].
//!
//! Boosters in this crate start from the uniform distribution over their
//! sample, so a booster cannot be handed the distribution `d` that
//! [`WeakLearner::produce`] receives. [`BoostedLearner`] draws a sample from
//! `d` instead and runs the booster on it.

use crate::core::{Classifier, Sample, WeakLearner};
use crate::optimization::sparsification::SplitMix64;
use serde::{Deserialize, Serialize};

/// Uses a booster together with its weak learner as a weak learner, by
/// resampling.
///
/// Given a sample `S` of `n` examples and a distribution `d` over them,
/// [`WeakLearner::produce`]
/// 1. draws `m` rows of `S` i.i.d. from `d`, with replacement (`m = n`
///    unless [`BoostedLearner::sample_size`] is set);
/// 2. calls the closure given to [`BoostedLearner::new`] with the drawn
///    sample `S'` and the inner weak learner. The closure builds a booster
///    for `S'`, runs it, and returns its combined hypothesis `f`;
/// 3. returns `f` wrapped in a [`BoostedHypothesis`], whose confidence is
///    that of `f`, or its sign `±1` with [`BoostedLearner::sign`].
///
/// If every drawn row has the same label, no booster is run (most of them
/// reject such a sample) and the constant hypothesis with that label is
/// returned.
///
/// The draw is a function of `d` and [`BoostedLearner::seed`] only, so the
/// same distribution always yields the same hypothesis, in any order of
/// calls and with any number of threads, as long as the closure is
/// deterministic.
///
/// This is *boosting by resampling* (Freund and Schapire, 1996, Section 3),
/// applied with a booster in the role of the learning algorithm. Running
/// the booster with `d` as its initial distribution would be the exact
/// alternative; this crate's boosters have no such parameter.
///
/// # Theoretical guarantee
/// None is proved here. The booster's guarantees (training error, margin)
/// hold on the drawn sample `S'`, not under `d`. For a hypothesis `h` fixed
/// before the draw with values in `[-1, 1]`, Hoeffding's inequality gives
/// `|edge_d(h) - edge_{S'}(h)| <= sqrt(2 ln(2/δ) / m)` with probability at
/// least `1 - δ`; the returned hypothesis depends on `S'`, so carrying its
/// edge over to `d` needs a uniform bound over the booster's output class
/// (e.g. through its VC dimension), which grows with the number of rounds.
///
/// # Limitations
/// - The edge under `d` is only approximated; see above. A larger
///   [`BoostedLearner::sample_size`] reduces the sampling error and costs
///   proportionally more time.
/// - Boosters that assume an edge-maximizing weak learner for their stopping
///   rule (the LPBoost family) lose that guarantee with this learner.
/// - Confidences must lie in `[-1, 1]` for the boosters of this crate.
///   The outputs of this crate's boosters satisfy this
///   ([`WeightedMajority`](crate::WeightedMajority),
///   [`MajorityVote`](crate::MajorityVote),
///   [`NaiveAggregation`](crate::NaiveAggregation), and
///   [`AdditiveModel`](crate::AdditiveModel) as a classifier); a closure
///   returning another type must ensure it, e.g. with
///   [`BoostedLearner::sign`].
/// - Labels must be `-1` or `+1`.
/// - Each call copies the `m` drawn rows and runs a whole booster, so one
///   round of the outer booster costs as much as a run of the inner one.
/// - The inner weak learner is shared by all calls. One that was built for a
///   particular sample, such as [`DecisionTree`](crate::DecisionTree), keeps
///   what it derived from that sample (the tree's bin boundaries).
///
/// # Paper
/// Yoav Freund and Robert E. Schapire, Experiments with a New Boosting
/// Algorithm, ICML 1996, describes boosting by resampling for learning
/// algorithms that take no weights. Using a booster as that algorithm is
/// this crate's construction.
///
/// # Examples
/// LPBoost over AdaBoost over decision stumps:
/// ```
/// use miniboosts::{Sample, prelude::*};
///
/// // Positive inside [10, 20), negative outside: no single stump is right
/// // on more than three quarters of the rows.
/// let rows: Vec<Vec<f64>> = (0..40).map(|i| vec![f64::from(i)]).collect();
/// let y: Vec<f64> = (0..40).map(|i| if (10..20).contains(&i) { 1.0 } else { -1.0 }).collect();
/// let sample = Sample::from_dense(&rows, Some(&y)).unwrap();
/// let stump = DecisionTreeBuilder::new(&sample).max_depth(1).build();
///
/// let learner = BoostedLearner::new(
///     |drawn, weak_learner| AdaBoost::init(drawn).force_quit_at(20).run(weak_learner),
///     stump,
/// )
/// .seed(1);
///
/// // Used directly: the hypothesis is AdaBoost's combined hypothesis.
/// let uniform = vec![1.0 / 40.0; 40];
/// let h = learner.produce(&sample, &uniform);
/// assert!(h.model().is_some_and(|f| !f.hypotheses.is_empty()));
/// assert!(h.confidence_all(&sample).iter().all(|c| c.abs() <= 1.0 + 1e-12));
///
/// // ... or as the weak learner of another booster.
/// let f = LpBoost::init(&sample).nu(1.0).run(&learner);
/// let correct = f.predict_all(&sample).iter().zip(&y).filter(|&(&p, &t)| p as f64 == t).count();
/// assert!(correct >= 36);
/// ```
#[derive(Debug, Clone)]
pub struct BoostedLearner<F, W> {
    train: F,
    weak_learner: W,
    sample_size: Option<usize>,
    seed: u64,
    sign: bool,
}

impl<F, W> BoostedLearner<F, W> {
    /// Constructs the learner from `train`, which runs a booster, and the
    /// inner `weak_learner`.
    ///
    /// `train(drawn, weak_learner)` receives the resampled sample and a
    /// reference to the inner weak learner, and returns the booster's
    /// combined hypothesis, typically
    /// `SomeBooster::init(drawn).run(weak_learner)` with the booster's
    /// parameters set in between. It is called once per
    /// [`WeakLearner::produce`] and builds a new booster each time, since a
    /// booster is tied to the sample it was constructed for.
    ///
    /// Defaults: as many drawn rows as the sample has, seed `0`, and
    /// real-valued confidences.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// See [`BoostedLearner`].
    pub fn new<O>(train: F, weak_learner: W) -> Self
    where
        W: WeakLearner,
        F: Fn(&Sample, &W) -> O,
    {
        Self {
            train,
            weak_learner,
            sample_size: None,
            seed: 0,
            sign: false,
        }
    }

    /// Sets the number `m` of rows drawn in each call. Default is the number
    /// of rows of the sample passed to [`WeakLearner::produce`].
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Panics
    /// Panics if `size == 0`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{Sample, prelude::*};
    ///
    /// let rows = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    /// let sample = Sample::from_dense(&rows, Some(&[-1.0, -1.0, 1.0, 1.0])).unwrap();
    /// let stump = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    /// // The closure sees the size of the sample it is given.
    /// let learner = BoostedLearner::new(|drawn, _| drawn.shape().0, stump).sample_size(100);
    /// assert_eq!(learner.rows(&[0.25; 4]).len(), 100);
    /// ```
    #[must_use]
    pub fn sample_size(mut self, size: usize) -> Self {
        assert!(size > 0, "at least one row must be drawn");
        self.sample_size = Some(size);
        self
    }

    /// Sets the seed of the draws. Default is `0`.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::prelude::*;
    ///
    /// let learner = |seed| BoostedLearner::new(|_, _: &NaiveBayes| (), NaiveBayes::new()).seed(seed);
    /// let dist = vec![0.02; 50];
    /// assert_eq!(learner(1).rows(&dist), learner(1).rows(&dist));
    /// assert_ne!(learner(1).rows(&dist), learner(2).rows(&dist));
    /// ```
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Makes the returned hypotheses `{-1, +1}`-valued: the confidence
    /// becomes the booster output's prediction `sign(f(x))`, with `+1` at
    /// `f(x) = 0` as in [`Classifier::predict`]. By default the confidence
    /// is `f(x)` itself, in `[-1, 1]` for voting classifiers.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{Sample, prelude::*};
    ///
    /// let rows: Vec<Vec<f64>> = (0..40).map(|i| vec![f64::from(i)]).collect();
    /// let y: Vec<f64> = (0..40).map(|i| if (10..20).contains(&i) { 1.0 } else { -1.0 }).collect();
    /// let sample = Sample::from_dense(&rows, Some(&y)).unwrap();
    /// let stump = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    ///
    /// let learner = BoostedLearner::new(
    ///     |drawn, weak_learner| AdaBoost::init(drawn).force_quit_at(5).run(weak_learner),
    ///     stump,
    /// )
    /// .sign();
    /// let h = learner.produce(&sample, &vec![1.0 / 40.0; 40]);
    /// assert!(h.confidence_all(&sample).iter().all(|&c| c == 1.0 || c == -1.0));
    /// ```
    #[must_use]
    pub fn sign(mut self) -> Self {
        self.sign = true;
        self
    }

    /// Returns the rows that [`WeakLearner::produce`] draws for the
    /// distribution `dist`, in increasing order (a row drawn `k` times is
    /// listed `k` times). The weights need not sum to one: row `i` is drawn
    /// with probability `dist[i] / Σ_j dist[j]`, so rows of weight zero are
    /// never drawn. See [`resample`].
    ///
    /// Time complexity: `O(n + m log(n m))`.
    ///
    /// # Panics
    /// Panics as [`resample`] does.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::prelude::*;
    ///
    /// let learner = BoostedLearner::new(|_, _: &NaiveBayes| (), NaiveBayes::new());
    /// assert_eq!(learner.rows(&[0.0, 0.5, 0.0, 0.5]).iter().filter(|&&i| i % 2 == 0).count(), 0);
    /// assert_eq!(learner.rows(&[0.0, 0.0, 1.0]), vec![2, 2, 2]);
    /// ```
    pub fn rows(&self, dist: &[f64]) -> Vec<usize> {
        resample(dist, self.sample_size.unwrap_or(dist.len()), self.seed)
    }
}

/// Draws `size` rows i.i.d. from the weights `dist`, with replacement, and
/// returns them in increasing order: the draw of [`BoostedLearner`].
///
/// Row `i` is drawn with probability `dist[i] / Σ_j dist[j]`, by inverting
/// the cumulative weights at uniform points of a SplitMix64 stream. The
/// stream is seeded with `seed` and every weight, so the result is a
/// function of `dist`, `size`, and `seed`.
///
/// Time complexity: `O(n + size log(n size))` for `n` weights.
///
/// # Panics
/// Panics if `dist` is empty, has a negative or non-finite entry, or sums
/// to zero or to a non-finite value.
///
/// # Examples
/// ```
/// use miniboosts::weak_learners::boosted::resample;
///
/// let rows = resample(&[0.5, 0.0, 0.5], 6, 0);
/// assert_eq!(rows.len(), 6);
/// assert!(rows.iter().all(|&i| i == 0 || i == 2));
/// assert_eq!(rows, resample(&[0.5, 0.0, 0.5], 6, 0));
/// ```
pub fn resample(dist: &[f64], size: usize, seed: u64) -> Vec<usize> {
    assert!(!dist.is_empty(), "the distribution is empty");
    assert!(
        dist.iter().all(|d| d.is_finite() && *d >= 0.0),
        "the distribution has a negative or non-finite entry"
    );
    let mut total = 0.0;
    let cumulative: Vec<f64> = dist
        .iter()
        .map(|d| {
            total += d;
            total
        })
        .collect();
    assert!(
        total > 0.0 && total.is_finite(),
        "the distribution sums to {total}"
    );
    // The last row of positive weight; it guards against a draw that rounds
    // up to `total`.
    let last = dist.iter().rposition(|&d| d > 0.0).unwrap_or(0);

    // The stream depends on the seed and on every weight, so that the draw
    // is a function of the distribution alone.
    let mut rng = SplitMix64::from_seed_and_weights(seed, dist);
    let mut rows: Vec<usize> = (0..size)
        .map(|_| {
            let point = rng.uniform() * total;
            // The first row whose cumulative weight exceeds `point`; its own
            // weight is positive.
            cumulative.partition_point(|&c| c <= point).min(last)
        })
        .collect();
    rows.sort_unstable();
    rows
}

impl<F, W, O> WeakLearner for BoostedLearner<F, W>
where
    W: WeakLearner,
    F: Fn(&Sample, &W) -> O,
{
    type Hypothesis = BoostedHypothesis<O>;

    fn name(&self) -> &str {
        "Boosted learner"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let size = self
            .sample_size
            .map_or_else(|| "sample size".to_string(), |m| m.to_string());
        Some(vec![
            ("Inner weak learner", self.weak_learner.name().to_string()),
            ("# of drawn rows", size),
            ("Seed", self.seed.to_string()),
            ("Sign output", self.sign.to_string()),
        ])
    }

    /// Draws rows from `dist`, runs the booster on them, and returns its
    /// combined hypothesis; see [`BoostedLearner`].
    ///
    /// # Panics
    /// Panics as [`BoostedLearner::rows`] does, if `dist` and `sample`
    /// differ in length, if `sample` has no target, or if the closure
    /// panics.
    fn produce(&self, sample: &Sample, dist: &[f64]) -> Self::Hypothesis {
        let n_sample = sample.shape().0;
        assert_eq!(
            dist.len(),
            n_sample,
            "the distribution has {} entries for {n_sample} examples",
            dist.len()
        );
        let target = sample.target();
        assert_eq!(target.len(), n_sample, "the sample has no target");
        let rows = self.rows(dist);

        let first = target[rows[0]];
        if rows.iter().all(|&i| target[i] == first) {
            return BoostedHypothesis {
                kind: Kind::Constant(first >= 0.0),
                sign: self.sign,
            };
        }
        let drawn = sample.select_rows(&rows);
        BoostedHypothesis {
            kind: Kind::Model((self.train)(&drawn, &self.weak_learner)),
            sign: self.sign,
        }
    }
}

/// The hypothesis returned by [`BoostedLearner`]: a booster's combined
/// hypothesis `f`, or a constant when the drawn sample had a single label.
///
/// Its confidence is `f(x)`, or `sign(f(x)) ∈ {-1, +1}` if the learner was
/// built with [`BoostedLearner::sign`]. A constant hypothesis has
/// confidence `+1` or `-1` on every example.
///
/// # Examples
/// See [`BoostedLearner`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoostedHypothesis<O> {
    kind: Kind<O>,
    sign: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum Kind<O> {
    /// The booster's output.
    Model(O),
    /// The constant `+1` (`true`) or `-1` (`false`).
    Constant(bool),
}

impl<O> BoostedHypothesis<O> {
    /// Returns the booster's combined hypothesis, or `None` for a constant
    /// hypothesis.
    ///
    /// # Examples
    /// See [`BoostedLearner`].
    pub fn model(&self) -> Option<&O> {
        match &self.kind {
            Kind::Model(model) => Some(model),
            Kind::Constant(_) => None,
        }
    }

    /// Returns the label `+1` or `-1` of a constant hypothesis, or `None`
    /// if the hypothesis is a booster's output.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{Sample, prelude::*};
    ///
    /// let rows = vec![vec![0.0], vec![1.0], vec![2.0]];
    /// let sample = Sample::from_dense(&rows, Some(&[-1.0, 1.0, 1.0])).unwrap();
    /// let learner = BoostedLearner::new(
    ///     |drawn, weak_learner| AdaBoost::init(drawn).run(weak_learner),
    ///     NaiveBayes::new(),
    /// );
    /// // All the weight on a negative example: only that row is drawn.
    /// let h = learner.produce(&sample, &[1.0, 0.0, 0.0]);
    /// assert_eq!(h.constant(), Some(-1));
    /// assert_eq!(h.predict_all(&sample), vec![-1, -1, -1]);
    /// ```
    pub fn constant(&self) -> Option<i64> {
        match self.kind {
            Kind::Model(_) => None,
            Kind::Constant(positive) => Some(if positive { 1 } else { -1 }),
        }
    }

    /// Returns `true` if the confidences are `{-1, +1}`-valued by
    /// [`BoostedLearner::sign`].
    ///
    /// # Examples
    /// See [`BoostedLearner::sign`].
    pub fn is_sign(&self) -> bool {
        self.sign
    }
}

impl<O: Classifier> Classifier for BoostedHypothesis<O> {
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        match &self.kind {
            Kind::Model(model) if self.sign => model.predict(sample, row) as f64,
            Kind::Model(model) => model.confidence(sample, row),
            Kind::Constant(positive) => constant(*positive),
        }
    }

    /// Same values as [`Classifier::confidence`] row by row, with the
    /// booster's output evaluated on all rows at once.
    fn confidence_all(&self, sample: &Sample) -> Vec<f64> {
        match &self.kind {
            Kind::Model(model) if self.sign => model
                .predict_all(sample)
                .into_iter()
                .map(|p| p as f64)
                .collect(),
            Kind::Model(model) => model.confidence_all(sample),
            Kind::Constant(positive) => vec![constant(*positive); sample.shape().0],
        }
    }
}

fn constant(positive: bool) -> f64 {
    if positive { 1.0 } else { -1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use crate::{MajorityVote, NaiveBayesClassifier};

    // Positive inside [10, 20), negative outside.
    fn interval() -> (Sample, Vec<f64>) {
        let rows: Vec<Vec<f64>> = (0..40).map(|i| vec![f64::from(i)]).collect();
        let y: Vec<f64> = (0..40)
            .map(|i| if (10..20).contains(&i) { 1.0 } else { -1.0 })
            .collect();
        (Sample::from_dense(&rows, Some(&y)).unwrap(), y)
    }

    fn counting() -> BoostedLearner<impl Fn(&Sample, &NaiveBayes) -> usize, NaiveBayes> {
        BoostedLearner::new(
            |drawn: &Sample, _: &NaiveBayes| drawn.shape().0,
            NaiveBayes::new(),
        )
    }

    #[test]
    fn rows_follow_the_distribution() {
        // 20000 draws from (0.1, 0.2, 0, 0.7): each frequency has standard
        // deviation at most sqrt(0.25 / 20000) < 0.0036, so 0.02 is more
        // than five standard deviations.
        let dist = [0.1, 0.2, 0.0, 0.7];
        let rows = counting().sample_size(20_000).seed(3).rows(&dist);
        assert_eq!(rows.len(), 20_000);
        assert!(rows.windows(2).all(|w| w[0] <= w[1]));
        for (i, &d) in dist.iter().enumerate() {
            let frequency = rows.iter().filter(|&&r| r == i).count() as f64 / 20_000.0;
            assert!((frequency - d).abs() < 0.02, "row {i}: {frequency}");
        }
        assert!(!rows.contains(&2));
        // Unnormalized weights give the same draws.
        let scaled = counting()
            .sample_size(50)
            .seed(3)
            .rows(&[1.0, 2.0, 0.0, 7.0]);
        assert!(!scaled.contains(&2));
    }

    #[test]
    fn draws_depend_on_the_seed_and_the_distribution_only() {
        let dist = vec![0.02; 50];
        let other: Vec<f64> = (0..50).map(|i| f64::from(i + 1) / 1275.0).collect();
        let learner = counting().seed(9);
        assert_eq!(learner.rows(&dist), learner.rows(&dist));
        assert_ne!(learner.rows(&dist), learner.rows(&other));
        assert_ne!(learner.rows(&dist), counting().seed(10).rows(&dist));
        assert_eq!(learner.rows(&dist).len(), 50);
    }

    #[test]
    fn invalid_distributions_panic() {
        for dist in [
            vec![],
            vec![0.0, 0.0],
            vec![0.5, -0.5],
            vec![0.5, f64::NAN],
            vec![f64::INFINITY, 1.0],
        ] {
            assert!(
                std::panic::catch_unwind(|| counting().rows(&dist)).is_err(),
                "{dist:?}"
            );
        }
        assert!(std::panic::catch_unwind(|| counting().sample_size(0)).is_err());
        let (sample, _) = interval();
        assert!(std::panic::catch_unwind(|| counting().produce(&sample, &[1.0])).is_err());
    }

    #[test]
    fn the_closure_gets_the_drawn_rows_and_the_inner_learner() {
        let (sample, _) = interval();
        let dist: Vec<f64> = (0..40).map(|i| f64::from(i + 1) / 820.0).collect();
        let learner = BoostedLearner::new(
            |drawn: &Sample, inner: &NaiveBayes| {
                let values: Vec<usize> = (0..drawn.shape().0)
                    .map(|i| drawn["feature_0"][i] as usize)
                    .collect();
                (values, drawn.target().to_vec(), inner.name().to_string())
            },
            NaiveBayes::new(),
        )
        .sample_size(60)
        .seed(2);
        let h = learner.produce(&sample, &dist);
        let (values, target, name) = h.model().unwrap();
        // The feature is the row index, so the values are the drawn rows.
        assert_eq!(values, &learner.rows(&dist));
        let expected: Vec<f64> = values.iter().map(|&i| sample.target()[i]).collect();
        assert_eq!(target, &expected);
        assert_eq!(name, NaiveBayes::new().name());
    }

    #[test]
    fn a_single_label_gives_a_constant_without_running_the_booster() {
        let (sample, _) = interval();
        let learner = BoostedLearner::new(
            |_: &Sample, _: &NaiveBayes| -> MajorityVote<NaiveBayesClassifier> {
                panic!("the booster must not run")
            },
            NaiveBayes::new(),
        );
        for (row, label) in [(0, -1), (15, 1)] {
            let mut dist = vec![0.0; 40];
            dist[row] = 1.0;
            let h = learner.produce(&sample, &dist);
            assert_eq!(h.constant(), Some(label));
            assert!(h.model().is_none());
            assert_eq!(h.confidence_all(&sample), vec![label as f64; 40]);
            assert_eq!(h.confidence(&sample, 3), label as f64);
        }
    }

    #[test]
    fn sign_turns_confidences_into_predictions() {
        let (sample, _) = interval();
        let make = || {
            BoostedLearner::new(
                |drawn, w| AdaBoost::init(drawn).force_quit_at(7).run(w),
                DecisionTreeBuilder::new(&sample).max_depth(1).build(),
            )
            .seed(4)
        };
        let uniform = vec![1.0 / 40.0; 40];
        let real = make().produce(&sample, &uniform);
        let signed = make().sign().produce(&sample, &uniform);
        assert!(!real.is_sign() && signed.is_sign());
        // The same draw and the same deterministic booster.
        assert_eq!(
            real.model().unwrap().weights,
            signed.model().unwrap().weights
        );
        let confidences = real.confidence_all(&sample);
        // ‖w‖₁ = 1 and stump confidences in [-1, 1]; the tolerance covers
        // the rounding of at most 7 weighted terms.
        assert!(confidences.iter().all(|c| c.abs() <= 1.0 + 1e-12));
        assert!(confidences.iter().any(|c| c.abs() < 1.0 - 1e-6));
        let expected: Vec<f64> = real
            .predict_all(&sample)
            .into_iter()
            .map(|p| p as f64)
            .collect();
        assert_eq!(signed.confidence_all(&sample), expected);
        for row in [0, 12, 39] {
            assert_eq!(signed.confidence(&sample, row), expected[row]);
            assert_eq!(real.confidence(&sample, row), confidences[row]);
        }
    }

    #[test]
    fn boosters_run_over_a_boosted_learner() {
        let (sample, y) = interval();
        let learner = BoostedLearner::new(
            |drawn, w| AdaBoost::init(drawn).force_quit_at(20).run(w),
            DecisionTreeBuilder::new(&sample).max_depth(1).build(),
        )
        .seed(1);
        let accuracy = |predictions: Vec<i64>| {
            predictions
                .iter()
                .zip(&y)
                .filter(|&(&p, &t)| p as f64 == t)
                .count()
        };
        // A stump alone is right on at most 30 of the 40 rows.
        let f = AdaBoost::init(&sample).force_quit_at(10).run(&learner);
        assert!(accuracy(f.predict_all(&sample)) >= 36);
        let f = LpBoost::init(&sample).nu(1.0).run(&learner);
        assert!(accuracy(f.predict_all(&sample)) >= 36);
        // Two levels of nesting, and the same result when run again.
        let nested = BoostedLearner::new(
            |drawn, w| AdaBoost::init(drawn).force_quit_at(3).run(w),
            learner,
        );
        let uniform = vec![1.0 / 40.0; 40];
        let h = nested.produce(&sample, &uniform);
        assert_eq!(
            h.confidence_all(&sample),
            nested.produce(&sample, &uniform).confidence_all(&sample)
        );
        assert!(accuracy(h.predict_all(&sample)) >= 36);
    }
}
