use super::{shuffled, vote};
use crate::MajorityVote;
use crate::core::{Classifier, Sample};

/// The optimal weak-to-strong learner of Larsen and Ritzert (2022): run a
/// booster on Hanneke's highly overlapping sub-samples and take the
/// unweighted majority vote of the resulting voting classifiers.
///
/// # Algorithm
/// Algorithm 1 of the paper, `Sub-Sample(A, B)` (Hanneke, 2016): if
/// `|A| <= 3`, return `{A ∪ B}`. Otherwise split `A` in order into `A_0`
/// (the first `|A| - 3⌊|A|/4⌋` elements) and `A_1, A_2, A_3` (the next
/// `⌊|A|/4⌋` elements each), and return the union of
/// ```txt
/// Sub-Sample(A_0, A_2 ∪ A_3 ∪ B),
/// Sub-Sample(A_0, A_1 ∪ A_3 ∪ B),
/// Sub-Sample(A_0, A_1 ∪ A_2 ∪ B).
/// ```
/// Algorithm 2: compute `C_1, ..., C_k = Sub-Sample(S, ∅)`, run a booster
/// on each `C_i`, and return the unweighted majority vote of the results.
/// There are `k = 3^L` sub-samples, where `L` is the number of levels until
/// `|A| <= 3`; `k ≈ m^{log_4 3} ≈ m^{0.79}`.
///
/// Here the booster is the closure passed to [`LarsenRitzert::run`]. The
/// sub-samples depend on the order of `S`; this implementation first
/// shuffles the rows with [`LarsenRitzert::seed`], which changes nothing
/// for i.i.d. samples but avoids sub-samples that follow a sorted file.
/// [`LarsenRitzert::max_subsamples`] uses only a random subset of the
/// sub-samples, as the paper's follow-up did in its experiments on large
/// datasets; this is an approximation without the guarantee below.
///
/// # Theoretical guarantee
/// Theorems 1 and 5 of the paper: for a `γ`-weak learner (`0 < γ < 1/2`)
/// whose hypothesis class has VC dimension `d`, if the booster returns on
/// every sub-sample a voting classifier with margins at least `γ/2` on all of
/// its examples (as AdaBoost*_ν of Rätsch and Warmuth does), then with
/// probability at least `1 - δ` over `m` i.i.d. examples the error is
/// `O((d / γ² + ln(1/δ)) / m)`. This sample complexity is optimal by the
/// paper's matching lower bound and improves AdaBoost's by two logarithmic
/// factors.
///
/// # Limitations
/// - The guarantee needs margins `γ/2` on every sub-sample, so `γ` must be
///   known to configure the booster; a fixed round count or another booster
///   carries no such guarantee.
/// - It runs the booster `k ≈ m^{0.79}` times on sub-samples of size about
///   `3m/4` or more (e.g. `k = 243` for `m = 1024`), so it is far slower
///   than one run of the booster.
/// - The closure must handle any sub-sample it receives; many boosters panic
///   on a sub-sample with a single label.
/// - Not a [`Booster`](crate::Booster): there are no rounds, logging, or
///   [`CurrentHypothesis`](crate::core::CurrentHypothesis).
///
/// # Paper
/// Kasper Green Larsen and Martin Ritzert, [Optimal Weak to Strong
/// Learning](https://arxiv.org/abs/2206.01563), NeurIPS 2022. The
/// sub-sampling is from Steve Hanneke, [The Optimal Sample Complexity of PAC
/// Learning](https://jmlr.org/papers/v17/15-389.html), JMLR 17(38), 2016.
///
/// # Examples
/// ```
/// use miniboosts::{Classifier, Sample, prelude::*};
///
/// // A gap between the classes, so that every sub-sample's stump separates them.
/// let rows: Vec<Vec<f64>> = (0..16).map(|i| vec![if i < 8 { i } else { i + 10 } as f64]).collect();
/// let y: Vec<f64> = (0..16).map(|i| if i < 8 { -1.0 } else { 1.0 }).collect();
/// let sample = Sample::from_dense(&rows, Some(&y)).unwrap();
///
/// let learner = LarsenRitzert::init(&sample);
/// // 16 → 4 → 1 rows in A: two levels, 3² sub-samples.
/// assert_eq!(learner.n_subsamples(), 9);
/// let f = learner.run(|sub: &Sample| {
///     let stump = DecisionTreeBuilder::new(sub).max_depth(1).build();
///     AdaBoost::init(sub).force_quit_at(10).run(&stump)
/// });
/// assert_eq!(f.hypotheses.len(), 9);
/// assert_eq!(f.predict_all(&sample), y.iter().map(|&v| v as i64).collect::<Vec<_>>());
/// ```
pub struct LarsenRitzert<'a> {
    sample: &'a Sample,
    seed: u64,
    max_subsamples: Option<usize>,
}

impl<'a> LarsenRitzert<'a> {
    /// Constructs the learner for `sample`, using every sub-sample, with seed
    /// `0`.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// See [`LarsenRitzert`].
    pub fn init(sample: &'a Sample) -> Self {
        Self {
            sample,
            seed: 0,
            max_subsamples: None,
        }
    }

    /// Sets the seed of the shuffle of the rows (and of the choice of
    /// sub-samples with [`LarsenRitzert::max_subsamples`]). Default is `0`.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{LarsenRitzert, Sample};
    ///
    /// let sample = Sample::dummy(16);
    /// let a = LarsenRitzert::init(&sample).seed(1).subsamples();
    /// assert_eq!(a, LarsenRitzert::init(&sample).seed(1).subsamples());
    /// assert_ne!(a, LarsenRitzert::init(&sample).seed(2).subsamples());
    /// ```
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Uses at most `count` of the sub-samples, chosen uniformly at random
    /// without replacement. This is an approximation outside the paper's
    /// guarantee; see [`LarsenRitzert`].
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Panics
    /// Panics if `count == 0`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{LarsenRitzert, Sample};
    ///
    /// let sample = Sample::dummy(64); // 27 sub-samples
    /// let learner = LarsenRitzert::init(&sample).max_subsamples(5);
    /// assert_eq!(learner.subsamples().len(), 5);
    /// ```
    pub fn max_subsamples(mut self, count: usize) -> Self {
        assert!(count > 0, "at least one sub-sample is needed");
        self.max_subsamples = Some(count);
        self
    }

    /// Returns the number `3^L` of sub-samples that `Sub-Sample(S, ∅)`
    /// produces for this sample, before [`LarsenRitzert::max_subsamples`].
    ///
    /// Time complexity: `O(log m)`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{LarsenRitzert, Sample};
    ///
    /// // m = 1024: |A| goes 1024 → 256 → 64 → 16 → 4 → 1, five levels.
    /// assert_eq!(LarsenRitzert::init(&Sample::dummy(1024)).n_subsamples(), 243);
    /// assert_eq!(LarsenRitzert::init(&Sample::dummy(3)).n_subsamples(), 1);
    /// ```
    pub fn n_subsamples(&self) -> usize {
        3usize.pow(levels(self.sample.shape().0).len() as u32)
    }

    /// Returns the row indices of the sub-samples used by
    /// [`LarsenRitzert::run`], in the order of the paper's recursion (or a
    /// random subset of them with [`LarsenRitzert::max_subsamples`]). Rows
    /// within a sub-sample are listed in shuffled order.
    ///
    /// Time complexity: `O(k m)` for `k` sub-samples.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{LarsenRitzert, Sample};
    ///
    /// let sample = Sample::dummy(16);
    /// let subsamples = LarsenRitzert::init(&sample).subsamples();
    /// // |A_0| = 1 at the leaves, plus two of the three quarters at both levels.
    /// assert!(subsamples.iter().all(|c| c.len() == 1 + 2 + 8));
    /// ```
    pub fn subsamples(&self) -> Vec<Vec<usize>> {
        let m = self.sample.shape().0;
        let order = shuffled(m, self.seed);
        let levels = levels(m);
        let total = self.n_subsamples();
        let chosen: Vec<usize> = match self.max_subsamples {
            Some(count) if count < total => {
                // A stream independent of the row shuffle.
                let mut picks = shuffled(total, self.seed ^ 0x5DEE_CE66_D1CE_4E5B);
                picks.truncate(count);
                picks.sort_unstable();
                picks
            }
            _ => (0..total).collect(),
        };
        chosen
            .into_iter()
            .map(|k| {
                subsample_positions(m, &levels, k)
                    .into_iter()
                    .map(|p| order[p])
                    .collect()
            })
            .collect()
    }

    /// Runs `train` on every sub-sample, in parallel, and returns the
    /// unweighted majority vote of the results. `train` receives the
    /// sub-sample as its own [`Sample`] and should return a voting
    /// classifier with margins at least `γ/2` on it (see [`LarsenRitzert`]).
    ///
    /// Time complexity: the total time of the calls to `train`, plus
    /// `O(k m f)` to copy `k` sub-samples with `f` features.
    ///
    /// # Panics
    /// Panics if `train` panics.
    ///
    /// # Examples
    /// See [`LarsenRitzert`].
    pub fn run<H, F>(&self, train: F) -> MajorityVote<H>
    where
        H: Classifier + Send,
        F: Fn(&Sample) -> H + Sync,
    {
        vote(self.sample, self.subsamples(), train)
    }
}

/// The recursion levels of `Sub-Sample(S, ∅)` for `|S| = m`: at each level,
/// `(|A|, ⌊|A|/4⌋)`, while `|A| > 3`. `A` is always a prefix `0..|A|` of
/// the (shuffled) rows.
fn levels(m: usize) -> Vec<(usize, usize)> {
    let mut levels = Vec::new();
    let mut len = m;
    while len > 3 {
        let quarter = len / 4;
        levels.push((len, quarter));
        len -= 3 * quarter;
    }
    levels
}

/// Positions of the `k`-th sub-sample: `k` in base 3, most significant digit
/// first, says which of `A_1, A_2, A_3` each level leaves out (digit `0`
/// leaves out `A_1`, the first recursive call). Positions are sorted.
fn subsample_positions(m: usize, levels: &[(usize, usize)], mut k: usize) -> Vec<usize> {
    let mut omitted = vec![0; levels.len()];
    for slot in omitted.iter_mut().rev() {
        *slot = k % 3;
        k /= 3;
    }
    // The final `A`, of size at most 3.
    let leaf = levels.last().map_or(m, |&(len, quarter)| len - 3 * quarter);
    let mut positions: Vec<usize> = (0..leaf).collect();
    for (&(len, quarter), &skip) in levels.iter().zip(&omitted) {
        let start = len - 3 * quarter;
        for part in 0..3 {
            if part != skip {
                let from = start + part * quarter;
                positions.extend(from..from + quarter);
            }
        }
    }
    positions.sort_unstable();
    positions
}

#[cfg(test)]
mod tests {
    use super::*;

    // Algorithm 1 of the paper, literally, on positions 0..m.
    fn sub_sample(a: &[usize], b: &[usize]) -> Vec<Vec<usize>> {
        if a.len() <= 3 {
            let mut c = [a, b].concat();
            c.sort_unstable();
            return vec![c];
        }
        let q = a.len() / 4;
        let n0 = a.len() - 3 * q;
        let (a0, a1, a2, a3) = (
            &a[..n0],
            &a[n0..n0 + q],
            &a[n0 + q..n0 + 2 * q],
            &a[n0 + 2 * q..],
        );
        let mut out = sub_sample(a0, &[a2, a3, b].concat());
        out.extend(sub_sample(a0, &[a1, a3, b].concat()));
        out.extend(sub_sample(a0, &[a1, a2, b].concat()));
        out
    }

    #[test]
    fn decoding_matches_the_recursion() {
        for m in [1, 3, 4, 5, 7, 15, 16, 17, 40, 64, 100, 257] {
            let positions: Vec<usize> = (0..m).collect();
            let expected = sub_sample(&positions, &[]);
            let levels = levels(m);
            assert_eq!(expected.len(), 3usize.pow(levels.len() as u32), "m = {m}");
            for (k, c) in expected.iter().enumerate() {
                assert_eq!(&subsample_positions(m, &levels, k), c, "m = {m}, k = {k}");
            }
        }
    }

    #[test]
    fn subsamples_map_positions_through_the_shuffle() {
        let sample = Sample::dummy(40);
        let learner = LarsenRitzert::init(&sample).seed(5);
        let order = shuffled(40, 5);
        let positions = sub_sample(&(0..40).collect::<Vec<_>>(), &[]);
        let subsamples = learner.subsamples();
        assert_eq!(subsamples.len(), positions.len());
        for (c, p) in subsamples.iter().zip(&positions) {
            assert_eq!(c, &p.iter().map(|&i| order[i]).collect::<Vec<_>>());
        }
    }

    #[test]
    fn max_subsamples_picks_distinct_subsamples() {
        let sample = Sample::dummy(64);
        let all = LarsenRitzert::init(&sample).seed(2).subsamples();
        let some = LarsenRitzert::init(&sample)
            .seed(2)
            .max_subsamples(10)
            .subsamples();
        assert_eq!(some.len(), 10);
        assert!(some.iter().all(|c| all.contains(c)));
        let mut unique = some.clone();
        unique.dedup();
        assert_eq!(unique.len(), 10);
        // More than available keeps all of them.
        assert_eq!(
            LarsenRitzert::init(&sample)
                .seed(2)
                .max_subsamples(100)
                .subsamples(),
            all
        );
        assert!(
            std::panic::catch_unwind(|| LarsenRitzert::init(&sample).max_subsamples(0)).is_err()
        );
    }
}
