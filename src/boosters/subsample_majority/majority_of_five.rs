use super::{shuffled, vote};
use crate::MajorityVote;
use crate::core::{Classifier, Sample};

/// Majority-of-5 (Høgsgaard, Larsen, and Mathiasen, 2024): split the sample
/// into five disjoint parts, run a booster on each, and take the unweighted
/// majority vote of the five voting classifiers.
///
/// # Algorithm
/// Algorithm 1 of the paper:
/// 1. partition the training sample `S` into 5 disjoint parts
///    `S_1, ..., S_5` of size `m / 5`;
/// 2. run AdaBoost with the weak learner on each part, giving voting
///    classifiers `f_1, ..., f_5`;
/// 3. return `g = sign(f_1 + ... + f_5)`, where each `f_t` contributes its
///    prediction `±1`.
///
/// Here the booster of step 2 is the closure passed to
/// [`MajorityOfFive::run`]. The paper leaves the partition open; this
/// implementation shuffles the rows with [`MajorityOfFive::seed`] and cuts
/// the shuffled order into contiguous parts whose sizes differ by at most
/// one. [`MajorityOfFive::n_parts`] changes the number of parts, as in the
/// paper's Majority-of-X experiments.
///
/// # Theoretical guarantee
/// Theorem 1 of the paper: for a `γ`-weak learner whose hypothesis class has
/// VC dimension `d`, if each run of AdaBoost reaches margins `Ω(γ)` on its
/// part (e.g. after `O(ln(m) / γ²)` rounds), the expected error of `g` over
/// the draw of `m` i.i.d. examples is `O(d / (γ² m))`. This is the optimal
/// sample complexity of weak-to-strong learning, with `O(ln(m) / γ²)` weak
/// learner calls in total, as for AdaBoost. The bound holds in expectation
/// and for five parts; it does not cover other part counts.
///
/// # Limitations
/// - The guarantee needs the closure to reach the margins above; a fixed
///   round count or another booster carries no such guarantee.
/// - Each part has about `m / 5` examples, so on small samples each voter
///   sees little data; the paper's experiments found Majority-of-X weak on
///   small datasets.
/// - The closure must handle any part it receives. In particular, many
///   boosters panic on a part with a single label, which can happen on
///   small or very unbalanced samples.
/// - Not a [`Booster`](crate::Booster): there are no rounds, logging, or
///   [`CurrentHypothesis`](crate::core::CurrentHypothesis).
///
/// # Paper
/// Mikael Møller Høgsgaard, Kasper Green Larsen, and Markus Engelund
/// Mathiasen, [The Many Faces of Optimal Weak-to-Strong
/// Learning](https://arxiv.org/abs/2408.17148), NeurIPS 2024.
///
/// # Examples
/// ```
/// use miniboosts::{Classifier, Sample, prelude::*};
///
/// let rows: Vec<Vec<f64>> = (0..40).map(|i| vec![i as f64]).collect();
/// let y: Vec<f64> = (0..40).map(|i| if i < 20 { -1.0 } else { 1.0 }).collect();
/// let sample = Sample::from_dense(&rows, Some(&y)).unwrap();
///
/// let f = MajorityOfFive::init(&sample).seed(1).run(|part: &Sample| {
///     let stump = DecisionTreeBuilder::new(part).max_depth(1).build();
///     AdaBoost::init(part).force_quit_at(10).run(&stump)
/// });
/// assert_eq!(f.hypotheses.len(), 5);
/// assert_eq!(f.predict_all(&sample), y.iter().map(|&v| v as i64).collect::<Vec<_>>());
/// ```
pub struct MajorityOfFive<'a> {
    sample: &'a Sample,
    n_parts: usize,
    seed: u64,
}

impl<'a> MajorityOfFive<'a> {
    /// Constructs Majority-of-5 for `sample`, with 5 parts and seed `0`.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// See [`MajorityOfFive`].
    pub fn init(sample: &'a Sample) -> Self {
        Self {
            sample,
            n_parts: 5,
            seed: 0,
        }
    }

    /// Sets the number of parts. Default is `5`, the value the paper's
    /// guarantee covers.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Panics
    /// Panics if `n_parts == 0`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{MajorityOfFive, Sample};
    ///
    /// let sample = Sample::dummy(9);
    /// let parts = MajorityOfFive::init(&sample).n_parts(3).parts();
    /// assert_eq!(parts.iter().map(Vec::len).collect::<Vec<_>>(), vec![3, 3, 3]);
    /// ```
    pub fn n_parts(mut self, n_parts: usize) -> Self {
        assert!(n_parts > 0, "Majority-of-X needs at least one part");
        self.n_parts = n_parts;
        self
    }

    /// Sets the seed of the shuffle that assigns rows to parts. Default is
    /// `0`.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{MajorityOfFive, Sample};
    ///
    /// let sample = Sample::dummy(10);
    /// let a = MajorityOfFive::init(&sample).seed(1).parts();
    /// assert_eq!(a, MajorityOfFive::init(&sample).seed(1).parts());
    /// assert_ne!(a, MajorityOfFive::init(&sample).seed(2).parts());
    /// ```
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Returns the row indices of each part: a partition of the rows into
    /// [`MajorityOfFive::n_parts`] parts whose sizes differ by at most one
    /// (the first `m mod n_parts` parts have one more row).
    ///
    /// Time complexity: `O(m)`.
    ///
    /// # Panics
    /// Panics if the sample has fewer rows than parts.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{MajorityOfFive, Sample};
    ///
    /// let sample = Sample::dummy(12);
    /// let parts = MajorityOfFive::init(&sample).parts();
    /// assert_eq!(parts.iter().map(Vec::len).collect::<Vec<_>>(), vec![3, 3, 2, 2, 2]);
    /// ```
    pub fn parts(&self) -> Vec<Vec<usize>> {
        let m = self.sample.shape().0;
        assert!(
            m >= self.n_parts,
            "{m} rows cannot be split into {} nonempty parts",
            self.n_parts
        );
        let order = shuffled(m, self.seed);
        let (size, extra) = (m / self.n_parts, m % self.n_parts);
        let mut start = 0;
        (0..self.n_parts)
            .map(|t| {
                let end = start + size + usize::from(t < extra);
                let part = order[start..end].to_vec();
                start = end;
                part
            })
            .collect()
    }

    /// Runs `train` on every part, in parallel, and returns the unweighted
    /// majority vote of the results. `train` receives the part as its own
    /// [`Sample`] and should return a voting classifier, e.g. by running a
    /// booster with a weak learner built for that sample.
    ///
    /// Time complexity: the total time of the calls to `train`, plus `O(m k)`
    /// to copy the parts for `k` features.
    ///
    /// # Panics
    /// Panics as [`MajorityOfFive::parts`] does, or if `train` panics.
    ///
    /// # Examples
    /// See [`MajorityOfFive`].
    pub fn run<H, F>(&self, train: F) -> MajorityVote<H>
    where
        H: Classifier + Send,
        F: Fn(&Sample) -> H + Sync,
    {
        vote(self.sample, self.parts(), train)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_partition_the_rows() {
        let sample = Sample::dummy(23);
        for n_parts in [1, 2, 5, 23] {
            let parts = MajorityOfFive::init(&sample)
                .n_parts(n_parts)
                .seed(3)
                .parts();
            assert_eq!(parts.len(), n_parts);
            let mut rows: Vec<usize> = parts.concat();
            rows.sort_unstable();
            assert_eq!(rows, (0..23).collect::<Vec<_>>());
            let sizes: Vec<usize> = parts.iter().map(Vec::len).collect();
            assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1);
        }
    }

    #[test]
    fn run_trains_once_per_part_on_its_rows() {
        // The single feature is the row index, so each voter can report the
        // rows it was trained on.
        let rows: Vec<Vec<f64>> = (0..12).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..12)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let sample = Sample::from_dense(&rows, Some(&y)).unwrap();
        #[derive(Clone)]
        struct Seen(Vec<usize>);
        impl Classifier for Seen {
            fn confidence(&self, _: &Sample, _: usize) -> f64 {
                1.0
            }
        }
        let learner = MajorityOfFive::init(&sample).seed(4);
        let f = learner.run(|part| {
            Seen(
                (0..part.shape().0)
                    .map(|i| part["feature_0"][i] as usize)
                    .collect(),
            )
        });
        let expected = learner.parts();
        assert_eq!(
            f.hypotheses.iter().map(|h| h.0.clone()).collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn invalid_settings_panic() {
        let sample = Sample::dummy(3);
        assert!(std::panic::catch_unwind(|| MajorityOfFive::init(&sample).n_parts(0)).is_err());
        assert!(std::panic::catch_unwind(|| MajorityOfFive::init(&sample).parts()).is_err());
    }
}
