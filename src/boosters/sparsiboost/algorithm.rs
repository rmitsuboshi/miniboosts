//! Implementation of SparsiBoost (Grønlund, Larsen, and Mathiasen, 2019).
//! See the [`SparsiBoost`] struct for details and an example.
use crate::boosters::adaboostv::AdaBoostV;
use crate::core::CurrentHypothesis;
use crate::core::{Booster, Classifier, Sample, WeakLearner};
use crate::hypotheses::WeightedMajority;
use crate::optimization::sparsification::sparsify_ensemble;

use std::ops::ControlFlow;

/// SparsiBoost by Grønlund, Larsen, and Mathiasen: AdaBoostV followed by a
/// margin-preserving sparsification to at most `T` hypotheses.
///
/// # Algorithm
/// Let `m` be the number of training examples and `T` the target number of
/// hypotheses. Algorithm 1 of the paper:
/// 1. Run [`AdaBoostV`] for `c T` rounds, where
///    `c = ⌈lg m / lg(2 + m/T)⌉`, to obtain hypotheses `h_1, ..., h_{cT}`
///    with weights `w`.
/// 2. Form the margin matrix `u_ij = y_i h_j(x_i)` and normalize
///    `‖w‖_1 = 1`.
/// 3. Find `w'` with `‖w'‖_0 ≤ T`, `‖w'‖_1 = 1`, and small
///    `‖U w - U w'‖_∞` by Algorithm 2 (repeated halving with discrepancy
///    minimization), implemented by
///    [`sparsify`](crate::optimization::sparsification::sparsify).
/// 4. Return the hypotheses with nonzero `w'_j` and weights `w'_j`.
///
/// AdaBoostV runs with margin accuracy `ν = √(2 ln m / (c T))`, the value
/// for which its iteration bound `2 ln m / ν²` equals `c T`; this is the
/// choice behind the `O(√(lg m / (c T)))` gap used in the proof of
/// Theorem 2.1.
///
/// Implementation choices (not part of the original method):
/// - `c` is at least `1`, and the default `ν` is capped at `1/2`. AdaBoostV
///   needs `ν < 1 + γ^min` (its smallest edge), so without the cap a small
///   `T` would make it panic whenever an edge is close to `0`. The cap is
///   active only when `√(2 ln m / (c T)) > 1/2`, where the bound of
///   Theorem 2.1 is a constant anyway. [`SparsiBoost::tolerance`] overrides
///   `ν`.
/// - AdaBoostV stops early if a hypothesis has edge `±1`; the ensemble then
///   has one hypothesis and needs no sparsification.
/// - The column subsets come from a heuristic discrepancy minimizer; see
///   [`discrepancy_coloring`](crate::optimization::sparsification::discrepancy_coloring)
///   and *Limitations*.
///
/// # Theoretical guarantee
/// Theorem 2.1: SparsiBoost finds a combination of at most `T` hypotheses
/// whose minimal margin is at least `ρ* - O(√(lg(2 + m/T) / T))`, where `ρ*`
/// is the best minimal margin of any convex combination of hypotheses in
/// the class. Theorem 3.1 shows this is optimal: for some data sets and
/// hypothesis classes no combination of `T` hypotheses does better. The
/// guarantee combines AdaBoostV's (which needs a weak learner returning
/// edges `γ_t ≥ ρ*`) with Theorem 2.2, whose constructive proof needs a
/// coloring that meets Spencer's bound in every halving step.
///
/// # Limitations
/// - Neither assumption is checked. In particular, the discrepancy step is
///   a heuristic that does not certify Spencer's bound, so the margin
///   guarantee is not certified. [`SparsiBoost::sparsification_error`]
///   reports the measured `‖U w - U w'‖_∞` on the training sample.
/// - The result depends on [`SparsiBoost::seed`]; equal seeds give equal
///   results.
/// - The output may have fewer than `T` hypotheses, because each halving
///   step can overshoot. Equal hypotheses returned in different rounds are
///   kept as separate columns.
/// - Hypotheses must return confidences in `[-1, 1]` (as AdaBoostV
///   requires); labels must be `-1` or `+1` and both classes must be
///   present.
/// - Postprocessing evaluates every hypothesis on every example
///   (`O(m c T)` confidences and memory) and the sparsification takes
///   `O(m (cT)² + (cT)³)` time. [`CurrentHypothesis::current_hypothesis`]
///   repeats this work on every call, so logging every round with
///   [`crate::Logger`] is expensive.
///
/// # Parameters
/// | Builder method | Default | Meaning |
/// | --- | --- | --- |
/// | [`init`](SparsiBoost::init) | required | Target number of hypotheses `T`. |
/// | [`tolerance`](SparsiBoost::tolerance) | `min(√(2 ln m / (cT)), 1/2)` | AdaBoostV's margin accuracy `ν`; does not change the `c T` rounds. |
/// | [`seed`](SparsiBoost::seed) | `0` | Seed of the randomized discrepancy minimization. |
///
/// # Reference
/// Allan Grønlund, Kasper Green Larsen, and Alexander Mathiasen.
/// *Optimal Minimal Margin Maximization with Boosting.*
/// Proceedings of the 36th International Conference on Machine Learning
/// (ICML), PMLR 97:4392–4401, 2019. <https://arxiv.org/abs/1901.10789>
///
/// # Examples
/// Train on a held-out split and keep at most 3 hypotheses.
/// ```
/// use std::io::BufReader;
/// use miniboosts::{Sample, prelude::*};
///
/// // Labels follow the sign of `x + y`, so no single stump separates them.
/// let csv = b"x,y,class\n\
///     -2,1,-1\n-1,-2,-1\n1,-3,-1\n-3,2,-1\n0,-1,-1\n-2,-1,-1\n\
///     2,1,1\n1,2,1\n-1,3,1\n3,-1,1\n0,2,1\n2,2,1\n";
/// let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
///     .unwrap()
///     .set_target("class");
///
/// // A fixed permutation keeps the split reproducible:
/// // rows ix[0..4] form the test set, the other 8 rows the training set.
/// let ix = [0, 6, 4, 10, 1, 2, 3, 5, 7, 8, 9, 11];
/// let (train, test) = sample.split(&ix, 0, 4);
/// let learner = DecisionTreeBuilder::new(&train).max_depth(1).build();
///
/// // m = 8, T = 3: c = ⌈lg 8 / lg(2 + 8/3)⌉ = ⌈3 / 2.22⌉ = 2 AdaBoostV
/// // rounds per output hypothesis.
/// let mut booster = SparsiBoost::init(&train, 3).seed(42);
/// assert_eq!(booster.max_loop(), 6);
/// let f = booster.run(&learner);
///
/// assert!(!f.hypotheses.is_empty() && f.hypotheses.len() <= 3);
/// assert!((f.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
/// // How far the training margins moved during sparsification.
/// let error = booster.sparsification_error().unwrap();
/// assert!((0.0..=2.0).contains(&error));
///
/// let predictions = f.predict_all(&test);
/// assert!(predictions.iter().all(|&p| p == -1 || p == 1));
/// ```
pub struct SparsiBoost<'a, H> {
    // Training sample
    sample: &'a Sample,

    // Target number of hypotheses `T`.
    n_hypotheses: usize,

    // User override of AdaBoostV's margin accuracy `ν`.
    tolerance: Option<f64>,

    // Seed of the discrepancy minimization.
    seed: u64,

    // The first stage; rebuilt by `preprocess`.
    adaboostv: AdaBoostV<'a, H>,

    // `‖U w - U w'‖_∞` of the last `postprocess`.
    sparsification_error: Option<f64>,
}

impl<'a, H> SparsiBoost<'a, H> {
    /// Constructs SparsiBoost for `sample` with target number of hypotheses
    /// `n_hypotheses` (the `T` of the paper). The seed defaults to `0`, and
    /// AdaBoostV's `ν` to `min(√(2 ln m / (cT)), 1/2)`. The sample is
    /// validated when training starts.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Panics
    /// Panics if `n_hypotheses == 0`.
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{Sample, prelude::*};
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    /// let f = SparsiBoost::init(&sample, 2).run(&learner);
    /// // A separating stump has edge 1, so AdaBoostV stops in round 1.
    /// assert_eq!(f.hypotheses.len(), 1);
    /// assert_eq!(f.predict_all(&sample), vec![-1, -1, -1, 1, 1, 1]);
    /// ```
    pub fn init(sample: &'a Sample, n_hypotheses: usize) -> Self {
        assert!(n_hypotheses > 0, "SparsiBoost needs a positive target");
        Self {
            sample,
            n_hypotheses,
            tolerance: None,
            seed: 0,
            adaboostv: AdaBoostV::init(sample),
            sparsification_error: None,
        }
    }

    /// Sets AdaBoostV's margin accuracy `ν`, replacing the default
    /// `min(√(2 ln m / (cT)), 1/2)`. The number of AdaBoostV rounds stays
    /// [`SparsiBoost::max_loop`]; a `ν` smaller than the default means
    /// AdaBoostV's guarantee for `ν` needs more rounds than are run.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Panics
    /// Panics if `tolerance` is not finite and strictly positive.
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{Sample, prelude::*};
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// use miniboosts::DecisionTreeClassifier;
    ///
    /// let booster = SparsiBoost::<DecisionTreeClassifier>::init(&sample, 2);
    /// // m = 6, T = 2: c = 2, so ν = √(2 ln 6 / 4) ≈ 0.95 is capped at 1/2.
    /// assert_eq!(booster.adaboostv_tolerance(), 0.5);
    /// let booster = booster.tolerance(0.25);
    /// assert_eq!(booster.adaboostv_tolerance(), 0.25);
    /// ```
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        assert!(
            tolerance.is_finite() && tolerance > 0.0,
            "SparsiBoost tolerance must be finite and positive"
        );
        self.tolerance = Some(tolerance);
        self
    }

    /// Sets the seed of the randomized discrepancy minimization. Equal seeds
    /// give equal results for the same sample and weak learner.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{Sample, prelude::*};
    /// # let csv = b"x,y,class\n\
    /// #     -2,1,-1\n-1,-2,-1\n1,-3,-1\n-3,2,-1\n0,-1,-1\n-2,-1,-1\n\
    /// #     2,1,1\n1,2,1\n-1,3,1\n3,-1,1\n0,2,1\n2,2,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    /// let f = SparsiBoost::init(&sample, 2).seed(7).run(&learner);
    /// let g = SparsiBoost::init(&sample, 2).seed(7).run(&learner);
    /// assert_eq!(f.weights, g.weights);
    /// ```
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Returns the number of AdaBoostV rounds, `c T` with
    /// `c = max(1, ⌈lg m / lg(2 + m/T)⌉)` for `m` training examples.
    /// Values exceeding `usize::MAX` saturate.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{Sample, prelude::*};
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// use miniboosts::DecisionTreeClassifier;
    ///
    /// // m = 6, T = 2: c = ⌈lg 6 / lg 5⌉ = ⌈1.11⌉ = 2.
    /// let booster = SparsiBoost::<DecisionTreeClassifier>::init(&sample, 2);
    /// assert_eq!(booster.max_loop(), 4);
    /// ```
    pub fn max_loop(&self) -> usize {
        let m = self.sample.shape().0 as f64;
        let t = self.n_hypotheses as f64;
        let c = (m.log2() / (2.0 + m / t).log2()).ceil().max(1.0) as usize;
        c.saturating_mul(self.n_hypotheses)
    }

    /// Returns the margin accuracy `ν` passed to AdaBoostV: the value set by
    /// [`SparsiBoost::tolerance`], or `min(√(2 ln m / (cT)), 1/2)` with
    /// `cT =` [`SparsiBoost::max_loop`].
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// See [`SparsiBoost::tolerance`].
    pub fn adaboostv_tolerance(&self) -> f64 {
        self.tolerance.unwrap_or_else(|| {
            let m = self.sample.shape().0 as f64;
            (2.0 * m.ln() / self.max_loop() as f64).sqrt().min(0.5)
        })
    }

    /// Returns `‖U w - U w'‖_∞` from the last call to
    /// [`Booster::postprocess`] (and thus [`Booster::run`]): the largest
    /// change of a training margin caused by the sparsification, with `w`
    /// AdaBoostV's normalized weights and `w'` the output weights. Returns
    /// `None` before the first run.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// See [`SparsiBoost`].
    pub fn sparsification_error(&self) -> Option<f64> {
        self.sparsification_error
    }
}

impl<H> SparsiBoost<'_, H>
where
    H: Classifier + Clone,
{
    fn sparsify(&self, f: &WeightedMajority<H>) -> (WeightedMajority<H>, f64) {
        sparsify_ensemble(self.sample, f, self.n_hypotheses, self.seed)
    }
}

impl<H> Booster<H> for SparsiBoost<'_, H>
where
    H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;

    fn name(&self) -> &str {
        "SparsiBoost"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_examples, n_feature) = self.sample.shape();
        let info = Vec::from([
            ("# of examples", format!("{}", n_examples)),
            ("# of features", format!("{}", n_feature)),
            ("Target # of hypotheses", format!("{}", self.n_hypotheses)),
            ("AdaBoostV rounds", format!("{}", self.max_loop())),
            (
                "AdaBoostV tolerance",
                format!("{}", self.adaboostv_tolerance()),
            ),
            ("Seed", format!("{}", self.seed)),
        ]);
        Some(info)
    }

    fn preprocess(&mut self) {
        self.sample.is_valid_binary_instance();
        self.sparsification_error = None;
        self.adaboostv = AdaBoostV::init(self.sample)
            .tolerance(self.adaboostv_tolerance())
            .force_quit_at(self.max_loop());
        self.adaboostv.preprocess();
    }

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize) -> ControlFlow<usize>
    where
        W: WeakLearner<Hypothesis = H>,
    {
        self.adaboostv.boost(weak_learner, iteration)
    }

    fn postprocess(&mut self) -> Self::Output {
        let f = self.adaboostv.postprocess();
        let (f, error) = self.sparsify(&f);
        assert!(error.is_finite(), "SparsiBoost margin change is not finite");
        self.sparsification_error = Some(error);
        f
    }
}

impl<H> CurrentHypothesis for SparsiBoost<'_, H>
where
    H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;

    /// Sparsifies AdaBoostV's current ensemble, as `postprocess` would.
    fn current_hypothesis(&self) -> Self::Output {
        self.sparsify(&self.adaboostv.current_hypothesis()).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DecisionTreeBuilder;
    use crate::optimization::sparsification::{margin_columns, max_margin_change};
    use std::io::BufReader;

    fn xor_like_sample() -> Sample {
        // Labels follow the sign of x + y; no single stump separates them.
        let csv = b"x,y,class\n\
            -2,1,-1\n-1,-2,-1\n1,-3,-1\n-3,2,-1\n0,-1,-1\n-2,-1,-1\n\
            2,1,1\n1,2,1\n-1,3,1\n3,-1,1\n0,2,1\n2,2,1\n";
        Sample::from_reader(BufReader::new(&csv[..]), true)
            .unwrap()
            .set_target("class")
    }

    #[test]
    fn max_loop_follows_algorithm_one() {
        let sample = xor_like_sample();
        // m = 12. T = 1: lg 12 / lg 14 = 0.94 -> c = 1.
        let b = SparsiBoost::<crate::DecisionTreeClassifier>::init(&sample, 1);
        assert_eq!(b.max_loop(), 1);
        // T = 3: lg 12 / lg 6 = 1.39 -> c = 2.
        let b = SparsiBoost::<crate::DecisionTreeClassifier>::init(&sample, 3);
        assert_eq!(b.max_loop(), 6);
        // T = 100: lg 12 / lg 2.12 = 3.31 -> c = 4.
        let b = SparsiBoost::<crate::DecisionTreeClassifier>::init(&sample, 100);
        assert_eq!(b.max_loop(), 400);
        // ν = √(2 ln 12 / 400) ≈ 0.111 is below the cap.
        let nu = (2.0 * 12f64.ln() / 400.0).sqrt();
        assert!((b.adaboostv_tolerance() - nu).abs() < 1e-15);
    }

    #[test]
    fn output_is_the_reported_sparsification_of_adaboostv() {
        let sample = xor_like_sample();
        let tree = DecisionTreeBuilder::new(&sample).max_depth(1).build();
        let target = 4;
        let mut booster = SparsiBoost::init(&sample, target).seed(3);
        let f = booster.run(&tree);

        // Rebuild the first stage with the same parameters.
        let mut adaboostv = AdaBoostV::init(&sample)
            .tolerance(booster.adaboostv_tolerance())
            .force_quit_at(booster.max_loop());
        let full = adaboostv.run(&tree);
        assert!(full.hypotheses.len() > target);

        assert!(!f.hypotheses.is_empty() && f.hypotheses.len() <= target);
        assert!(f.weights.iter().all(|w| *w > 0.0));
        assert!((f.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        // The reported error equals ‖U w - U w'‖_∞ on the training sample:
        // compare margins of the full and sparse ensembles directly.
        let full_margins = margin_columns(&sample, &[full]);
        let sparse_margins = margin_columns(&sample, &[f]);
        let direct = max_margin_change(
            &[full_margins[0].clone(), sparse_margins[0].clone()],
            &[1.0, 0.0],
            &[0.0, 1.0],
        );
        let reported = booster.sparsification_error().unwrap();
        assert!((direct - reported).abs() < 1e-12, "{direct} vs {reported}");

        // A second run with the same seed reproduces the output.
        let g = booster.run(&tree);
        assert_eq!(
            margin_columns(&sample, &[g]),
            sparse_margins,
            "same seed, same ensemble"
        );
    }

    #[test]
    fn current_hypothesis_is_sparse_and_empty_before_training() {
        let sample = xor_like_sample();
        let tree = DecisionTreeBuilder::new(&sample).max_depth(1).build();
        // m = 12, T = 2: c = ⌈lg 12 / lg 8⌉ = 2, so 4 AdaBoostV rounds.
        let mut booster = SparsiBoost::init(&sample, 2);
        booster.preprocess();
        assert!(booster.current_hypothesis().hypotheses.is_empty());
        for iteration in 1..=4 {
            assert!(booster.boost(&tree, iteration).is_continue());
            assert!(booster.current_hypothesis().hypotheses.len() <= 2);
        }
        assert!(booster.boost(&tree, 5).is_break());
        assert_eq!(booster.sparsification_error(), None);
    }

    #[test]
    #[should_panic(expected = "positive target")]
    fn zero_target_panics() {
        let sample = xor_like_sample();
        SparsiBoost::<crate::DecisionTreeClassifier>::init(&sample, 0);
    }
}
