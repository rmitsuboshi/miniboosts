//! Implementation of SparsiBoost (Grønlund, Larsen, and Mathiasen, 2019).
//! See the [`SparsiBoost`] struct for details and an example.
use crate::boosters::adaboostv::AdaBoostV;
use crate::core::CurrentHypothesis;
use crate::core::{Booster, Classifier, Sample, WeakLearner};
use crate::hypotheses::WeightedMajority;
use crate::optimization::sparsification::{
    margin_columns, max_margin_change, sparsify, sparsify_within,
};

use std::ops::ControlFlow;

const NO_TARGET: &str = "SparsiBoost needs a target: set n_hypotheses(T), tolerance(v), or both";

/// SparsiBoost by Grønlund, Larsen, and Mathiasen: AdaBoostV followed by a
/// margin-preserving sparsification of its ensemble.
///
/// The target is set by [`SparsiBoost::n_hypotheses`] (at most `T`
/// hypotheses), [`SparsiBoost::tolerance`] (gap `v`), or both; at least one
/// is required. `m` is the number of training examples, and the gap of a
/// combination is `ρ* - ρ`, where `ρ` is its minimal margin and `ρ*` the best
/// minimal margin of any convex combination of hypotheses in the class.
///
/// # Algorithm
/// Both forms in the paper run [`AdaBoostV`] and then sparsify its weights
/// `w` (normalized to `‖w‖_1 = 1`) with the halving passes of Algorithm 2,
/// keeping every margin `(U w)_i`, `u_ij = y_i h_j(x_i)`, nearly unchanged.
///
/// **Target `T` (Algorithm 1).**
/// 1. Run AdaBoostV for `c T` rounds, `c = ⌈lg m / lg(2 + m/T)⌉`, with
///    margin accuracy `ν = √(2 ln m / (c T))`, the value for which its
///    iteration bound `2 ln m / ν²` equals `c T`.
/// 2. Halve until at most `T` hypotheses have nonzero weight
///    ([`sparsify`]).
///
/// **Gap `v` (Section 1.2).**
/// 1. Run AdaBoostV with `ν = v/2` for `⌈2 ln m / ν²⌉ = ⌈8 ln m / v²⌉`
///    rounds, which gives gap at most `v/2`.
/// 2. Halve while every margin stays within `v/2` of AdaBoostV's
///    ([`sparsify_within`]), so the result has gap at most `v`.
///
/// **Both.** AdaBoostV runs as for `v`. Halving stops as for `v`, unless
/// more than `T` hypotheses are left; then the cap `T` takes precedence and
/// halving continues until at most `T` are left, even if margins move by
/// more than `v/2`. [`SparsiBoost::sparsification_within_tolerance`] tells
/// whether the `v/2` budget was kept.
///
/// Implementation choices (not part of the original method):
/// - For `T`, `c` is at least `1`. In both forms `ν` is capped at `1/2`:
///   AdaBoostV needs `ν < 1 + γ^min` (its smallest edge), so a larger `ν`
///   would make it panic whenever an edge is close to `0`. The cap is active
///   only when the paper's `ν` exceeds `1/2`, where the bounds are
///   constants anyway.
/// - For `v`, the paper states `T = O(lg(m v²) / v²)` with unstated
///   constants. Instead of choosing them, halving stops on the measured
///   margin change, which can be checked exactly; see [`sparsify_within`].
/// - AdaBoostV stops early if a hypothesis has edge `±1`; the ensemble then
///   has one hypothesis.
/// - The column subsets come from a heuristic discrepancy minimizer; see
///   [`discrepancy_coloring`](crate::optimization::sparsification::discrepancy_coloring)
///   and *Limitations*.
///
/// # Theoretical guarantee
/// Theorem 2.1: with target `T`, SparsiBoost returns at most `T` hypotheses
/// with gap `O(√(lg(2 + m/T) / T))`; equivalently, gap `v` is reached with
/// `T = O(lg(m v²) / v²)` hypotheses. Theorem 3.1 shows this is optimal:
/// for some data sets and hypothesis classes no combination of `T`
/// hypotheses does better. Both forms assume a weak learner returning edges
/// `γ_t ≥ ρ*` (AdaBoostV's assumption). With `T`, the sparsification step
/// also needs colorings that meet Spencer's bound in every halving pass
/// (Theorem 2.2). With `v`, the step's `v/2` budget is checked on the
/// training sample instead, so only AdaBoostV's assumption remains, but the
/// number of hypotheses is not bounded by `O(lg(m v²) / v²)` unless the
/// colorings meet Spencer's bound.
///
/// # Limitations
/// - The weak-learning assumption is not checked, and the discrepancy step
///   does not certify Spencer's bound. [`SparsiBoost::sparsification_error`]
///   reports the measured `‖U w - U w'‖_∞` on the training sample.
/// - The result depends on [`SparsiBoost::seed`]; equal seeds give equal
///   results.
/// - The output may have fewer than `T` hypotheses, because each halving
///   pass can overshoot. Equal hypotheses returned in different rounds are
///   kept as separate columns.
/// - A small `v` needs many AdaBoostV rounds: `⌈8 ln m / v²⌉`.
/// - Hypotheses must return confidences in `[-1, 1]` (as AdaBoostV
///   requires); labels must be `-1` or `+1` and both classes must be
///   present.
/// - Postprocessing evaluates every hypothesis on every example (`O(m R)`
///   confidences and memory for `R` AdaBoostV rounds) and the
///   sparsification takes `O(m R² + R³)` time.
///   [`CurrentHypothesis::current_hypothesis`] repeats this work on every
///   call, so logging every round with [`crate::Logger`] is expensive.
///
/// # Parameters
/// | Builder method | Default | Meaning |
/// | --- | --- | --- |
/// | [`n_hypotheses`](SparsiBoost::n_hypotheses) | none | At most `T` hypotheses in the output; takes precedence over `tolerance`. |
/// | [`tolerance`](SparsiBoost::tolerance) | none | Target gap `v`: AdaBoostV's `ν = v/2`, margin-change budget `v/2`. |
/// | [`seed`](SparsiBoost::seed) | `0` | Seed of the randomized discrepancy minimization. |
///
/// # Reference
/// Allan Grønlund, Kasper Green Larsen, and Alexander Mathiasen.
/// *Optimal Minimal Margin Maximization with Boosting.*
/// Proceedings of the 36th International Conference on Machine Learning
/// (ICML), PMLR 97:4392–4401, 2019. <https://arxiv.org/abs/1901.10789>
///
/// # Examples
/// Train on a held-out split, once with a target size and once with a
/// target gap.
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
/// // At most 3 hypotheses. m = 8, T = 3: c = ⌈lg 8 / lg(2 + 8/3)⌉ = 2.
/// let mut booster = SparsiBoost::init(&train).n_hypotheses(3).seed(42);
/// assert_eq!(booster.max_loop(), 6);
/// let f = booster.run(&learner);
/// assert!(!f.hypotheses.is_empty() && f.hypotheses.len() <= 3);
/// assert!((f.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
/// let predictions = f.predict_all(&test);
/// assert!(predictions.iter().all(|&p| p == -1 || p == 1));
///
/// // Gap v = 0.5: ⌈8 ln 8 / 0.25⌉ = 67 AdaBoostV rounds, then every training
/// // margin may move by at most v/2 = 0.25.
/// let mut booster = SparsiBoost::init(&train).tolerance(0.5);
/// assert_eq!(booster.max_loop(), 67);
/// let f = booster.run(&learner);
/// assert!(booster.sparsification_error().unwrap() <= 0.25);
/// assert_eq!(booster.sparsification_within_tolerance(), Some(true));
/// println!("{} hypotheses for gap 0.5", f.hypotheses.len());
/// ```
pub struct SparsiBoost<'a, H> {
    // Training sample
    sample: &'a Sample,

    // Cap `T` on the number of output hypotheses.
    n_hypotheses: Option<usize>,

    // Target gap `v`.
    tolerance: Option<f64>,

    // Seed of the discrepancy minimization.
    seed: u64,

    // The first stage; rebuilt by `preprocess`.
    adaboostv: AdaBoostV<'a, H>,

    // `‖U w - U w'‖_∞` of the last `postprocess`.
    sparsification_error: Option<f64>,
}

impl<'a, H> SparsiBoost<'a, H> {
    /// Constructs SparsiBoost for `sample`. Set a target with
    /// [`SparsiBoost::n_hypotheses`], [`SparsiBoost::tolerance`], or both
    /// before training. The seed defaults to `0`. The sample is validated
    /// when training starts.
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
    /// let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    /// let f = SparsiBoost::init(&sample).n_hypotheses(2).run(&learner);
    /// // A separating stump has edge 1, so AdaBoostV stops in round 1.
    /// assert_eq!(f.hypotheses.len(), 1);
    /// assert_eq!(f.predict_all(&sample), vec![-1, -1, -1, 1, 1, 1]);
    /// ```
    pub fn init(sample: &'a Sample) -> Self {
        Self {
            sample,
            n_hypotheses: None,
            tolerance: None,
            seed: 0,
            adaboostv: AdaBoostV::init(sample),
            sparsification_error: None,
        }
    }

    /// Caps the output at `n_hypotheses` hypotheses (the `T` of Algorithm 1).
    /// Without [`SparsiBoost::tolerance`], this also sets the AdaBoostV
    /// stage as in Algorithm 1. With it, `T` only caps the output and takes
    /// precedence over the margin-change budget `v/2`.
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
    /// # let csv = b"x,y,class\n\
    /// #     -2,1,-1\n-1,-2,-1\n1,-3,-1\n-3,2,-1\n0,-1,-1\n-2,-1,-1\n\
    /// #     2,1,1\n1,2,1\n-1,3,1\n3,-1,1\n0,2,1\n2,2,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// let learner = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    /// // Gap 0.25 sets the AdaBoostV stage; the output keeps at most 2
    /// // hypotheses even if that moves a margin by more than 0.125.
    /// let mut booster = SparsiBoost::init(&sample).tolerance(0.25).n_hypotheses(2);
    /// let f = booster.run(&learner);
    /// assert!(f.hypotheses.len() <= 2);
    /// let kept = booster.sparsification_within_tolerance().unwrap();
    /// assert_eq!(kept, booster.sparsification_error().unwrap() <= 0.125);
    /// ```
    pub fn n_hypotheses(mut self, n_hypotheses: usize) -> Self {
        assert!(n_hypotheses > 0, "SparsiBoost needs a positive target");
        self.n_hypotheses = Some(n_hypotheses);
        self
    }

    /// Sets the target gap `v`: AdaBoostV runs with margin accuracy
    /// `ν = min(v/2, 1/2)` for `⌈2 ln m / ν²⌉` rounds, and halving keeps every
    /// training margin within `v/2` of AdaBoostV's. This is not a bound on
    /// classification error.
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
    /// let booster = SparsiBoost::<DecisionTreeClassifier>::init(&sample).tolerance(0.5);
    /// assert_eq!(booster.adaboostv_tolerance(), 0.25);
    /// // m = 6: ⌈2 ln 6 / 0.25²⌉ = ⌈57.3⌉ = 58 rounds.
    /// assert_eq!(booster.max_loop(), 58);
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
    /// let f = SparsiBoost::init(&sample).n_hypotheses(2).seed(7).run(&learner);
    /// let g = SparsiBoost::init(&sample).n_hypotheses(2).seed(7).run(&learner);
    /// assert_eq!(f.weights, g.weights);
    /// ```
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Returns AdaBoostV's margin accuracy `ν` and number of rounds.
    fn first_stage(&self) -> (f64, usize) {
        let m = self.sample.shape().0 as f64;
        match (self.tolerance, self.n_hypotheses) {
            (Some(v), _) => {
                let nu = (v / 2.0).min(0.5);
                (nu, (2.0 * m.ln() / (nu * nu)).ceil() as usize)
            }
            (None, Some(t)) => {
                let c = (m.log2() / (2.0 + m / t as f64).log2()).ceil().max(1.0) as usize;
                let rounds = c.saturating_mul(t);
                ((2.0 * m.ln() / rounds as f64).sqrt().min(0.5), rounds)
            }
            (None, None) => panic!("{NO_TARGET}"),
        }
    }

    /// Returns the number of AdaBoostV rounds for `m` training examples:
    /// `⌈2 ln m / ν²⌉` with `ν = min(v/2, 1/2)` if a tolerance `v` is set,
    /// and otherwise `c T` with `c = max(1, ⌈lg m / lg(2 + m/T)⌉)`.
    /// Values exceeding `usize::MAX` saturate.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Panics
    /// Panics if neither [`SparsiBoost::n_hypotheses`] nor
    /// [`SparsiBoost::tolerance`] is set.
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
    /// let booster = SparsiBoost::<DecisionTreeClassifier>::init(&sample).n_hypotheses(2);
    /// assert_eq!(booster.max_loop(), 4);
    /// // With a tolerance, the AdaBoostV stage follows v instead of T.
    /// let booster = booster.tolerance(0.5);
    /// assert_eq!(booster.max_loop(), 58);
    /// ```
    pub fn max_loop(&self) -> usize {
        self.first_stage().1
    }

    /// Returns the margin accuracy `ν` passed to AdaBoostV:
    /// `min(v/2, 1/2)` if a tolerance `v` is set, and otherwise
    /// `min(√(2 ln m / (cT)), 1/2)` with `cT =` [`SparsiBoost::max_loop`].
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Panics
    /// Panics if neither [`SparsiBoost::n_hypotheses`] nor
    /// [`SparsiBoost::tolerance`] is set.
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
    /// let booster = SparsiBoost::<DecisionTreeClassifier>::init(&sample).n_hypotheses(2);
    /// // m = 6, cT = 4: √(2 ln 6 / 4) ≈ 0.95 is capped at 1/2.
    /// assert_eq!(booster.adaboostv_tolerance(), 0.5);
    /// ```
    pub fn adaboostv_tolerance(&self) -> f64 {
        self.first_stage().0
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

    /// Returns whether the last run kept every training margin within `v/2`
    /// of AdaBoostV's, i.e. [`SparsiBoost::sparsification_error`] `≤ v/2`.
    /// This can be `false` only when the cap [`SparsiBoost::n_hypotheses`]
    /// forced extra halving. It concerns the sparsification step only: the
    /// gap of AdaBoostV's ensemble is not measured, since `ρ*` is unknown.
    /// Returns `None` before the first run or when no tolerance is set.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// See [`SparsiBoost::n_hypotheses`].
    pub fn sparsification_within_tolerance(&self) -> Option<bool> {
        let v = self.tolerance?;
        self.sparsification_error.map(|e| e <= v / 2.0)
    }
}

impl<H> SparsiBoost<'_, H>
where
    H: Classifier + Clone,
{
    /// Sparsifies `f` by the rule set by the builder; returns the result and
    /// its `‖U w - U w'‖_∞`.
    fn sparsify(&self, f: &WeightedMajority<H>) -> (WeightedMajority<H>, f64) {
        if f.hypotheses.is_empty() {
            return (WeightedMajority::from_slices(&[], &f.hypotheses), 0.0);
        }
        let columns = margin_columns(self.sample, &f.hypotheses);
        let sparse = match (self.tolerance, self.n_hypotheses) {
            (Some(v), cap) => sparsify_within(&columns, &f.weights, v / 2.0, cap, self.seed),
            (None, Some(t)) => sparsify(&columns, &f.weights, t, self.seed),
            (None, None) => panic!("{NO_TARGET}"),
        };
        let l1 = f.weights.iter().map(|w| w.abs()).sum::<f64>();
        let before: Vec<f64> = f.weights.iter().map(|w| w / l1).collect();
        let error = max_margin_change(&columns, &before, &sparse);
        (WeightedMajority::from_slices(&sparse, &f.hypotheses), error)
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
        let or_dash = |v: Option<String>| v.unwrap_or_else(|| "-".to_string());
        let (nu, rounds) = if self.tolerance.is_some() || self.n_hypotheses.is_some() {
            let (nu, rounds) = self.first_stage();
            (Some(nu.to_string()), Some(rounds.to_string()))
        } else {
            (None, None)
        };
        let info = Vec::from([
            ("# of examples", format!("{}", n_examples)),
            ("# of features", format!("{}", n_feature)),
            (
                "Max # of hypotheses",
                or_dash(self.n_hypotheses.map(|t| t.to_string())),
            ),
            (
                "Tolerance (gap)",
                or_dash(self.tolerance.map(|v| v.to_string())),
            ),
            ("AdaBoostV rounds", or_dash(rounds)),
            ("AdaBoostV tolerance", or_dash(nu)),
            ("Seed", format!("{}", self.seed)),
        ]);
        Some(info)
    }

    fn preprocess(&mut self) {
        let (nu, rounds) = self.first_stage();
        self.sample.is_valid_binary_instance();
        self.sparsification_error = None;
        self.adaboostv = AdaBoostV::init(self.sample)
            .tolerance(nu)
            .force_quit_at(rounds);
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
    use crate::{DecisionTreeBuilder, DecisionTreeClassifier};
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

    /// `‖U w - U w'‖_∞` computed from the two ensembles' margins directly.
    fn margin_distance<H: Classifier>(
        sample: &Sample,
        f: &WeightedMajority<H>,
        g: &WeightedMajority<H>,
    ) -> f64 {
        let fm = margin_columns(sample, std::slice::from_ref(f));
        let gm = margin_columns(sample, std::slice::from_ref(g));
        fm[0]
            .iter()
            .zip(&gm[0])
            .fold(0.0, |acc, (a, b)| f64::max(acc, (a - b).abs()))
    }

    #[test]
    fn first_stage_follows_algorithm_one_for_a_target_size() {
        let sample = xor_like_sample();
        let b = |t| SparsiBoost::<DecisionTreeClassifier>::init(&sample).n_hypotheses(t);
        // m = 12. T = 1: lg 12 / lg 14 = 0.94 -> c = 1.
        assert_eq!(b(1).max_loop(), 1);
        // T = 3: lg 12 / lg 6 = 1.39 -> c = 2.
        assert_eq!(b(3).max_loop(), 6);
        // T = 100: lg 12 / lg 2.12 = 3.31 -> c = 4; ν = √(2 ln 12 / 400).
        assert_eq!(b(100).max_loop(), 400);
        let nu = (2.0 * 12f64.ln() / 400.0).sqrt();
        assert!((b(100).adaboostv_tolerance() - nu).abs() < 1e-15);
    }

    #[test]
    fn first_stage_follows_the_gap_when_a_tolerance_is_set() {
        let sample = xor_like_sample();
        let b = SparsiBoost::<DecisionTreeClassifier>::init(&sample).tolerance(0.5);
        // ν = v/2 = 0.25; ⌈2 ln 12 / 0.0625⌉ = ⌈79.5⌉ = 80.
        assert_eq!((b.adaboostv_tolerance(), b.max_loop()), (0.25, 80));
        // T does not change the first stage once v is set.
        let b = b.n_hypotheses(3);
        assert_eq!((b.adaboostv_tolerance(), b.max_loop()), (0.25, 80));
        // ν is capped at 1/2 for v > 1: ⌈2 ln 12 / 0.25⌉ = 20.
        let b = SparsiBoost::<DecisionTreeClassifier>::init(&sample).tolerance(3.0);
        assert_eq!((b.adaboostv_tolerance(), b.max_loop()), (0.5, 20));
    }

    #[test]
    fn target_size_output_is_the_reported_sparsification_of_adaboostv() {
        let sample = xor_like_sample();
        let tree = DecisionTreeBuilder::new(&sample).max_depth(1).build();
        let target = 4;
        let mut booster = SparsiBoost::init(&sample).n_hypotheses(target).seed(3);
        let f = booster.run(&tree);

        let full = AdaBoostV::init(&sample)
            .tolerance(booster.adaboostv_tolerance())
            .force_quit_at(booster.max_loop())
            .run(&tree);
        assert!(full.hypotheses.len() > target);

        assert!(!f.hypotheses.is_empty() && f.hypotheses.len() <= target);
        assert!(f.weights.iter().all(|w| *w > 0.0));
        assert!((f.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let reported = booster.sparsification_error().unwrap();
        let direct = margin_distance(&sample, &full, &f);
        assert!((direct - reported).abs() < 1e-12, "{direct} vs {reported}");
        // No tolerance, so there is no budget to report on.
        assert_eq!(booster.sparsification_within_tolerance(), None);

        // A second run with the same seed reproduces the output.
        let g = booster.run(&tree);
        assert_eq!(margin_distance(&sample, &f, &g), 0.0);
    }

    #[test]
    fn tolerance_alone_keeps_margins_within_half_the_gap() {
        let sample = xor_like_sample();
        let tree = DecisionTreeBuilder::new(&sample).max_depth(1).build();
        let v = 0.5;
        let mut booster = SparsiBoost::init(&sample).tolerance(v).seed(1);
        let f = booster.run(&tree);
        let full = AdaBoostV::init(&sample)
            .tolerance(booster.adaboostv_tolerance())
            .force_quit_at(booster.max_loop())
            .run(&tree);
        // Without a cap the v/2 budget is never exceeded.
        assert!(margin_distance(&sample, &full, &f) <= v / 2.0 + 1e-12);
        assert_eq!(booster.sparsification_within_tolerance(), Some(true));
        assert!(f.hypotheses.len() < full.hypotheses.len());
    }

    #[test]
    fn the_cap_takes_precedence_over_the_tolerance() {
        let sample = xor_like_sample();
        let tree = DecisionTreeBuilder::new(&sample).max_depth(1).build();
        // v = 0.5 (80 AdaBoostV rounds, budget 0.25); the cap of one
        // hypothesis forces halving past the budget.
        let mut booster = SparsiBoost::init(&sample)
            .tolerance(0.5)
            .n_hypotheses(1)
            .seed(2);
        let f = booster.run(&tree);
        assert_eq!(f.hypotheses.len(), 1);
        let error = booster.sparsification_error().unwrap();
        assert_eq!(
            booster.sparsification_within_tolerance(),
            Some(error <= 0.25)
        );
        // On this sample one stump moves some margin by more than 0.25.
        assert_eq!(booster.sparsification_within_tolerance(), Some(false));
    }

    #[test]
    fn current_hypothesis_is_sparse_and_empty_before_training() {
        let sample = xor_like_sample();
        let tree = DecisionTreeBuilder::new(&sample).max_depth(1).build();
        // m = 12, T = 2: c = ⌈lg 12 / lg 8⌉ = 2, so 4 AdaBoostV rounds.
        let mut booster = SparsiBoost::init(&sample).n_hypotheses(2);
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
    #[should_panic(expected = "needs a target")]
    fn training_without_a_target_panics() {
        let sample = xor_like_sample();
        SparsiBoost::<DecisionTreeClassifier>::init(&sample).preprocess();
    }

    #[test]
    fn invalid_parameters_panic() {
        let sample = xor_like_sample();
        let init = || SparsiBoost::<DecisionTreeClassifier>::init(&sample);
        assert!(std::panic::catch_unwind(|| init().n_hypotheses(0)).is_err());
        for v in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(std::panic::catch_unwind(|| init().tolerance(v)).is_err());
        }
        assert!(std::panic::catch_unwind(|| init().max_loop()).is_err());
    }
}
