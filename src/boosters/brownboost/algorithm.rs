//! Implementation of BrownBoost (Freund, 2001) as [`BrownBoost`].
//! See the [`BrownBoost`] struct for details and an example.
use super::solver::{Event, Round};
use crate::core::CurrentHypothesis;
use crate::core::tools::special::erfc_inverse;
use crate::core::{Booster, Classifier, Sample, WeakLearner, tools::helpers};
use crate::hypotheses::WeightedMajority;

use std::ops::ControlFlow;

/// BrownBoost by Freund: an adaptive version of boost-by-majority that
/// "gives up" on examples far on the wrong side of the margin, which makes it
/// more tolerant of label noise than AdaBoost.
///
/// # Algorithm
/// Let `m` be the number of training examples, `y_j ∈ {-1, +1}` the labels,
/// `h_i(x_j) ∈ [-1, 1]` the weak hypotheses' confidences, `ε = tolerance`
/// the target training error, and `c = (erfc⁻¹(ε))²`, so that
/// `1 - erf(√c) = ε`. Starting from margins `r_j = 0` and remaining time
/// `s = c`, round `i` follows Figure 2 of the paper:
/// 1. Give example `j` the weight `W_j = exp(-(r_j + s)² / c)` and call the
///    weak learner on the normalized weights to obtain `h_i`, with
///    correlation `γ_i = Σ_j W_j y_j h_i(x_j) / Σ_j W_j`.
/// 2. Solve the differential Equation (5),
///    ```txt
///    dt/dα = γ(α, t) = Σ_j exp(-d_j²/c) b_j / Σ_j exp(-d_j²/c),
///    d_j = r_j + s + α b_j - t,   b_j = y_j h_i(x_j),
///    ```
///    from `α = t = 0` until `γ(α, t) ≤ ν` or `t = s`, giving `α_i, t_i`.
/// 3. Update `r_j ← r_j + α_i b_j` and `s ← s - t_i`; stop once `s ≤ 0`.
///
/// The output is the [`WeightedMajority`] `sign(Σ_i α_i h_i(x))` (the
/// deterministic output of Figure 2) with normalized weights; every
/// `α_i > 0`.
///
/// Implementation choices (not part of the original method):
/// - Equation (5) is solved through the invariant used in the proof of
///   Theorem 2: along its solution the potential
///   `Σ_j erf(d_j / √c)` stays equal to its value at `α = t = 0`. That
///   fixes `t` as a function of `α`, computed by safeguarded Newton steps;
///   `α` is then found by geometric growth and bisection on the stopping
///   condition. The paper suggests a two-dimensional Newton–Raphson
///   iteration (Section 7.2) instead.
/// - Example weights are computed in log space, and differences of `erf`
///   use `erfc` in the tails, so that the potential does not cancel to
///   zero for large margins.
/// - If `γ_i ≤ ν` (the solution would be `α = t = 0` and training could not
///   progress) or the solver fails, training stops without adding `h_i`
///   and [`BrownBoost::numerical_stop_reason`] says why.
///
/// # Theoretical guarantee
/// Theorem 2 of the paper: if BrownBoost exits its loop (`s ≤ 0`, see
/// [`BrownBoost::remaining_time`]), then
/// `(1/m) Σ_j |y_j - erf(Σ_i α_i h_i(x_j) / √c)| = 1 - erf(√c) = ε`.
/// A misclassified example has `|y_j - erf(·)| ≥ 1`, so the training error of
/// the deterministic output is at most `ε`. The theorem assumes exact
/// solutions of Equation (5) and makes no assumption on the weak learner,
/// but the loop exits only if the weak hypotheses keep enough correlation.
/// For a variant of the update (Figure 3, not implemented here), Theorem 3
/// bounds the number of rounds when `Σ_i (γ_i² - ν²) ≥ 32 c^{3/2}
/// √(ln 2 + 2 ln(c/ε))`. As `ε → 0` (`c → ∞`), the weight choice tends to
/// that of AdaBoost (Section 5).
///
/// # Limitations
/// - There is no iteration bound: if the weak hypotheses' correlations
///   decay, `s` may decrease very slowly. Set [`BrownBoost::force_quit_at`]
///   to bound the work; stopping before `s ≤ 0` voids Theorem 2.
/// - A `tolerance` below the attainable training error (for example, below
///   the label-noise rate) can keep the loop from exiting; Section 5 of the
///   paper suggests searching `c` instead.
/// - The equation is solved to floating-point accuracy, not exactly.
///   The search for `α` returns a crossing of the stopping condition that is
///   the first one unless the condition switches back and forth within one
///   doubling of `α`.
/// - `tolerance` is the target training error, not a bound on test error,
///   and has a different meaning than in other boosters.
/// - Training panics if a confidence lies outside `[-1, 1]` or is not
///   finite. Labels must be `-1` or `+1` and both classes must be present.
/// - Each round costs `O(m)` for the weights plus `O(m)` evaluations of
///   `erf` per solver step (a few hundred steps in total).
///
/// # Parameters
/// | Builder method | Default | Meaning |
/// | --- | --- | --- |
/// | [`tolerance`](BrownBoost::tolerance) | `0.1` | Target training error `ε`; sets `c = (erfc⁻¹(ε))²`. |
/// | [`edge_threshold`](BrownBoost::edge_threshold) | `1e-3` | `ν`: each round's solution stops when the correlation drops to `ν`. |
/// | [`force_quit_at`](BrownBoost::force_quit_at) | none | Stops after the given number of rounds. |
///
/// # Reference
/// Yoav Freund.
/// *An Adaptive Version of the Boost by Majority Algorithm.*
/// Machine Learning, 43(3):293–318, 2001.
/// <https://doi.org/10.1023/A:1010852229904>
///
/// # Examples
/// Train on a held-out split with decision stumps and check whether the
/// loop exited, which is when Theorem 2 applies.
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
/// // Target training error 0.2, at most 100 rounds.
/// let mut booster = BrownBoost::init(&train).tolerance(0.2).force_quit_at(100);
/// let f = booster.run(&learner);
///
/// if booster.remaining_time() <= Some(0.0) {
///     // Theorem 2: training error at most ε = 0.2.
///     let errors = f
///         .predict_all(&train)
///         .iter()
///         .zip(train.target())
///         .filter(|&(&p, &y)| p as f64 != y)
///         .count();
///     assert!(errors as f64 / 8.0 <= 0.2);
/// } else if let Some(reason) = booster.numerical_stop_reason() {
///     eprintln!("stopped early: {reason}");
/// }
/// assert!(f.predict_all(&test).iter().all(|&p| p == -1 || p == 1));
/// ```
pub struct BrownBoost<'a, F> {
    // Training sample
    sample: &'a Sample,

    // Target training error `ε`.
    tolerance: f64,

    // Threshold `ν` on the correlation in Equation (5).
    edge_threshold: f64,

    // Optional round limit.
    force_quit_at: Option<usize>,

    // `c = (erfc⁻¹(ε))²`, set by `preprocess`.
    c: f64,

    // Remaining time `s`; `None` before training.
    remaining_time: Option<f64>,

    // Margins `r_j = y_j Σ_i α_i h_i(x_j)`.
    margins: Vec<f64>,

    // Hypothesis weights `α_i`.
    weights: Vec<f64>,

    // Hypotheses obtained by the weak learner.
    hypotheses: Vec<F>,

    // Why training stopped early, if it did.
    numerical_stop_reason: Option<&'static str>,
}

impl<'a, F> BrownBoost<'a, F> {
    /// Constructs a new instance of `BrownBoost` for `sample` with target
    /// training error `0.1`, edge threshold `1e-3`, and no round limit.
    /// The sample is validated when training starts.
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
    /// let mut booster = BrownBoost::init(&sample);
    /// let f = booster.run(&learner);
    /// // A separating stump keeps γ = 1, so one round uses up all the time.
    /// assert_eq!(f.hypotheses.len(), 1);
    /// assert_eq!(booster.remaining_time(), Some(0.0));
    /// assert_eq!(f.predict_all(&sample), vec![-1, -1, -1, 1, 1, 1]);
    /// ```
    pub fn init(sample: &'a Sample) -> Self {
        Self {
            sample,
            tolerance: 0.1,
            edge_threshold: 1e-3,
            force_quit_at: None,
            c: f64::NAN,
            remaining_time: None,
            margins: Vec::new(),
            weights: Vec::new(),
            hypotheses: Vec::new(),
            numerical_stop_reason: None,
        }
    }

    /// Sets the target training error `ε`, which determines the total time
    /// `c = (erfc⁻¹(ε))²` (see [`BrownBoost::total_time`]). A smaller `ε`
    /// gives a larger `c` and more rounds. This is not a bound on test error.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Panics
    /// Panics unless `0 < tolerance < 1`.
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
    /// let booster = BrownBoost::<DecisionTreeClassifier>::init(&sample).tolerance(0.1);
    /// // erfc(1.1631) = 0.1, so c = 1.1631² ≈ 1.3528.
    /// assert!((booster.total_time() - 1.352_771).abs() < 1e-6);
    /// ```
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        assert!(
            tolerance > 0.0 && tolerance < 1.0,
            "BrownBoost tolerance must lie in (0, 1)"
        );
        self.tolerance = tolerance;
        self
    }

    /// Sets the threshold `ν` of Equation (5): each round's solution stops
    /// once the weighted correlation `γ(α, t)` drops to `ν` (unless the time
    /// runs out first). A weak hypothesis with `γ_i ≤ ν` stops training.
    /// The paper requires a small `ν > 0` "to avoid degenerate cases"; the
    /// default `1e-3` is not taken from the paper.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Panics
    /// Panics unless `0 ≤ edge_threshold < 1`.
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
    /// let f = BrownBoost::init(&sample)
    ///     .edge_threshold(0.05)
    ///     .force_quit_at(20)
    ///     .run(&learner);
    /// assert!(f.weights.iter().all(|w| *w > 0.0));
    /// ```
    pub fn edge_threshold(mut self, edge_threshold: f64) -> Self {
        assert!(
            (0.0..1.0).contains(&edge_threshold),
            "BrownBoost edge threshold must lie in [0, 1)"
        );
        self.edge_threshold = edge_threshold;
        self
    }

    /// Stops after `iteration` rounds if the remaining time has not run out
    /// by then. Stopping before `s ≤ 0` voids Theorem 2.
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
    /// let f = BrownBoost::init(&sample).tolerance(0.01).force_quit_at(3).run(&learner);
    /// assert!(f.hypotheses.len() <= 3);
    /// ```
    pub fn force_quit_at(mut self, iteration: usize) -> Self {
        self.force_quit_at = Some(iteration);
        self
    }

    /// Returns the total time `c = (erfc⁻¹(ε))²` for the configured
    /// tolerance `ε`, i.e. the `c` with `1 - erf(√c) = ε`.
    ///
    /// Time complexity: `O(1)` (a fixed-length bisection).
    ///
    /// # Examples
    /// See [`BrownBoost::tolerance`].
    pub fn total_time(&self) -> f64 {
        erfc_inverse(self.tolerance).powi(2)
    }

    /// Returns the remaining time `s` after the last round, or `None` before
    /// training. `Some(s)` with `s ≤ 0` means the loop exited, so Theorem 2
    /// applies; a positive value means training stopped early (round limit,
    /// or see [`BrownBoost::numerical_stop_reason`]).
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// See [`BrownBoost`].
    pub fn remaining_time(&self) -> Option<f64> {
        self.remaining_time
    }

    /// Returns why training stopped before the remaining time ran out,
    /// other than the round limit: a weak hypothesis with correlation at
    /// most the edge threshold, or a failure to solve Equation (5). The
    /// hypothesis of that round is not added. Returns `None` otherwise; the
    /// value is reset by [`Booster::preprocess`].
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
    /// use miniboosts::{Classifier, WeakLearner};
    ///
    /// // A weak learner whose hypothesis abstains everywhere (correlation 0).
    /// #[derive(Clone)]
    /// struct Abstain;
    /// impl Classifier for Abstain {
    ///     fn confidence(&self, _: &Sample, _: usize) -> f64 {
    ///         0.0
    ///     }
    /// }
    /// impl WeakLearner for Abstain {
    ///     type Hypothesis = Self;
    ///     fn produce(&self, _: &Sample, _: &[f64]) -> Self {
    ///         Self
    ///     }
    /// }
    ///
    /// let mut booster = BrownBoost::init(&sample);
    /// let f = booster.run(&Abstain);
    /// assert!(f.hypotheses.is_empty());
    /// assert!(booster.numerical_stop_reason().is_some());
    /// assert_eq!(booster.remaining_time(), Some(booster.total_time()));
    /// ```
    pub fn numerical_stop_reason(&self) -> Option<&str> {
        self.numerical_stop_reason
    }
}

impl<F> Booster<F> for BrownBoost<'_, F>
where
    F: Classifier + Clone,
{
    type Output = WeightedMajority<F>;

    fn name(&self) -> &str {
        "BrownBoost"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_examples, n_feature) = self.sample.shape();
        let quit = if let Some(it) = self.force_quit_at {
            format!("At round {it}")
        } else {
            "-".to_string()
        };
        let info = Vec::from([
            ("# of examples", format!("{}", n_examples)),
            ("# of features", format!("{}", n_feature)),
            ("Tolerance", format!("{}", self.tolerance)),
            ("Total time c", format!("{}", self.total_time())),
            ("Edge threshold", format!("{}", self.edge_threshold)),
            ("Force quit", quit),
        ]);
        Some(info)
    }

    fn preprocess(&mut self) {
        self.sample.is_valid_binary_instance();
        let n_examples = self.sample.shape().0;
        self.c = self.total_time();
        self.remaining_time = Some(self.c);
        self.margins = vec![0.0; n_examples];
        self.weights = Vec::new();
        self.hypotheses = Vec::new();
        self.numerical_stop_reason = None;
    }

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize) -> ControlFlow<usize>
    where
        W: WeakLearner<Hypothesis = F>,
    {
        let s = self.remaining_time.expect("call preprocess before boost");
        if s <= 0.0 || self.numerical_stop_reason.is_some() {
            return ControlFlow::Break(iteration - 1);
        }
        if self.force_quit_at.is_some_and(|n| iteration > n) {
            return ControlFlow::Break(iteration - 1);
        }

        // Step 1: W_j = exp(-(r_j + s)² / c), normalized in log space.
        let c = self.c;
        let log_weights: Vec<f64> = self.margins.iter().map(|r| -(r + s).powi(2) / c).collect();
        let dist = helpers::softmax(&log_weights);
        let h = weak_learner.produce(self.sample, &dist);

        let steps: Vec<f64> = helpers::margins(self.sample, &h).collect();
        assert!(
            steps.iter().all(|b| (-1.0..=1.0).contains(b)),
            "BrownBoost needs finite confidences in [-1, 1]"
        );

        // Step 3: solve Equation (5).
        let old_margins: Vec<f64> = self.margins.iter().map(|r| r + s).collect();
        let round = Round::new(&old_margins, &steps, c);
        let step = match round.solve(self.edge_threshold, s) {
            Ok(step) => step,
            Err(reason) => {
                self.numerical_stop_reason = Some(reason);
                return ControlFlow::Break(iteration - 1);
            }
        };

        // Steps 4-5.
        self.margins
            .iter_mut()
            .zip(&steps)
            .for_each(|(r, b)| *r += step.alpha * b);
        let s = match step.event {
            Event::ClockExpired => 0.0,
            Event::EdgeReached => s - step.time,
        };
        self.remaining_time = Some(s);
        self.weights.push(step.alpha);
        self.hypotheses.push(h);

        if s <= 0.0 {
            ControlFlow::Break(iteration)
        } else {
            ControlFlow::Continue(())
        }
    }

    fn postprocess(&mut self) -> Self::Output {
        WeightedMajority::from_slices(&self.weights, &self.hypotheses)
    }
}

impl<H> CurrentHypothesis for BrownBoost<'_, H>
where
    H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        WeightedMajority::from_slices(&self.weights, &self.hypotheses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DecisionTreeBuilder, DecisionTreeClassifier};
    use libm::erf;
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

    /// Average potential `(1/m) Σ_j erf((r_j + s) / √c)`.
    fn average_potential<F>(booster: &BrownBoost<'_, F>) -> f64 {
        let s = booster.remaining_time.unwrap();
        let sqrt_c = booster.c.sqrt();
        let total: f64 = booster.margins.iter().map(|r| erf((r + s) / sqrt_c)).sum();
        total / booster.margins.len() as f64
    }

    #[test]
    fn the_average_potential_is_invariant() {
        // Proof of Theorem 2: the average potential stays erf(√c) in every
        // round. Each round solves Φ = 0 to about 1e-15 per example.
        let sample = xor_like_sample();
        let tree = DecisionTreeBuilder::new(&sample).max_depth(1).build();
        let mut booster = BrownBoost::init(&sample).tolerance(0.2);
        booster.preprocess();
        let start = erf(booster.c.sqrt());
        assert!((average_potential(&booster) - start).abs() < 1e-15);
        let mut rounds = 0;
        for iteration in 1..=50 {
            let flow = booster.boost(&tree, iteration);
            rounds = iteration;
            let drift = (average_potential(&booster) - start).abs();
            assert!(drift < 1e-12, "round {iteration}: drift {drift}");
            if flow.is_break() {
                break;
            }
        }
        assert!(rounds > 1, "the sample needs more than one stump");
    }

    #[test]
    fn exiting_the_loop_bounds_the_training_error() {
        let sample = xor_like_sample();
        let tree = DecisionTreeBuilder::new(&sample).max_depth(1).build();
        for epsilon in [0.3, 0.2] {
            let mut booster = BrownBoost::init(&sample)
                .tolerance(epsilon)
                .force_quit_at(500);
            let f = booster.run(&tree);
            assert_eq!(booster.numerical_stop_reason(), None);
            assert_eq!(booster.remaining_time(), Some(0.0), "ε = {epsilon}");
            // Theorem 2 with the stochastic output p = erf(r/√c).
            let sqrt_c = booster.c.sqrt();
            let loss: f64 = booster
                .margins
                .iter()
                .map(|r| 1.0 - erf(r / sqrt_c))
                .sum::<f64>()
                / 12.0;
            assert!((loss - epsilon).abs() < 1e-12, "{loss} vs {epsilon}");
            // The deterministic output errs on at most a fraction ε.
            let errors = f
                .predict_all(&sample)
                .iter()
                .zip(sample.target())
                .filter(|&(&p, &y)| p as f64 != y)
                .count();
            assert!(errors as f64 / 12.0 <= epsilon);
        }
    }

    #[test]
    fn margins_match_the_output_ensemble() {
        let sample = xor_like_sample();
        let tree = DecisionTreeBuilder::new(&sample).max_depth(1).build();
        let mut booster = BrownBoost::init(&sample).tolerance(0.2);
        let f = booster.run(&tree);
        let total: f64 = booster.weights.iter().sum();
        for (j, r) in booster.margins.iter().enumerate() {
            let y = sample.target()[j];
            let expected = y * f.confidence(&sample, j) * total;
            assert!((r - expected).abs() < 1e-12, "{j}: {r} vs {expected}");
        }
    }

    #[test]
    fn force_quit_and_reset() {
        let sample = xor_like_sample();
        let tree = DecisionTreeBuilder::new(&sample).max_depth(1).build();
        let mut booster = BrownBoost::init(&sample).tolerance(0.01).force_quit_at(2);
        assert_eq!(booster.remaining_time(), None);
        let f = booster.run(&tree);
        assert_eq!(f.hypotheses.len(), 2);
        assert!(booster.remaining_time().unwrap() > 0.0);
        // A second run starts afresh.
        let g = booster.run(&tree);
        assert_eq!(f.weights, g.weights);
    }

    #[test]
    fn invalid_parameters_panic() {
        let sample = xor_like_sample();
        let init = || BrownBoost::<DecisionTreeClassifier>::init(&sample);
        for eps in [0.0, 1.0, -0.1, f64::NAN] {
            assert!(std::panic::catch_unwind(|| init().tolerance(eps)).is_err());
        }
        for nu in [-0.1, 1.0, f64::NAN] {
            assert!(std::panic::catch_unwind(|| init().edge_threshold(nu)).is_err());
        }
    }
}
