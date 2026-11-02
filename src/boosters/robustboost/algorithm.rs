//! Implementation of RobustBoost (Freund, 2009) as [`RobustBoost`].
//! See the [`RobustBoost`] struct for details and an example.
use super::solver::{Event, Potential, Round};
use crate::core::CurrentHypothesis;
use crate::core::{Booster, Classifier, Sample, WeakLearner, tools::helpers};
use crate::hypotheses::WeightedMajority;

use std::ops::ControlFlow;

/// RobustBoost by Freund: a variant of [`BrownBoost`](crate::BrownBoost)
/// with a non-convex potential that aims to keep most training margins above
/// a goal `θ` while giving up on (possibly mislabeled) examples, which makes
/// it robust to label noise.
///
/// # Algorithm
/// Time runs from `t = 0` to `t = 1`. With the goal margin `θ`, the final
/// width `σ_f`, and the error goal `ε = tolerance`, the potential is
/// ```txt
/// σ²(t) = (σ_f² + 1) e^{2(1-t)} - 1,   μ(t) = (θ - 2ρ) e^{1-t} + 2ρ,
/// Φ(m, t) = 1 - N((m - μ(t)) / σ(t)),   w(m, t) = exp(-(m - μ(t))² / (2σ(t)²)),
/// ```
/// (Equations (4)–(7); `N` is the standard normal distribution function,
/// see below) with `ρ` set by Equation (8), `ε = Φ(0, 0)`. Starting from
/// scores `m_j = 0`, each round of the paper's algorithm
/// 1. calls the weak learner on the distribution `D(j) ∝ w(m_j, t)` to obtain
///    `h_k`;
/// 2. finds `Δm > 0` and `0 < Δt ≤ 1 - t` such that, with
///    `m'_j = m_j e^{-Δt} + y_j h_k(x_j) Δm`, the average potential is kept,
///    `Σ_j Φ(m'_j, t + Δt) = Σ_j Φ(m_j, t)`, and either
///    `Σ_j y_j h_k(x_j) w(m'_j, t + Δt) = 0` or `Δt = 1 - t`;
/// 3. updates `t ← t + Δt`, `m_j ← m'_j`, and `H ← H e^{-Δt} + Δm h_k`,
///
/// until `t ≥ 1`. The factor `e^{-Δt}` makes the scores mean-reverting
/// (Ornstein–Uhlenbeck), which keeps their scale bounded. The output is the
/// [`WeightedMajority`] with the (normalized) coefficients of `H`.
///
/// Implementation choices (not part of the original method):
/// - The paper defines `erf(a) = (1/√π) ∫_{-∞}^a e^{-x²} dx` in (4) but uses
///   the weight (7) with `2σ²`, which is proportional to `-∂Φ/∂m` only if
///   `erf` is the standard normal distribution function `N`. This
///   implementation uses `N`, making the weight the derivative of the
///   potential, as the algorithm requires.
/// - Step 2 is solved along the potential-preserving curve from
///   `(Δm, Δt) = (0, 0)`: for each `Δm`, `Δt` is found by bisection between
///   the previous value and `1 - t`, assuming the potential increases in
///   `Δt` there; `Δm` grows geometrically and is then bisected on the
///   stopping condition.
/// - A weak hypothesis with nonpositive weighted correlation, or a failure
///   of step 2, stops training without adding the hypothesis; see
///   [`RobustBoost::numerical_stop_reason`].
///
/// # Theoretical guarantee
/// Step 2 keeps the average potential at its initial value
/// `Φ(0, 0) = ε`. If training reaches `t = 1` (see
/// [`RobustBoost::remaining_time`]), then `(1/m) Σ_j Φ(m_j, 1) = ε`, and since
/// `μ(1) = θ` gives `Φ(m, 1) ≥ 1/2` for `m ≤ θ`, at most a fraction `2ε` of
/// the training examples have score `y_j H(x_j) ≤ θ` (this consequence is
/// derived here from the invariant). The paper does not bound the number
/// of rounds.
///
/// # Limitations
/// - If `ε` is too small, the clock may never reach `1`; the paper suggests
///   searching for the smallest `ε` for which training terminates. Set
///   [`RobustBoost::force_quit_at`] to bound the work.
/// - `θ` is to be chosen by cross-validation; the scores `y_j H(x_j)` are
///   not normalized by the coefficients, so `θ` refers to their scale.
/// - The solver assumes that the potential increases in `Δt` along the
///   search; this is not guaranteed by the paper.
/// - Confidences must lie in `[-1, 1]` (the paper uses `±1`). Labels must be
///   `-1` or `+1` and both classes must be present.
///
/// # Parameters
/// | Builder method | Default | Meaning |
/// | --- | --- | --- |
/// | [`tolerance`](RobustBoost::tolerance) | `0.1` | Error goal `ε`; sets `ρ`. |
/// | [`theta`](RobustBoost::theta) | `0` | Goal margin `θ ≥ 0`. |
/// | [`sigma_f`](RobustBoost::sigma_f) | `0.1` | Final width `σ_f > 0` (the paper's setting). |
/// | [`force_quit_at`](RobustBoost::force_quit_at) | none | Round limit. |
///
/// # Reference
/// Yoav Freund.
/// *A More Robust Boosting Algorithm.*
/// arXiv:0905.2138, 2009.
/// <https://arxiv.org/abs/0905.2138>
///
/// # Examples
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
/// let tree = DecisionTreeBuilder::new(&sample).max_depth(2).build();
///
/// let mut booster = RobustBoost::init(&sample)
///     .tolerance(0.2)
///     .theta(0.1)
///     .force_quit_at(200);
/// let f = booster.run(&tree);
/// if booster.remaining_time() == Some(0.0) {
///     // At most a fraction 2ε of the scores y H(x) are at most θ.
///     println!("finished in {} rounds", f.hypotheses.len());
/// } else if let Some(reason) = booster.numerical_stop_reason() {
///     eprintln!("stopped early: {reason}");
/// }
/// assert!(f.weights.iter().all(|w| *w > 0.0));
/// ```
pub struct RobustBoost<'a, F> {
    // Training sample
    sample: &'a Sample,

    // Error goal ε.
    tolerance: f64,

    // Goal margin θ.
    theta: f64,

    // Final width σ_f.
    sigma_f: f64,

    // Optional round limit.
    force_quit_at: Option<usize>,

    // Potential parameters, set by `preprocess`.
    potential: Option<Potential>,

    // Current time t; `None` before training.
    time: Option<f64>,

    // Scores m_j = y_j H(x_j).
    margins: Vec<f64>,

    // Coefficients of H.
    weights: Vec<f64>,

    // Hypotheses obtained by the weak learner.
    hypotheses: Vec<F>,

    // Why training stopped early, if it did.
    numerical_stop_reason: Option<&'static str>,
}

impl<'a, F> RobustBoost<'a, F> {
    /// Constructs RobustBoost for `sample` with `ε = 0.1`, `θ = 0`,
    /// `σ_f = 0.1`, and no round limit.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// See [`RobustBoost`].
    pub fn init(sample: &'a Sample) -> Self {
        Self {
            sample,
            tolerance: 0.1,
            theta: 0.0,
            sigma_f: 0.1,
            force_quit_at: None,
            potential: None,
            time: None,
            margins: Vec::new(),
            weights: Vec::new(),
            hypotheses: Vec::new(),
            numerical_stop_reason: None,
        }
    }

    /// Sets the error goal `ε = Φ(0, 0)`, the initial (and, at `t = 1`,
    /// final) average potential.
    ///
    /// # Panics
    /// Panics unless `0 < tolerance < 1`.
    ///
    /// # Examples
    /// See [`RobustBoost`].
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        assert!(
            tolerance > 0.0 && tolerance < 1.0,
            "RobustBoost tolerance must lie in (0, 1)"
        );
        self.tolerance = tolerance;
        self
    }

    /// Sets the goal margin `θ`.
    ///
    /// # Panics
    /// Panics unless `θ` is finite and nonnegative.
    ///
    /// # Examples
    /// See [`RobustBoost`].
    pub fn theta(mut self, theta: f64) -> Self {
        assert!(
            theta.is_finite() && theta >= 0.0,
            "RobustBoost theta must be finite and nonnegative"
        );
        self.theta = theta;
        self
    }

    /// Sets the final width `σ_f`, the slope of the step in the final
    /// potential. The paper uses `0.1` to avoid numerical instability near
    /// `t = 1`.
    ///
    /// # Panics
    /// Panics unless `σ_f` is finite and positive.
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{Sample, prelude::*};
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// let tree = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    /// let mut booster = RobustBoost::init(&sample).sigma_f(0.2).force_quit_at(20);
    /// let f = booster.run(&tree);
    /// assert!(f.hypotheses.len() <= 20);
    /// ```
    pub fn sigma_f(mut self, sigma_f: f64) -> Self {
        assert!(
            sigma_f.is_finite() && sigma_f > 0.0,
            "RobustBoost sigma_f must be finite and positive"
        );
        self.sigma_f = sigma_f;
        self
    }

    /// Stops after `iteration` rounds if the clock has not reached `1` by
    /// then; the guarantee then does not apply.
    ///
    /// # Examples
    /// See [`RobustBoost`].
    pub fn force_quit_at(mut self, iteration: usize) -> Self {
        self.force_quit_at = Some(iteration);
        self
    }

    /// Returns `1 - t` after the last round, or `None` before training.
    /// `Some(0.0)` means the clock reached `1`, so the guarantee applies.
    ///
    /// # Examples
    /// See [`RobustBoost`].
    pub fn remaining_time(&self) -> Option<f64> {
        self.time.map(|t| 1.0 - t)
    }

    /// Returns why training stopped before the clock reached `1`, other
    /// than the round limit: a weak hypothesis with nonpositive weighted
    /// correlation, or a failure to solve the round. The hypothesis of that
    /// round is not added. Reset by [`Booster::preprocess`].
    ///
    /// # Examples
    /// See [`RobustBoost`].
    pub fn numerical_stop_reason(&self) -> Option<&str> {
        self.numerical_stop_reason
    }
}

impl<F> Booster<F> for RobustBoost<'_, F>
where
    F: Classifier + Clone,
{
    type Output = WeightedMajority<F>;

    fn name(&self) -> &str {
        "RobustBoost"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_examples, n_feature) = self.sample.shape();
        let quit = self
            .force_quit_at
            .map_or("-".to_string(), |it| format!("At round {it}"));
        Some(vec![
            ("# of examples", n_examples.to_string()),
            ("# of features", n_feature.to_string()),
            ("Tolerance", self.tolerance.to_string()),
            ("Theta", self.theta.to_string()),
            ("Sigma_f", self.sigma_f.to_string()),
            ("Force quit", quit),
        ])
    }

    fn preprocess(&mut self) {
        self.sample.is_valid_binary_instance();
        self.potential = Some(Potential::new(self.tolerance, self.theta, self.sigma_f));
        self.time = Some(0.0);
        self.margins = vec![0.0; self.sample.shape().0];
        self.weights = Vec::new();
        self.hypotheses = Vec::new();
        self.numerical_stop_reason = None;
    }

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize) -> ControlFlow<usize>
    where
        W: WeakLearner<Hypothesis = F>,
    {
        let t = self.time.expect("call preprocess before boost");
        let potential = self.potential.expect("call preprocess before boost");
        if t >= 1.0 || self.numerical_stop_reason.is_some() {
            return ControlFlow::Break(iteration - 1);
        }
        if self.force_quit_at.is_some_and(|n| iteration > n) {
            return ControlFlow::Break(iteration - 1);
        }

        let log_weights: Vec<f64> = self
            .margins
            .iter()
            .map(|&m| potential.log_weight(m, t))
            .collect();
        let dist = helpers::softmax(&log_weights);
        let h = weak_learner.produce(self.sample, &dist);
        let steps: Vec<f64> = helpers::margins(self.sample, &h).collect();
        assert!(
            steps.iter().all(|b| (-1.0..=1.0).contains(b)),
            "RobustBoost needs finite confidences in [-1, 1]"
        );

        let round = Round::new(potential, &self.margins, &steps, t);
        let step = match round.solve() {
            Ok(step) => step,
            Err(reason) => {
                self.numerical_stop_reason = Some(reason);
                return ControlFlow::Break(iteration - 1);
            }
        };

        let decay = (-step.delta_t).exp();
        self.margins
            .iter_mut()
            .zip(&steps)
            .for_each(|(m, b)| *m = *m * decay + b * step.delta_m);
        self.weights.iter_mut().for_each(|w| *w *= decay);
        self.weights.push(step.delta_m);
        self.hypotheses.push(h);
        let t = match step.event {
            Event::ClockExpired => 1.0,
            Event::EdgeReached => t + step.delta_t,
        };
        self.time = Some(t);

        if t >= 1.0 {
            ControlFlow::Break(iteration)
        } else {
            ControlFlow::Continue(())
        }
    }

    fn postprocess(&mut self) -> Self::Output {
        WeightedMajority::from_slices(&self.weights, &self.hypotheses)
    }
}

impl<H> CurrentHypothesis for RobustBoost<'_, H>
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
    use std::io::BufReader;

    fn xor_like_sample() -> Sample {
        let csv = b"x,y,class\n\
            -2,1,-1\n-1,-2,-1\n1,-3,-1\n-3,2,-1\n0,-1,-1\n-2,-1,-1\n\
            2,1,1\n1,2,1\n-1,3,1\n3,-1,1\n0,2,1\n2,2,1\n";
        Sample::from_reader(BufReader::new(&csv[..]), true)
            .unwrap()
            .set_target("class")
    }

    fn average_potential<F>(b: &RobustBoost<'_, F>) -> f64 {
        let (p, t) = (b.potential.unwrap(), b.time.unwrap());
        b.margins.iter().map(|&m| p.value(m, t)).sum::<f64>() / b.margins.len() as f64
    }

    #[test]
    fn the_average_potential_stays_at_epsilon() {
        let sample = xor_like_sample();
        let tree = DecisionTreeBuilder::new(&sample).max_depth(2).build();
        let mut booster = RobustBoost::init(&sample).tolerance(0.2).theta(0.1);
        booster.preprocess();
        assert!((average_potential(&booster) - 0.2).abs() < 1e-12);
        for iteration in 1..=200 {
            let flow = booster.boost(&tree, iteration);
            let drift = (average_potential(&booster) - 0.2).abs();
            assert!(drift < 1e-9, "round {iteration}: drift {drift}");
            if flow.is_break() {
                break;
            }
        }
    }

    #[test]
    fn scores_match_the_output_coefficients() {
        let sample = xor_like_sample();
        let tree = DecisionTreeBuilder::new(&sample).max_depth(2).build();
        let mut booster = RobustBoost::init(&sample).tolerance(0.2).force_quit_at(30);
        let f = booster.run(&tree);
        let total: f64 = booster.weights.iter().sum();
        for (j, m) in booster.margins.iter().enumerate() {
            let expected = sample.target()[j] * f.confidence(&sample, j) * total;
            assert!((m - expected).abs() < 1e-10, "{j}: {m} vs {expected}");
        }
    }

    #[test]
    fn reaching_t_one_bounds_the_scores_below_theta() {
        let sample = xor_like_sample();
        let tree = DecisionTreeBuilder::new(&sample).max_depth(2).build();
        let (epsilon, theta) = (0.2, 0.1);
        let mut booster = RobustBoost::init(&sample)
            .tolerance(epsilon)
            .theta(theta)
            .force_quit_at(500);
        booster.run(&tree);
        assert_eq!(booster.numerical_stop_reason(), None);
        assert_eq!(booster.remaining_time(), Some(0.0));
        let low = booster.margins.iter().filter(|&&m| m <= theta).count();
        assert!(low as f64 / 12.0 <= 2.0 * epsilon);
    }

    #[test]
    fn invalid_parameters_panic() {
        let sample = xor_like_sample();
        let init = || RobustBoost::<DecisionTreeClassifier>::init(&sample);
        assert!(std::panic::catch_unwind(|| init().tolerance(0.0)).is_err());
        assert!(std::panic::catch_unwind(|| init().theta(-0.1)).is_err());
        assert!(std::panic::catch_unwind(|| init().sigma_f(0.0)).is_err());
    }
}
