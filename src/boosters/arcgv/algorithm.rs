//! Implementation of arc-gv (Breiman, 1999) as [`ArcGv`].
//! See the [`ArcGv`] struct for details and an example.
use crate::core::CurrentHypothesis;
use crate::core::{Booster, Classifier, Sample, WeakLearner, tools::helpers};
use crate::hypotheses::WeightedMajority;

use std::ops::ControlFlow;

/// Arc-gv ("gv" for game value) by Breiman: an arcing algorithm whose
/// largest weighted training error `top(c)` converges to the value of the
/// game, i.e. whose minimal margin converges to the best achievable one.
///
/// # Algorithm
/// Breiman writes the ensemble as nonnegative coefficients `b` on the
/// hypotheses, `c = b / |b|`, and measures losses instead of margins: the
/// loss of hypothesis `h` on example `z = (x, y)` is
/// `l_h(z) = (1 - y h(x)) / 2 ∈ [0, 1]` (the 0-1 loss for `h(x) = ±1`), the
/// ensemble loss is `er(z, b) = Σ_m b_m l_{h_m}(z)`, and
/// `top(c) = max_n er(z_n, c)`. Starting from `b = 0`, round `k` follows
/// Definition 5.1 of the paper:
/// 1. With `t_k = top(c^{(k)})`, give example `n` the weight
///    `Q_k(z_n) ∝ exp(er(z_n, b^{(k)}))` and call the weak learner to obtain
///    `h_k`.
/// 2. Let `Δ_k` minimize `E_{Q_k}[exp(Δ (l_{h_k}(z) - t_k))]` over `[0, 1]`.
/// 3. If `Δ_k = 0`, stop; otherwise add `Δ_k` to the coefficient of `h_k`.
///
/// The output is the [`WeightedMajority`] with weights `c = b / |b|`. In
/// margin terms, `1 - 2 er(z, c) = y Σ_m c_m h_m(x)`, so `t_k` corresponds
/// to the minimal margin `ρ_k = 1 - 2 t_k`, and for `h(x) = ±1` with weighted
/// error `e_k` the minimizer is
/// `Δ_k = ln[(1 - e_k) t_k / (e_k (1 - t_k))] = 2 (atanh γ_k - atanh ρ_k)`
/// (clipped to `[0, 1]`), with edge `γ_k = 1 - 2 e_k`.
///
/// Implementation choices (not part of the original method):
/// - `t_1 = top(c^{(1)})` is undefined for `b = 0`; this implementation uses
///   `t_1 = 1/2` (minimal margin `0`), so the first step is AdaBoost's
///   capped at `1`.
/// - For confidence-rated hypotheses, `l_h(z) = (1 - y h(x)) / 2` is not
///   binary, and `Δ_k` is found by bisection on the derivative of the
///   convex objective instead of the closed form.
/// - Breiman's weak learner returns the hypothesis minimizing `Q_k`-weighted
///   loss (`m_{k+1} = argmin_m Q_k(e_m)`); here, any weak learner may be
///   used.
///
/// # Theoretical guarantee
/// Theorem 5.2 of the paper: if arc-gv stops at step `k`, then
/// `top(c^{(k)}) = φ*`; otherwise `top(c^{(k)}) → φ*`, where `φ*` is the value
/// of the game (equivalently, the minimal margin converges to the best
/// achievable `ρ* = 1 - 2 φ*`). This assumes that each `h_k` minimizes the
/// weighted loss over the hypothesis class.
///
/// # Limitations
/// - The guarantee is asymptotic and gives no convergence rate, so training
///   needs an explicit round limit: set [`ArcGv::force_quit_at`] (training
///   panics otherwise).
/// - With a weak learner that does not minimize the weighted loss, a stop
///   (`Δ_k = 0`) only means that this weak hypothesis does not beat the
///   current minimal margin; see [`ArcGv::reached_zero_step`].
/// - Breiman's experiments (Section 7) found that arc-gv attains larger
///   minimal margins than AdaBoost but often a higher test error.
/// - Confidences must lie in `[-1, 1]`; labels must be `-1` or `+1` and both
///   classes must be present.
///
/// # Parameters
/// | Builder method | Default | Meaning |
/// | --- | --- | --- |
/// | [`force_quit_at`](ArcGv::force_quit_at) | required | Round limit. |
///
/// # Reference
/// Leo Breiman.
/// *Prediction Games and Arcing Algorithms.*
/// Neural Computation, 11(7):1493–1517, 1999.
/// <https://doi.org/10.1162/089976699300016106>
/// (Technical Report 504, Statistics Department, UC Berkeley, 1997.)
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
/// let tree = DecisionTreeBuilder::new(&sample)
///     .max_depth(1)
///     .split_by(SplitBy::Edge)
///     .build();
///
/// let mut booster = ArcGv::init(&sample).force_quit_at(100);
/// let f = booster.run(&tree);
/// // The minimal margin of the normalized combination.
/// let rho = (0..12)
///     .map(|i| sample.target()[i] * f.confidence(&sample, i))
///     .fold(f64::INFINITY, f64::min);
/// assert!((-1.0..=1.0).contains(&rho));
/// assert!(f.weights.iter().all(|w| *w > 0.0));
/// ```
pub struct ArcGv<'a, F> {
    // Training sample
    sample: &'a Sample,

    // Required round limit.
    force_quit_at: Option<usize>,

    // er(z_n, b) = Σ_m b_m l_m(z_n).
    losses: Vec<f64>,

    // |b| = Σ_m b_m.
    total: f64,

    // Coefficients `b_m`.
    weights: Vec<f64>,

    // Hypotheses obtained by the weak learner.
    hypotheses: Vec<F>,

    // Whether the last run stopped with Δ_k = 0.
    zero_step: bool,
}

impl<'a, F> ArcGv<'a, F> {
    /// Constructs arc-gv for `sample`. Set the round limit with
    /// [`ArcGv::force_quit_at`] before training.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// See [`ArcGv`].
    pub fn init(sample: &'a Sample) -> Self {
        Self {
            sample,
            force_quit_at: None,
            losses: Vec::new(),
            total: 0.0,
            weights: Vec::new(),
            hypotheses: Vec::new(),
            zero_step: false,
        }
    }

    /// Stops after `iteration` rounds unless `Δ_k = 0` stops training
    /// earlier. Required, because the convergence guarantee is asymptotic.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// See [`ArcGv`].
    pub fn force_quit_at(mut self, iteration: usize) -> Self {
        self.force_quit_at = Some(iteration);
        self
    }

    /// Returns whether the last run stopped because `Δ_k = 0`, i.e. the
    /// weak hypothesis's weighted loss was at least `t_k` (its edge at most
    /// the current minimal margin). The hypothesis of that round is not
    /// added. Theorem 5.2 then gives `top(c) = φ*` if the weak learner
    /// minimizes the weighted loss over the class.
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
    /// let tree = DecisionTreeBuilder::new(&sample).max_depth(1).build();
    /// let mut booster = ArcGv::init(&sample).force_quit_at(10);
    /// let f = booster.run(&tree);
    /// // The separating stump has margin 1 everywhere, so the next stump
    /// // (the same one) cannot improve it: Δ = 0 in round 2.
    /// assert!(booster.reached_zero_step());
    /// assert_eq!(f.hypotheses.len(), 1);
    /// ```
    pub fn reached_zero_step(&self) -> bool {
        self.zero_step
    }
}

/// Returns the minimizer over `[0, 1]` of
/// `φ(Δ) = Σ_n q_n exp(Δ (l_n - t))`, a convex function of `Δ`.
fn step_size(q: &[f64], losses: &[f64], t: f64) -> f64 {
    let derivative = |delta: f64| -> f64 {
        q.iter()
            .zip(losses)
            .map(|(q, l)| q * (l - t) * (delta * (l - t)).exp())
            .sum()
    };
    if derivative(0.0) >= 0.0 {
        return 0.0;
    }
    if derivative(1.0) <= 0.0 {
        return 1.0;
    }
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if derivative(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

impl<F> Booster<F> for ArcGv<'_, F>
where
    F: Classifier + Clone,
{
    type Output = WeightedMajority<F>;

    fn name(&self) -> &str {
        "Arc-gv"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_examples, n_feature) = self.sample.shape();
        let quit = self
            .force_quit_at
            .map_or("-".to_string(), |it| format!("At round {it}"));
        Some(vec![
            ("# of examples", n_examples.to_string()),
            ("# of features", n_feature.to_string()),
            ("Force quit", quit),
        ])
    }

    fn preprocess(&mut self) {
        assert!(
            self.force_quit_at.is_some(),
            "arc-gv needs a round limit; set it with `force_quit_at`"
        );
        self.sample.is_valid_binary_instance();
        self.losses = vec![0.0; self.sample.shape().0];
        self.total = 0.0;
        self.weights = Vec::new();
        self.hypotheses = Vec::new();
        self.zero_step = false;
    }

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize) -> ControlFlow<usize>
    where
        W: WeakLearner<Hypothesis = F>,
    {
        let limit = self.force_quit_at.expect("call preprocess before boost");
        if iteration > limit {
            return ControlFlow::Break(limit);
        }

        // Q_k(z_n) ∝ exp(er(z_n, b)), normalized in log space.
        let q = helpers::softmax(&self.losses);
        let h = weak_learner.produce(self.sample, &q);
        let losses: Vec<f64> = helpers::margins(self.sample, &h)
            .map(|m| {
                assert!(
                    (-1.0..=1.0).contains(&m),
                    "arc-gv needs finite confidences in [-1, 1]"
                );
                0.5 * (1.0 - m)
            })
            .collect();

        // t_k = top(c^{(k)}); t_1 = 1/2 by convention (see the docs).
        let t = if self.total > 0.0 {
            self.losses
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
                / self.total
        } else {
            0.5
        };
        let delta = step_size(&q, &losses, t);
        if delta == 0.0 {
            self.zero_step = true;
            return ControlFlow::Break(iteration - 1);
        }

        self.losses
            .iter_mut()
            .zip(&losses)
            .for_each(|(er, l)| *er += delta * l);
        self.total += delta;
        self.weights.push(delta);
        self.hypotheses.push(h);
        ControlFlow::Continue(())
    }

    fn postprocess(&mut self) -> Self::Output {
        WeightedMajority::from_slices(&self.weights, &self.hypotheses)
    }
}

impl<H> CurrentHypothesis for ArcGv<'_, H>
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
    use crate::weak_learners::finite_class::FiniteClass;
    use crate::{DecisionTreeClassifier, soft_margin_optimization};

    #[test]
    fn step_size_matches_the_closed_form_for_binary_losses() {
        // Weighted error e = 0.3, t = 0.4: Δ = ln((1 - e) t / (e (1 - t))).
        let q = [0.1, 0.2, 0.3, 0.4];
        let losses = [1.0, 1.0, 0.0, 0.0]; // e = 0.1 + 0.2
        let expected = ((0.7_f64 * 0.4) / (0.3 * 0.6)).ln();
        let delta = step_size(&q, &losses, 0.4);
        assert!((delta - expected).abs() < 1e-12, "{delta} vs {expected}");
        // Clipping: e = 0.3 > t gives 0; a tiny error gives 1.
        assert_eq!(step_size(&q, &losses, 0.25), 0.0);
        assert_eq!(step_size(&[0.999, 0.001], &[0.0, 1.0], 0.4), 1.0);
    }

    #[test]
    fn the_minimal_margin_approaches_the_game_value() {
        // A finite class with value ρ* = 1/3 (each hypothesis misclassifies
        // one of examples 1, 2, 3); FiniteClass returns the weighted-loss
        // minimizer, as Theorem 5.2 assumes.
        let sample = Sample::dummy(4);
        let y = sample.target().to_vec();
        let margins = [
            [1.0, -1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0, -1.0],
        ];
        let table = margins
            .iter()
            .map(|m| m.iter().zip(&y).map(|(u, yi)| u * yi).collect())
            .collect();
        let learner = FiniteClass::from_table(&sample, table);
        let (rho_star, _) = soft_margin_optimization(1.0, &sample, learner.hypotheses());
        let f = ArcGv::init(&sample).force_quit_at(2000).run(&learner);
        let rho = (0..4)
            .map(|i| y[i] * f.confidence(&sample, i))
            .fold(f64::INFINITY, f64::min);
        // ρ* - ρ ≥ 0 always; 2000 rounds bring it within 0.02 here.
        assert!(rho <= rho_star + 1e-6);
        assert!(rho_star - rho < 0.02, "ρ* = {rho_star}, ρ = {rho}");
    }

    #[test]
    #[should_panic(expected = "round limit")]
    fn training_without_a_round_limit_panics() {
        let sample = Sample::dummy(2);
        ArcGv::<DecisionTreeClassifier>::init(&sample).preprocess();
    }
}
