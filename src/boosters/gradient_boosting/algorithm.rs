//! Implementation of gradient boosting (Friedman, 2001) for binary
//! classification as [`GradientBoosting`].
//! See the [`GradientBoosting`] struct for details and an example.
use crate::core::CurrentHypothesis;
use crate::core::{Booster, Regressor, Sample, WeakLearner};
use crate::hypotheses::{AdditiveModel, ScoreLink};

use std::ops::ControlFlow;

/// [`GradientBoosting::numerical_stop_reason`] when the line search has no
/// minimizer.
const NO_MINIMIZER: &str =
    "the line search has no minimizer: the loss decreases for every step size along h";

/// The loss `L(y, F)` minimized by [`GradientBoosting`], for labels
/// `y ∈ {-1, +1}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GbmLoss {
    /// Binomial deviance `ln(1 + exp(-2yF))`, with `F` half the log-odds
    /// (Friedman, 2001, Section 4.5). Confidences are `tanh(F)`.
    Logistic,
    /// Exponential loss `exp(-yF)`, AdaBoost's loss (Friedman, Hastie, and
    /// Tibshirani, 2000). Confidences are `tanh(F)`.
    Exponential,
    /// Squared error `(y - F)² / 2` (Friedman, 2001, LS_Boost).
    /// Confidences are `F` clamped to `[-1, 1]`.
    Squared,
}

impl GbmLoss {
    fn value(self, y: f64, f: f64) -> f64 {
        match self {
            Self::Logistic => softplus(-2.0 * y * f),
            Self::Exponential => (-y * f).exp(),
            Self::Squared => 0.5 * (y - f).powi(2),
        }
    }

    /// `∂L/∂F`.
    fn derivative(self, y: f64, f: f64) -> f64 {
        match self {
            Self::Logistic => -2.0 * y * logistic(-2.0 * y * f),
            Self::Exponential => -y * (-y * f).exp(),
            Self::Squared => f - y,
        }
    }

    /// `argmin_c Σ_i L(y_i, c)` for labels with mean `ȳ ∈ (-1, 1)`.
    fn initial_score(self, mean: f64) -> f64 {
        match self {
            Self::Logistic | Self::Exponential => 0.5 * ((1.0 + mean) / (1.0 - mean)).ln(),
            Self::Squared => mean,
        }
    }

    fn link(self) -> ScoreLink {
        match self {
            Self::Logistic | Self::Exponential => ScoreLink::Tanh,
            Self::Squared => ScoreLink::Clamp,
        }
    }
}

fn logistic(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn softplus(u: f64) -> f64 {
    u.max(0.0) + (-u.abs()).exp().ln_1p()
}

/// Gradient boosting (Friedman's `Gradient_Boost`) for binary
/// classification with a regression weak learner.
///
/// # Algorithm
/// Algorithm 1 of the paper with shrinkage (Section 5). Starting from the
/// constant `F_0 = argmin_c Σ_i L(y_i, c)`, round `t`
/// 1. computes the pseudo-residuals `ỹ_i = -∂L(y_i, F)/∂F` at
///    `F = F_{t-1}(x_i)`;
/// 2. calls the weak learner on the sample with targets `ỹ` (see
///    [`Sample::with_target`]) and uniform weights, i.e. a least-squares fit
///    `h_t`;
/// 3. finds `ρ_t = argmin_ρ Σ_i L(y_i, F_{t-1}(x_i) + ρ h_t(x_i))` (line
///    search);
/// 4. sets `F_t = F_{t-1} + ν ρ_t h_t` with the shrinkage `ν ∈ (0, 1]`.
///
/// The output is an [`AdditiveModel`]; its confidence is `tanh(F)` for the
/// logistic and exponential losses (the class-probability difference when
/// `F` is half the log-odds) and `F` clamped to `[-1, 1]` for the squared
/// loss. See [`GbmLoss`] for the losses.
///
/// Implementation choices (not part of the original method):
/// - The line search is a bisection on the derivative of the convex
///   one-dimensional objective. If `h_t` is zero on the sample (`ρ_t` is
///   undefined), the objective is not finite, or it has no minimizer,
///   training stops without adding `h_t`; see
///   [`GradientBoosting::numerical_stop_reason`]. For the logistic and
///   exponential losses, the minimizer exists exactly when some example's
///   margin `y_i h_t(x_i)` has the sign opposite to the descent direction;
///   this is checked before the search, rather than relying on the
///   derivative, which underflows to `0` for large steps.
/// - The terminal-node updates of `TreeBoost` (one `ρ` per leaf, Section 4.3)
///   are not implemented, since boosters here do not depend on a concrete
///   weak learner; with a regression tree this is plain `Gradient_Boost`.
///
/// # Theoretical guarantee
/// None stated in the paper beyond being a greedy (steepest-descent) method
/// for the training loss: with exact line search, each round does not
/// increase `Σ_i L(y_i, F_t(x_i))` when `ν = 1`.
///
/// # Limitations
/// - The number of rounds is not set by a stopping rule: set
///   [`GradientBoosting::force_quit_at`] (training panics otherwise).
/// - The weak learner must accept real-valued targets (for example
///   [`RegressionTree`](crate::weak_learners::regression_tree::RegressionTree));
///   a classification learner would ignore the residual values.
/// - Each round copies the features (`O(m d)`) to attach the residuals.
/// - With the logistic or exponential loss, a weak hypothesis that orders
///   every training example correctly (`y_i h_t(x_i) ≥ 0` for all `i`, as a
///   deep tree often does on small or separable samples) leaves the line
///   search without a minimizer, so training stops, possibly in the first
///   round with only the constant `F_0`. Use a weaker learner (for example a
///   shallower tree) or the squared loss in that case.
/// - Labels must be `-1` or `+1` and both classes must be present.
///
/// # Parameters
/// | Builder method | Default | Meaning |
/// | --- | --- | --- |
/// | [`loss`](GradientBoosting::loss) | [`GbmLoss::Logistic`] | The loss `L`. |
/// | [`shrinkage`](GradientBoosting::shrinkage) | `1.0` | `ν`; Friedman recommends small values such as `0.1`. |
/// | [`force_quit_at`](GradientBoosting::force_quit_at) | required | Number of rounds. |
///
/// # Reference
/// Jerome H. Friedman.
/// *Greedy Function Approximation: A Gradient Boosting Machine.*
/// The Annals of Statistics, 29(5):1189–1232, 2001.
/// <https://doi.org/10.1214/aos/1013203451>
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
/// let tree = RegressionTreeBuilder::new(&sample).max_depth(2).build();
///
/// let f = GradientBoosting::init(&sample)
///     .shrinkage(0.5)
///     .force_quit_at(50)
///     .run(&tree);
/// assert_eq!(f.hypotheses.len(), 50);
/// assert!(f.confidence_all(&sample).iter().all(|c| (-1.0..=1.0).contains(c)));
/// ```
pub struct GradientBoosting<'a, H> {
    sample: &'a Sample,
    loss: GbmLoss,
    shrinkage: f64,
    force_quit_at: Option<usize>,
    // Scores F_{t-1}(x_i) on the training sample.
    scores: Vec<f64>,
    model: Option<AdditiveModel<H>>,
    numerical_stop_reason: Option<&'static str>,
}

impl<'a, H> GradientBoosting<'a, H> {
    /// Constructs gradient boosting for `sample` with the logistic loss and
    /// no shrinkage. Set the number of rounds with
    /// [`GradientBoosting::force_quit_at`] before training.
    ///
    /// # Examples
    /// See [`GradientBoosting`].
    pub fn init(sample: &'a Sample) -> Self {
        Self {
            sample,
            loss: GbmLoss::Logistic,
            shrinkage: 1.0,
            force_quit_at: None,
            scores: Vec::new(),
            model: None,
            numerical_stop_reason: None,
        }
    }

    /// Sets the loss.
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{Sample, prelude::*};
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// use miniboosts::GbmLoss;
    ///
    /// let tree = RegressionTreeBuilder::new(&sample).max_depth(1).build();
    /// let f = GradientBoosting::init(&sample)
    ///     .loss(GbmLoss::Squared)
    ///     .force_quit_at(5)
    ///     .run(&tree);
    /// // Least squares on ±1 labels: the stump fits them exactly.
    /// assert_eq!(f.predict_all(&sample), vec![-1, -1, -1, 1, 1, 1]);
    /// ```
    pub fn loss(mut self, loss: GbmLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Sets the shrinkage `ν`, the factor applied to each step.
    ///
    /// # Panics
    /// Panics unless `0 < shrinkage ≤ 1`.
    ///
    /// # Examples
    /// See [`GradientBoosting`].
    pub fn shrinkage(mut self, shrinkage: f64) -> Self {
        assert!(
            shrinkage > 0.0 && shrinkage <= 1.0,
            "shrinkage must lie in (0, 1]"
        );
        self.shrinkage = shrinkage;
        self
    }

    /// Sets the number of rounds. Required.
    ///
    /// # Examples
    /// See [`GradientBoosting`].
    pub fn force_quit_at(mut self, iteration: usize) -> Self {
        self.force_quit_at = Some(iteration);
        self
    }

    /// Returns why training stopped before the round limit, if it did: a
    /// weak hypothesis that is zero on the sample, or a non-finite line
    /// search. Reset by [`Booster::preprocess`].
    ///
    /// # Examples
    /// See [`GradientBoosting`].
    pub fn numerical_stop_reason(&self) -> Option<&str> {
        self.numerical_stop_reason
    }

    /// `ρ = argmin_ρ Σ_i L(y_i, F_i + ρ h_i)` by bisection on the derivative.
    fn line_search(&self, h: &[f64]) -> Result<f64, &'static str> {
        let target = self.sample.target();
        let derivative = |rho: f64| -> f64 {
            self.scores
                .iter()
                .zip(h)
                .zip(target)
                .map(|((f, h), y)| self.loss.derivative(*y, f + rho * h) * h)
                .sum()
        };
        if h.iter().all(|v| *v == 0.0) {
            return Err("the weak hypothesis is zero on the training sample");
        }
        let at_zero = derivative(0.0);
        if !at_zero.is_finite() {
            return Err("the line search derivative is not finite");
        }
        if at_zero == 0.0 {
            return Ok(0.0);
        }
        // Bracket the root on the descent side.
        let direction = if at_zero < 0.0 { 1.0 } else { -1.0 };
        // The logistic and exponential losses decrease strictly in the margin
        // yF towards an infimum they never attain. If no example's margin
        // decreases along the descent direction (direction · y_i h_i ≥ 0 for
        // all i), the objective decreases for every step and has no
        // minimizer; otherwise it tends to infinity and has one. Checking
        // this exactly avoids returning a step where the derivative merely
        // underflows to 0. The squared loss is a quadratic with curvature
        // Σ h_i² > 0, so its minimizer always exists.
        if self.loss != GbmLoss::Squared
            && h.iter().zip(target).all(|(h, y)| direction * y * h >= 0.0)
        {
            return Err(NO_MINIMIZER);
        }
        let (mut lo, mut hi) = (0.0, direction);
        let mut steps = 0;
        while derivative(hi) * direction < 0.0 {
            lo = hi;
            hi *= 2.0;
            steps += 1;
            if steps > 1100 || !hi.is_finite() {
                return Err(NO_MINIMIZER);
            }
        }
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if mid == lo || mid == hi {
                break;
            }
            if derivative(mid) * direction < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Ok(0.5 * (lo + hi))
    }

    /// The training loss `(1/m) Σ_i L(y_i, F(x_i))` of the current scores.
    fn training_loss(&self) -> f64 {
        let target = self.sample.target();
        self.scores
            .iter()
            .zip(target)
            .map(|(f, y)| self.loss.value(*y, *f))
            .sum::<f64>()
            / target.len() as f64
    }
}

impl<H> Booster<H> for GradientBoosting<'_, H>
where
    H: Regressor + Clone,
{
    type Output = AdditiveModel<H>;

    fn name(&self) -> &str {
        "Gradient boosting"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_examples, n_feature) = self.sample.shape();
        let quit = self
            .force_quit_at
            .map_or("-".to_string(), |it| it.to_string());
        Some(vec![
            ("# of examples", n_examples.to_string()),
            ("# of features", n_feature.to_string()),
            ("Loss", format!("{:?}", self.loss)),
            ("Shrinkage", self.shrinkage.to_string()),
            ("Rounds", quit),
        ])
    }

    fn preprocess(&mut self) {
        assert!(
            self.force_quit_at.is_some(),
            "gradient boosting needs a number of rounds; set it with `force_quit_at`"
        );
        self.sample.is_valid_binary_instance();
        let target = self.sample.target();
        let mean = target.iter().sum::<f64>() / target.len() as f64;
        let intercept = self.loss.initial_score(mean);
        self.scores = vec![intercept; target.len()];
        self.model = Some(AdditiveModel::new(intercept, self.loss.link()));
        self.numerical_stop_reason = None;
    }

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize) -> ControlFlow<usize>
    where
        W: WeakLearner<Hypothesis = H>,
    {
        let limit = self.force_quit_at.expect("call preprocess before boost");
        if iteration > limit || self.numerical_stop_reason.is_some() {
            return ControlFlow::Break(iteration - 1);
        }
        let target = self.sample.target();
        let residuals: Vec<f64> = self
            .scores
            .iter()
            .zip(target)
            .map(|(f, y)| -self.loss.derivative(*y, *f))
            .collect();
        let m = target.len();
        let h = weak_learner.produce(
            &self.sample.with_target(residuals),
            &vec![1.0 / m as f64; m],
        );
        let values: Vec<f64> = (0..m).map(|i| h.predict(self.sample, i)).collect();
        assert!(
            values.iter().all(|v| v.is_finite()),
            "gradient boosting needs finite weak-hypothesis predictions"
        );
        let rho = match self.line_search(&values) {
            Ok(rho) => rho,
            Err(reason) => {
                self.numerical_stop_reason = Some(reason);
                return ControlFlow::Break(iteration - 1);
            }
        };
        let step = self.shrinkage * rho;
        self.scores
            .iter_mut()
            .zip(&values)
            .for_each(|(f, v)| *f += step * v);
        debug_assert!(self.training_loss().is_finite());
        self.model
            .as_mut()
            .expect("call preprocess before boost")
            .push(step, h);
        ControlFlow::Continue(())
    }

    fn postprocess(&mut self) -> Self::Output {
        self.model
            .clone()
            .expect("call preprocess before postprocess")
    }
}

impl<H> CurrentHypothesis for GradientBoosting<'_, H>
where
    H: Regressor + Clone,
{
    type Output = AdditiveModel<H>;
    fn current_hypothesis(&self) -> Self::Output {
        self.model
            .clone()
            .unwrap_or_else(|| AdditiveModel::new(0.0, self.loss.link()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weak_learners::regression_tree::{RegressionTreeBuilder, RegressionTreeRegressor};
    use std::io::BufReader;

    fn xor_like_sample() -> Sample {
        let csv = b"x,y,class\n\
            -2,1,-1\n-1,-2,-1\n1,-3,-1\n-3,2,-1\n0,-1,-1\n-2,-1,-1\n\
            2,1,1\n1,2,1\n-1,3,1\n3,-1,1\n0,2,1\n2,2,1\n";
        Sample::from_reader(BufReader::new(&csv[..]), true)
            .unwrap()
            .set_target("class")
    }

    #[test]
    fn derivatives_match_finite_differences() {
        for loss in [GbmLoss::Logistic, GbmLoss::Exponential, GbmLoss::Squared] {
            for (y, f) in [(1.0, 0.3), (-1.0, 0.3), (1.0, -2.0), (-1.0, 1.5)] {
                let h = 1e-6;
                let numeric = (loss.value(y, f + h) - loss.value(y, f - h)) / (2.0 * h);
                assert!((numeric - loss.derivative(y, f)).abs() < 1e-7, "{loss:?}");
            }
        }
        // F_0 minimizes Σ L(y_i, c): its derivative vanishes.
        let y = [1.0, 1.0, 1.0, -1.0];
        for loss in [GbmLoss::Logistic, GbmLoss::Exponential, GbmLoss::Squared] {
            let c = loss.initial_score(0.5);
            let d: f64 = y.iter().map(|yi| loss.derivative(*yi, c)).sum();
            assert!(d.abs() < 1e-12, "{loss:?}: {d}");
        }
    }

    #[test]
    fn the_training_loss_decreases_every_round() {
        let sample = xor_like_sample();
        let tree = RegressionTreeBuilder::new(&sample).max_depth(1).build();
        for loss in [GbmLoss::Logistic, GbmLoss::Exponential, GbmLoss::Squared] {
            let mut booster = GradientBoosting::<RegressionTreeRegressor>::init(&sample)
                .loss(loss)
                .force_quit_at(30);
            booster.preprocess();
            let mut previous = booster.training_loss();
            for iteration in 1..=30 {
                if booster.boost(&tree, iteration).is_break() {
                    break;
                }
                let current = booster.training_loss();
                assert!(current <= previous + 1e-12, "{loss:?}, round {iteration}");
                previous = current;
            }
        }
    }

    #[test]
    fn the_line_search_is_exact() {
        // After the step, Σ_i ∂L/∂F · h_i = 0 (with ν = 1).
        let sample = xor_like_sample();
        let tree = RegressionTreeBuilder::new(&sample).max_depth(2).build();
        let mut booster =
            GradientBoosting::<RegressionTreeRegressor>::init(&sample).force_quit_at(1);
        booster.preprocess();
        assert!(booster.boost(&tree, 1).is_continue());
        let h = &booster.model.as_ref().unwrap().hypotheses[0];
        let target = sample.target();
        let d: f64 = (0..12)
            .map(|i| {
                GbmLoss::Logistic.derivative(target[i], booster.scores[i]) * h.predict(&sample, i)
            })
            .sum();
        assert!(d.abs() < 1e-10, "{d}");
    }

    #[test]
    fn a_separating_hypothesis_has_no_line_search_minimizer() {
        // The residuals at F_0 = 0 are positive multiples of y, so the
        // depth-1 tree fits h with y_i h_i > 0 for every i. For the logistic
        // and exponential losses, L(y_i, ρ h_i) then decreases in ρ for all
        // i, and the infimum (as ρ → ∞) is not attained.
        let rows: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64]).collect();
        let sample = Sample::from_dense(&rows, Some(&[-1.0, -1.0, 1.0, 1.0])).unwrap();
        let tree = RegressionTreeBuilder::new(&sample).max_depth(1).build();
        for loss in [GbmLoss::Logistic, GbmLoss::Exponential] {
            let mut booster = GradientBoosting::<RegressionTreeRegressor>::init(&sample)
                .loss(loss)
                .force_quit_at(5);
            let f = booster.run(&tree);
            assert!(f.hypotheses.is_empty(), "{loss:?}");
            assert_eq!(
                booster.numerical_stop_reason(),
                Some(NO_MINIMIZER),
                "{loss:?}"
            );
        }
        // The squared loss is a quadratic in ρ: its minimizer always exists.
        let mut booster = GradientBoosting::<RegressionTreeRegressor>::init(&sample)
            .loss(GbmLoss::Squared)
            .force_quit_at(1);
        assert_eq!(booster.run(&tree).coefficients, vec![1.0]);
        assert_eq!(booster.numerical_stop_reason(), None);
    }

    #[test]
    #[should_panic(expected = "number of rounds")]
    fn missing_round_limit_panics() {
        let sample = xor_like_sample();
        GradientBoosting::<RegressionTreeRegressor>::init(&sample).preprocess();
    }
}
