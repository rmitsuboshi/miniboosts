//! Implementation of LogitBoost (Friedman, Hastie, and Tibshirani, 2000) for
//! two classes as [`LogitBoost`].
//! See the [`LogitBoost`] struct for details and an example.
use crate::core::CurrentHypothesis;
use crate::core::{Booster, Regressor, Sample, WeakLearner};
use crate::hypotheses::{AdditiveModel, ScoreLink};

use std::ops::ControlFlow;

/// LogitBoost (two classes) by Friedman, Hastie, and Tibshirani: Newton
/// steps on the binomial log-likelihood, each fitted by weighted least
/// squares.
///
/// # Algorithm
/// Algorithm 3 of the paper, with `y* = (y + 1)/2 ∈ {0, 1}`. Starting from
/// `F(x) = 0` and `p(x) = 1/2`, round `t`
/// 1. computes the working response and weights
///    ```txt
///    z_i = (y*_i - p(x_i)) / (p(x_i)(1 - p(x_i))),   w_i = p(x_i)(1 - p(x_i));
///    ```
/// 2. fits `f_t` by weighted least squares of `z` on `x` with weights `w`,
///    i.e. calls the weak learner on the sample with targets `z` (see
///    [`Sample::with_target`]) and the normalized weights;
/// 3. updates `F(x) ← F(x) + f_t(x)/2` and
///    `p(x) = e^{F(x)} / (e^{F(x)} + e^{-F(x)})`.
///
/// The output is the [`AdditiveModel`] `F = Σ_t f_t / 2`, whose confidence
/// is `tanh(F) = 2p - 1 = P(+1|x) - P(-1|x)`.
///
/// Numerical choices: the paper recommends limiting the working response
/// for numerical stability, with a bound `z_max` between `2` and `4`; it is
/// clipped to `[-z_max, z_max]` here (default `4`). With `p = σ(2F)`,
/// `z = 1 + e^{-2F}` for `y = +1` and `z = -(1 + e^{2F})` for `y = -1`, and
/// `w = σ(2F) σ(-2F)`, computed without overflow. If every weight
/// underflows to `0`, training stops; see
/// [`LogitBoost::numerical_stop_reason`].
///
/// # Theoretical guarantee
/// Each round is a Newton step for the expected log-likelihood
/// `E[log(1 + e^{-2yF})]` (Section 4 of the paper), when the weak learner
/// fits the weighted least-squares problem exactly over its class. No
/// convergence rate or training-error bound is stated.
///
/// # Limitations
/// - The number of rounds is not set by a stopping rule: set
///   [`LogitBoost::force_quit_at`] (training panics otherwise).
/// - The weak learner must accept real-valued targets and weights (for
///   example [`RegressionTree`](crate::weak_learners::regression_tree::RegressionTree)).
/// - On separable data the scores grow without bound; the clipping keeps
///   the working response finite, and the weights of well-classified
///   examples vanish.
/// - Labels must be `-1` or `+1` and both classes must be present.
///
/// # Parameters
/// | Builder method | Default | Meaning |
/// | --- | --- | --- |
/// | [`max_response`](LogitBoost::max_response) | `4.0` | `z_max`, the clipping bound of the working response. |
/// | [`force_quit_at`](LogitBoost::force_quit_at) | required | Number of rounds. |
///
/// # Reference
/// Jerome Friedman, Trevor Hastie, and Robert Tibshirani.
/// *Additive Logistic Regression: A Statistical View of Boosting.*
/// The Annals of Statistics, 28(2):337–407, 2000.
/// <https://doi.org/10.1214/aos/1016218223>
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
/// let tree = RegressionTreeBuilder::new(&sample).max_depth(1).build();
///
/// let f = LogitBoost::init(&sample).force_quit_at(40).run(&tree);
/// let correct = f
///     .predict_all(&sample)
///     .iter()
///     .zip(sample.target())
///     .filter(|&(&p, &y)| p as f64 == y)
///     .count();
/// println!("{correct} of 12 training examples correct");
/// assert!(f.confidence_all(&sample).iter().all(|c| (-1.0..=1.0).contains(c)));
/// ```
pub struct LogitBoost<'a, H> {
    sample: &'a Sample,
    max_response: f64,
    force_quit_at: Option<usize>,
    // Scores F(x_i) on the training sample.
    scores: Vec<f64>,
    model: Option<AdditiveModel<H>>,
    numerical_stop_reason: Option<&'static str>,
}

impl<'a, H> LogitBoost<'a, H> {
    /// Constructs LogitBoost for `sample` with `z_max = 4`. Set the number
    /// of rounds with [`LogitBoost::force_quit_at`] before training.
    ///
    /// # Examples
    /// See [`LogitBoost`].
    pub fn init(sample: &'a Sample) -> Self {
        Self {
            sample,
            max_response: 4.0,
            force_quit_at: None,
            scores: Vec::new(),
            model: None,
            numerical_stop_reason: None,
        }
    }

    /// Sets `z_max`, the bound to which the working response is clipped.
    ///
    /// # Panics
    /// Panics unless `z_max` is finite and at least `1`: every response
    /// has magnitude at least `1`, so a smaller bound would clip them all.
    ///
    /// # Examples
    /// ```
    /// # use std::io::BufReader;
    /// # use miniboosts::{Sample, prelude::*};
    /// # let csv = b"x,y,class\n-2,0,-1\n-1,1,-1\n-1,-1,-1\n1,0,1\n2,1,1\n1,-1,1\n";
    /// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
    /// #     .unwrap()
    /// #     .set_target("class");
    /// let tree = RegressionTreeBuilder::new(&sample).max_depth(1).build();
    /// let f = LogitBoost::init(&sample).max_response(2.0).force_quit_at(5).run(&tree);
    /// assert_eq!(f.predict_all(&sample), vec![-1, -1, -1, 1, 1, 1]);
    /// ```
    pub fn max_response(mut self, z_max: f64) -> Self {
        assert!(
            z_max.is_finite() && z_max >= 1.0,
            "the response bound must be finite and at least 1"
        );
        self.max_response = z_max;
        self
    }

    /// Sets the number of rounds. Required.
    ///
    /// # Examples
    /// See [`LogitBoost`].
    pub fn force_quit_at(mut self, iteration: usize) -> Self {
        self.force_quit_at = Some(iteration);
        self
    }

    /// Returns why training stopped before the round limit, if it did
    /// (every weight underflowed to `0`). Reset by [`Booster::preprocess`].
    ///
    /// # Examples
    /// See [`LogitBoost`].
    pub fn numerical_stop_reason(&self) -> Option<&str> {
        self.numerical_stop_reason
    }
}

/// `σ(z) = 1 / (1 + e^{-z})`, without overflow.
fn logistic(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

impl<H> Booster<H> for LogitBoost<'_, H>
where
    H: Regressor + Clone,
{
    type Output = AdditiveModel<H>;

    fn name(&self) -> &str {
        "LogitBoost"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_examples, n_feature) = self.sample.shape();
        let quit = self
            .force_quit_at
            .map_or("-".to_string(), |it| it.to_string());
        Some(vec![
            ("# of examples", n_examples.to_string()),
            ("# of features", n_feature.to_string()),
            ("Max response", self.max_response.to_string()),
            ("Rounds", quit),
        ])
    }

    fn preprocess(&mut self) {
        assert!(
            self.force_quit_at.is_some(),
            "LogitBoost needs a number of rounds; set it with `force_quit_at`"
        );
        self.sample.is_valid_binary_instance();
        self.scores = vec![0.0; self.sample.shape().0];
        self.model = Some(AdditiveModel::new(0.0, ScoreLink::Tanh));
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
        let z_max = self.max_response;
        let (responses, weights): (Vec<f64>, Vec<f64>) = self
            .scores
            .iter()
            .zip(target)
            .map(|(&f, &y)| {
                // p = σ(2F); z = 1/p for y = +1 and -1/(1 - p) for y = -1.
                let z = if y > 0.0 {
                    1.0 + (-2.0 * f).exp()
                } else {
                    -(1.0 + (2.0 * f).exp())
                };
                let w = logistic(2.0 * f) * logistic(-2.0 * f);
                (z.clamp(-z_max, z_max), w)
            })
            .unzip();
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            self.numerical_stop_reason = Some("every LogitBoost weight underflowed to 0");
            return ControlFlow::Break(iteration - 1);
        }
        let dist: Vec<f64> = weights.iter().map(|w| w / total).collect();
        let h = weak_learner.produce(&self.sample.with_target(responses), &dist);
        let values: Vec<f64> = (0..target.len())
            .map(|i| h.predict(self.sample, i))
            .collect();
        assert!(
            values.iter().all(|v| v.is_finite()),
            "LogitBoost needs finite weak-hypothesis predictions"
        );
        self.scores
            .iter_mut()
            .zip(&values)
            .for_each(|(f, v)| *f += 0.5 * v);
        self.model
            .as_mut()
            .expect("call preprocess before boost")
            .push(0.5, h);
        ControlFlow::Continue(())
    }

    fn postprocess(&mut self) -> Self::Output {
        self.model
            .clone()
            .expect("call preprocess before postprocess")
    }
}

impl<H> CurrentHypothesis for LogitBoost<'_, H>
where
    H: Regressor + Clone,
{
    type Output = AdditiveModel<H>;
    fn current_hypothesis(&self) -> Self::Output {
        self.model
            .clone()
            .unwrap_or_else(|| AdditiveModel::new(0.0, ScoreLink::Tanh))
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

    /// `(1/m) Σ_i ln(1 + e^{-2 y_i F_i})`, the negative log-likelihood.
    fn deviance(sample: &Sample, scores: &[f64]) -> f64 {
        scores
            .iter()
            .zip(sample.target())
            .map(|(f, y)| (-2.0 * y * f).exp().ln_1p())
            .sum::<f64>()
            / scores.len() as f64
    }

    #[test]
    fn the_first_round_is_the_least_squares_fit_of_twice_the_labels() {
        // At F = 0: p = 1/2, z = 2y, w = 1/4 (uniform). With a depth-0 tree
        // (a constant), f_1 = mean(2y) = 0 here (balanced labels).
        let sample = xor_like_sample();
        let tree = RegressionTreeBuilder::new(&sample).max_depth(0).build();
        let mut booster = LogitBoost::<RegressionTreeRegressor>::init(&sample).force_quit_at(1);
        booster.preprocess();
        assert!(booster.boost(&tree, 1).is_continue());
        assert!(booster.scores.iter().all(|f| f.abs() < 1e-15));
    }

    #[test]
    fn the_log_likelihood_improves_and_scores_match_the_model() {
        let sample = xor_like_sample();
        let tree = RegressionTreeBuilder::new(&sample).max_depth(1).build();
        let mut booster = LogitBoost::<RegressionTreeRegressor>::init(&sample).force_quit_at(20);
        booster.preprocess();
        let start = deviance(&sample, &booster.scores);
        for iteration in 1..=20 {
            assert!(booster.boost(&tree, iteration).is_continue());
        }
        // Newton steps need not decrease the loss every round; over 20
        // rounds on this sample they do so overall.
        assert!(deviance(&sample, &booster.scores) < start);
        let f = booster.postprocess();
        for i in 0..12 {
            let score = Regressor::predict(&f, &sample, i);
            assert!((score - booster.scores[i]).abs() < 1e-12);
        }
    }

    #[test]
    #[should_panic(expected = "number of rounds")]
    fn missing_round_limit_panics() {
        let sample = xor_like_sample();
        LogitBoost::<RegressionTreeRegressor>::init(&sample).preprocess();
    }
}
