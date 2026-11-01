//! Linear weak learners: [`BoundedLinear`] (exact edge maximization over
//! bounded linear functions) and [`LogisticRegression`] (weighted logistic
//! regression, returning halfspaces or confidence-rated linear rules).
//!
//! Both return a [`LinearClassifier`] `x ↦ φ(w·x + b)`, with `φ` given by
//! [`LinearOutput`]. Features are read by name, so a test sample with the
//! same columns in another order is evaluated correctly.
//!
//! **Which one to use.** A convex combination of linear functions is again
//! linear, so boosting [`BoundedLinear`] can only produce a linear
//! classifier; it is meant for checking theory (its edge maximization is
//! exact), not for gaining expressiveness. Halfspaces
//! ([`LinearOutput::Sign`]) and `tanh` rules combine into nonlinear
//! classifiers, but maximizing the edge over halfspaces is NP-hard in
//! general, so [`LogisticRegression`] fits a convex surrogate instead and
//! carries no edge guarantee.
//!
//! # Examples
//! Boost halfspaces on a quadrant: the labels are `+1` iff `x > 0` and
//! `y > 0`, which no single halfspace separates.
//! ```
//! use miniboosts::{Sample, prelude::*};
//! use miniboosts::weak_learners::linear::LogisticRegression;
//!
//! let mut rows = Vec::new();
//! let mut y = Vec::new();
//! for a in [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0] {
//!     for b in [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0] {
//!         rows.push(vec![a, b]);
//!         y.push(if a > 0.0 && b > 0.0 { 1.0 } else { -1.0 });
//!     }
//! }
//! let sample = Sample::from_dense(&rows, Some(&y)).unwrap();
//!
//! let f = AdaBoost::init(&sample).force_quit_at(30).run(&LogisticRegression::new());
//! let correct = f
//!     .predict_all(&sample)
//!     .iter()
//!     .zip(&y)
//!     .filter(|&(&p, &yi)| p as f64 == yi)
//!     .count();
//! // Better than predicting the majority label -1 (27 of 36).
//! assert!(correct > 27, "{correct}");
//! ```
use crate::core::{Classifier, Sample, WeakLearner};
use serde::{Deserialize, Serialize};

/// How a [`LinearClassifier`] maps the score `z = w·x + b` to a confidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinearOutput {
    /// `+1` if `z ≥ 0` and `-1` otherwise: a halfspace.
    Sign,
    /// `tanh(z / 2) = 2σ(z) - 1`, where `σ` is the logistic function: for
    /// logistic regression, the difference of the two class probabilities.
    Tanh,
    /// `z` clamped to `[-1, 1]`: a linear function. [`BoundedLinear`]
    /// scales `w` and `b` so that the clamp is inactive on its training
    /// sample.
    Clamp,
}

/// The hypothesis `x ↦ φ(w·x + b)` returned by the linear learners, with
/// `φ` given by [`LinearOutput`].
///
/// # Examples
/// ```
/// use miniboosts::{Classifier, Sample};
/// use miniboosts::weak_learners::linear::{LinearClassifier, LinearOutput};
///
/// let rows = vec![vec![1.0, 2.0], vec![-1.0, 0.0]];
/// let sample = Sample::from_dense(&rows, None).unwrap();
/// let names = vec!["feature_0".to_string(), "feature_1".to_string()];
/// // z = 0.5 x_0 - 0.25 x_1 + 0.1: 0.1 on row 0 and -0.4 on row 1.
/// let h = LinearClassifier::new(names, vec![0.5, -0.25], 0.1, LinearOutput::Sign);
/// assert_eq!(h.confidence_all(&sample), vec![1.0, -1.0]);
/// assert!((h.score(&sample, 1) + 0.4).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearClassifier {
    feature_names: Vec<String>,
    weights: Vec<f64>,
    bias: f64,
    output: LinearOutput,
}

impl LinearClassifier {
    /// Constructs `x ↦ φ(w·x + b)` with `w = weights` over the features
    /// `feature_names` and `b = bias`.
    ///
    /// # Panics
    /// Panics if the lengths differ or a parameter is not finite.
    ///
    /// # Examples
    /// See [`LinearClassifier`].
    pub fn new(
        feature_names: Vec<String>,
        weights: Vec<f64>,
        bias: f64,
        output: LinearOutput,
    ) -> Self {
        assert_eq!(feature_names.len(), weights.len(), "one weight per feature");
        assert!(
            weights.iter().all(|w| w.is_finite()) && bias.is_finite(),
            "linear classifier parameters must be finite"
        );
        Self {
            feature_names,
            weights,
            bias,
            output,
        }
    }

    /// Returns the score `w·x + b` of the example in `row`.
    ///
    /// # Panics
    /// Panics if `sample` lacks one of the features.
    ///
    /// # Examples
    /// See [`LinearClassifier`].
    pub fn score(&self, sample: &Sample, row: usize) -> f64 {
        self.feature_names
            .iter()
            .zip(&self.weights)
            .map(|(name, w)| w * sample[name][row])
            .sum::<f64>()
            + self.bias
    }

    /// Returns the weights `w`, in the order of
    /// [`LinearClassifier::feature_names`].
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Returns the bias `b`.
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// Returns the names of the features the weights refer to.
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Returns the output map `φ`.
    pub fn output(&self) -> LinearOutput {
        self.output
    }
}

impl Classifier for LinearClassifier {
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        let z = self.score(sample, row);
        match self.output {
            LinearOutput::Sign => {
                if z >= 0.0 {
                    1.0
                } else {
                    -1.0
                }
            }
            LinearOutput::Tanh => (0.5 * z).tanh(),
            LinearOutput::Clamp => z.clamp(-1.0, 1.0),
        }
    }
}

/// Returns the feature names and the rows `x_i` of `sample`.
fn names_and_rows(sample: &Sample) -> (Vec<String>, Vec<Vec<f64>>) {
    let features = sample.features();
    let names = features.iter().map(|f| f.name().to_string()).collect();
    let rows = (0..sample.shape().0)
        .map(|i| features.iter().map(|f| f[i]).collect())
        .collect();
    (names, rows)
}

fn check_inputs(sample: &Sample, dist: &[f64]) {
    let n_examples = sample.shape().0;
    assert_eq!(
        sample.target().len(),
        n_examples,
        "the sample needs labels; use `Sample::set_target`"
    );
    assert_eq!(
        dist.len(),
        n_examples,
        "the distribution needs one entry per example"
    );
    assert!(
        dist.iter().all(|d| d.is_finite() && *d >= 0.0),
        "distribution entries must be finite and nonnegative"
    );
}

/// The class of linear functions with bounded parameters, with exact edge
/// maximization in closed form.
///
/// # Hypothesis class
/// Let `R = max_i ‖x_i‖₂` over the training sample and
/// `S = √(R² + 1)`. The class is
/// ```txt
/// { x ↦ (w·x + b) / S  :  ‖(w, b)‖₂ ≤ 1 }.
/// ```
/// By Cauchy–Schwarz, `|w·x_i + b| ≤ ‖(w, b)‖₂ ‖(x_i, 1)‖₂ ≤ S`, so every
/// member has confidences in `[-1, 1]` on the training sample.
///
/// # Algorithm
/// The edge of `(w, b)` under `d` is `⟨(w, b), g⟩ / S` with
/// `g = Σ_i d_i y_i (x_i, 1)`, so `(w, b) = g / ‖g‖₂` maximizes it, with
/// edge `‖g‖₂ / S`. If `g = 0`, every member has edge `0` and the zero
/// function is returned. The result uses [`LinearOutput::Clamp`], which only
/// matters on examples outside the training radius.
///
/// # Guarantee
/// The returned hypothesis has the largest edge in the class (exactly, up
/// to rounding). The class is closed under negation.
///
/// # Limitations
/// - Convex combinations of members are again linear functions, so any
///   booster's output is a single linear classifier.
/// - `R` is taken from the sample passed to
///   [`produce`](WeakLearner::produce), so the class is defined by the
///   training sample; on other samples, confidences are clamped.
/// - The class depends on the scale of the features.
///
/// # Examples
/// ```
/// use miniboosts::{Classifier, Sample, WeakLearner};
/// use miniboosts::weak_learners::linear::BoundedLinear;
///
/// let rows = vec![vec![3.0], vec![-1.0]];
/// let sample = Sample::from_dense(&rows, Some(&[1.0, -1.0])).unwrap();
/// let d = [0.5, 0.5];
/// let h = BoundedLinear::new().produce(&sample, &d);
/// // g = 0.5 (3, 1) - 0.5 (-1, 1) = (2, 0), S = √10: edge 2 / √10.
/// let edge: f64 = (0..2).map(|i| d[i] * sample.target()[i] * h.confidence(&sample, i)).sum();
/// assert!((edge - 2.0 / 10f64.sqrt()).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct BoundedLinear;

impl BoundedLinear {
    /// Constructs the learner. It has no parameters.
    ///
    /// # Examples
    /// See [`BoundedLinear`].
    pub fn new() -> Self {
        Self
    }
}

impl WeakLearner for BoundedLinear {
    type Hypothesis = LinearClassifier;

    fn name(&self) -> &str {
        "Bounded linear"
    }

    /// Returns the edge-maximizing member; see [`BoundedLinear`].
    ///
    /// # Panics
    /// Panics if the sample has no labels, or `dist` has the wrong length
    /// or a negative or non-finite entry.
    fn produce(&self, sample: &Sample, dist: &[f64]) -> Self::Hypothesis {
        check_inputs(sample, dist);
        let (names, rows) = names_and_rows(sample);
        let dim = names.len();
        let radius_squared = rows
            .iter()
            .map(|x| x.iter().map(|v| v * v).sum::<f64>())
            .fold(0.0, f64::max);
        let scale = (radius_squared + 1.0).sqrt();

        // g = Σ_i d_i y_i (x_i, 1)
        let mut g = vec![0.0; dim + 1];
        for ((x, &y), &d) in rows.iter().zip(sample.target()).zip(dist) {
            let c = d * y;
            g.iter_mut().zip(x).for_each(|(gk, xk)| *gk += c * xk);
            g[dim] += c;
        }
        let norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        let factor = if norm > 0.0 {
            1.0 / (norm * scale)
        } else {
            0.0
        };
        let weights = g[..dim].iter().map(|v| v * factor).collect();
        LinearClassifier::new(names, weights, g[dim] * factor, LinearOutput::Clamp)
    }
}

/// Weighted, L2-regularized logistic regression as a weak learner.
///
/// # Algorithm
/// Given a distribution `d`, it minimizes
/// ```txt
/// L(w, b) = Σ_i d_i ln(1 + exp(-y_i (w·x_i + b))) + (λ/2) (‖w‖² + b²)
/// ```
/// by Newton's method with backtracking line search, starting from zero,
/// until the Newton decrement satisfies `δ²/2 ≤ 1e-14` (at most 100
/// iterations). `L` is strictly convex for `λ > 0`, so the minimizer is
/// unique. The result is returned as a halfspace ([`LinearOutput::Sign`],
/// the default) or as `tanh((w·x + b)/2)` ([`LinearOutput::Tanh`]).
///
/// The bias is regularized too, so that the Hessian stays positive definite
/// when the data are separable.
///
/// # Guarantee
/// None on the edge: the logistic loss is a convex surrogate, and the
/// returned halfspace need not maximize the edge (which is NP-hard over
/// halfspaces in general). The returned parameters minimize `L` to the
/// stated tolerance.
///
/// # Limitations
/// - Each Newton iteration costs `O(m d²)` time and `O(d²)` memory for `m`
///   examples and `d` features, so it is meant for moderate `d`.
/// - The result depends on the scale of the features through `λ`;
///   standardize features for comparable regularization.
/// - The surrogate can return a useless halfspace even when a good one
///   exists. For example, on data symmetric about a point (labels `+1`
///   on an interval around `0`) with balanced class weights, which AdaBoost
///   produces after one round, the minimizer is `w = 0, b = 0`: a
///   hypothesis with edge `0`, so boosting makes no progress.
/// - Training panics if the iteration does not converge within 100
///   iterations or produces non-finite values, instead of returning an
///   unconverged hypothesis.
///
/// # Parameters
/// | Builder method | Default | Meaning |
/// | --- | --- | --- |
/// | [`regularization`](LogisticRegression::regularization) | `1e-4` | `λ` (not taken from a paper). |
/// | [`output`](LogisticRegression::output) | [`LinearOutput::Sign`] | Halfspace or `tanh` rule. |
///
/// # Examples
/// See the [module documentation](self).
#[derive(Debug, Clone, Copy)]
pub struct LogisticRegression {
    regularization: f64,
    output: LinearOutput,
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl LogisticRegression {
    /// Constructs the learner with `λ = 1e-4` and halfspace output.
    ///
    /// # Examples
    /// See the [module documentation](self).
    pub fn new() -> Self {
        Self {
            regularization: 1e-4,
            output: LinearOutput::Sign,
        }
    }

    /// Sets the regularization parameter `λ`.
    ///
    /// # Panics
    /// Panics if `regularization` is not finite and positive.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{Classifier, Sample, WeakLearner};
    /// use miniboosts::weak_learners::linear::{LinearOutput, LogisticRegression};
    ///
    /// let rows = vec![vec![-1.0], vec![1.0]];
    /// let sample = Sample::from_dense(&rows, Some(&[-1.0, 1.0])).unwrap();
    /// // Strong regularization keeps the weights small, so the tanh rule is
    /// // far from ±1.
    /// let h = LogisticRegression::new()
    ///     .regularization(10.0)
    ///     .output(LinearOutput::Tanh)
    ///     .produce(&sample, &[0.5, 0.5]);
    /// assert!(h.confidence(&sample, 1) > 0.0 && h.confidence(&sample, 1) < 0.1);
    /// ```
    pub fn regularization(mut self, regularization: f64) -> Self {
        assert!(
            regularization.is_finite() && regularization > 0.0,
            "logistic regression needs a finite, positive regularization"
        );
        self.regularization = regularization;
        self
    }

    /// Sets how the score is turned into a confidence:
    /// [`LinearOutput::Sign`] (halfspace) or [`LinearOutput::Tanh`].
    ///
    /// # Panics
    /// Panics for [`LinearOutput::Clamp`], whose confidences need not lie in
    /// `[-1, 1]` for logistic-regression weights.
    ///
    /// # Examples
    /// See [`LogisticRegression::regularization`].
    pub fn output(mut self, output: LinearOutput) -> Self {
        assert!(
            output != LinearOutput::Clamp,
            "logistic regression outputs `Sign` or `Tanh`"
        );
        self.output = output;
        self
    }

    /// Minimizes `L` for the weights `dist` and returns `(w, b)` as one
    /// vector `θ = (w, b)`.
    // Index loops mirror the symmetric Hessian's formulas.
    #[allow(clippy::needless_range_loop)]
    fn fit(&self, rows: &[Vec<f64>], target: &[f64], dist: &[f64]) -> Vec<f64> {
        let dim = rows.first().map_or(0, |x| x.len()) + 1;
        let lambda = self.regularization;
        let augmented = |x: &[f64]| x.iter().copied().chain([1.0]).collect::<Vec<f64>>();
        let xs: Vec<Vec<f64>> = rows.iter().map(|x| augmented(x)).collect();
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let loss = |theta: &[f64]| {
            xs.iter()
                .zip(target)
                .zip(dist)
                .map(|((x, y), d)| d * softplus(-y * dot(theta, x)))
                .sum::<f64>()
                + 0.5 * lambda * dot(theta, theta)
        };

        let mut theta = vec![0.0; dim];
        let mut value = loss(&theta);
        for _ in 0..100 {
            // Gradient and Hessian.
            let mut grad: Vec<f64> = theta.iter().map(|t| lambda * t).collect();
            let mut hess = vec![vec![0.0; dim]; dim];
            for ((x, &y), &d) in xs.iter().zip(target).zip(dist) {
                if d == 0.0 {
                    continue;
                }
                let z = dot(&theta, x);
                let coef = -d * y * logistic(-y * z);
                let curv = d * logistic(z) * logistic(-z);
                for a in 0..dim {
                    grad[a] += coef * x[a];
                    for b in 0..=a {
                        hess[a][b] += curv * x[a] * x[b];
                    }
                }
            }
            for a in 0..dim {
                hess[a][a] += lambda;
                for b in 0..a {
                    hess[b][a] = hess[a][b];
                }
            }
            let step: Vec<f64> = cholesky_solve(&hess, &grad)
                .expect("logistic regression: the Hessian is not positive definite")
                .into_iter()
                .map(|v| -v)
                .collect();
            let decrement = -dot(&grad, &step);
            assert!(
                decrement.is_finite(),
                "logistic regression produced a non-finite Newton step"
            );
            if decrement / 2.0 <= 1e-14 {
                return theta;
            }
            // Backtracking (Armijo) line search.
            let mut t = 1.0;
            loop {
                let candidate: Vec<f64> = theta.iter().zip(&step).map(|(a, s)| a + t * s).collect();
                let candidate_value = loss(&candidate);
                if candidate_value <= value - 1e-4 * t * decrement {
                    theta = candidate;
                    value = candidate_value;
                    break;
                }
                t *= 0.5;
                if t < 1e-12 {
                    // No further decrease is representable: θ is optimal to
                    // rounding.
                    return theta;
                }
            }
        }
        panic!("logistic regression did not converge in 100 Newton iterations");
    }
}

impl WeakLearner for LogisticRegression {
    type Hypothesis = LinearClassifier;

    fn name(&self) -> &str {
        "Logistic regression"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        Some(vec![
            ("Regularization", self.regularization.to_string()),
            ("Output", format!("{:?}", self.output)),
        ])
    }

    /// Fits the weighted logistic regression; see [`LogisticRegression`].
    ///
    /// # Panics
    /// Panics if the sample has no labels, `dist` has the wrong length or a
    /// negative or non-finite entry, or the Newton iteration fails.
    fn produce(&self, sample: &Sample, dist: &[f64]) -> Self::Hypothesis {
        check_inputs(sample, dist);
        let (names, rows) = names_and_rows(sample);
        let theta = self.fit(&rows, sample.target(), dist);
        let dim = names.len();
        LinearClassifier::new(names, theta[..dim].to_vec(), theta[dim], self.output)
    }
}

/// The logistic function `1 / (1 + e^{-z})`, without overflow.
fn logistic(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// `ln(1 + e^u)`, without overflow.
fn softplus(u: f64) -> f64 {
    u.max(0.0) + (-u.abs()).exp().ln_1p()
}

/// Solves `A x = b` for a symmetric positive definite `A` by Cholesky
/// factorization; `None` if `A` is not numerically positive definite.
fn cholesky_solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                let diag = a[i][i] - sum;
                if diag <= 0.0 || !diag.is_finite() {
                    return None;
                }
                l[i][i] = diag.sqrt();
            } else {
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }
    // L y = b, then Lᵀ x = y.
    let mut y = vec![0.0; n];
    for i in 0..n {
        let sum: f64 = (0..i).map(|k| l[i][k] * y[k]).sum();
        y[i] = (b[i] - sum) / l[i][i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let sum: f64 = (i + 1..n).map(|k| l[k][i] * x[k]).sum();
        x[i] = (y[i] - sum) / l[i][i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_feature_sample() -> Sample {
        let rows = vec![
            vec![1.0, 2.0],
            vec![2.0, -1.0],
            vec![-1.0, 0.5],
            vec![-2.0, -2.0],
            vec![0.5, 1.0],
        ];
        Sample::from_dense(&rows, Some(&[1.0, 1.0, -1.0, -1.0, -1.0])).unwrap()
    }

    fn edge<H: Classifier>(sample: &Sample, dist: &[f64], h: &H) -> f64 {
        (0..dist.len())
            .map(|i| dist[i] * sample.target()[i] * h.confidence(sample, i))
            .sum()
    }

    #[test]
    fn bounded_linear_maximizes_the_edge_over_its_class() {
        let sample = two_feature_sample();
        let dist = [0.1, 0.3, 0.2, 0.15, 0.25];
        let h = BoundedLinear::new().produce(&sample, &dist);
        // Training confidences stay inside [-1, 1] without clamping.
        for i in 0..5 {
            assert!(h.score(&sample, i).abs() <= 1.0);
        }
        let best = edge(&sample, &dist, &h);
        // Every unit vector (w0, w1, b) on a grid has edge at most `best`.
        let scale = (8.0f64 + 1.0).sqrt(); // R² = max ‖x_i‖² = 8
        let names = vec!["feature_0".to_string(), "feature_1".to_string()];
        for a in 0..24 {
            for c in 0..12 {
                let (phi, psi) = (a as f64 * 0.2618, c as f64 * 0.2618);
                let v = [psi.sin() * phi.cos(), psi.sin() * phi.sin(), psi.cos()];
                let g = LinearClassifier::new(
                    names.clone(),
                    vec![v[0] / scale, v[1] / scale],
                    v[2] / scale,
                    LinearOutput::Clamp,
                );
                assert!(edge(&sample, &dist, &g) <= best + 1e-15);
            }
        }
    }

    #[test]
    fn bounded_linear_returns_zero_when_every_edge_is_zero() {
        // Identical rows with opposite labels and equal weight: g = 0.
        let rows = vec![vec![1.0], vec![1.0]];
        let sample = Sample::from_dense(&rows, Some(&[1.0, -1.0])).unwrap();
        let h = BoundedLinear::new().produce(&sample, &[0.5, 0.5]);
        assert_eq!((h.weights(), h.bias()), (&[0.0][..], 0.0));
    }

    /// The gradient of L at the returned parameters, which must vanish.
    fn gradient(sample: &Sample, dist: &[f64], lambda: f64, h: &LinearClassifier) -> Vec<f64> {
        let (_, rows) = names_and_rows(sample);
        let mut grad: Vec<f64> = h
            .weights()
            .iter()
            .chain([h.bias()].iter())
            .map(|t| lambda * t)
            .collect();
        for (i, x) in rows.iter().enumerate() {
            let y = sample.target()[i];
            let coef = -dist[i] * y * logistic(-y * h.score(sample, i));
            grad.iter_mut()
                .zip(x.iter().chain([1.0].iter()))
                .for_each(|(g, xk)| *g += coef * xk);
        }
        grad
    }

    #[test]
    fn logistic_regression_reaches_a_stationary_point() {
        let sample = two_feature_sample();
        let dist = [0.1, 0.3, 0.2, 0.15, 0.25];
        for lambda in [1e-4, 1e-2, 1.0] {
            let h = LogisticRegression::new()
                .regularization(lambda)
                .produce(&sample, &dist);
            let grad = gradient(&sample, &dist, lambda, &h);
            // Newton decrement ≤ √(2e-14) bounds the gradient in the
            // Hessian's inverse norm; 1e-6 is generous in the Euclidean norm.
            assert!(
                grad.iter().all(|g| g.abs() < 1e-6),
                "λ = {lambda}: {grad:?}"
            );
        }
    }

    #[test]
    fn duplicating_an_example_equals_doubling_its_weight() {
        // L depends on the data only through Σ_i d_i (loss of example i),
        // so a duplicated row with weights (1, 1, 1, 1)/4 and the original
        // rows with weights (2, 1, 1)/4 define the same objective.
        let rows = vec![vec![0.0], vec![1.0], vec![3.0]];
        let y = [-1.0, 1.0, -1.0];
        let sample = Sample::from_dense(&rows, Some(&y)).unwrap();
        let rows_dup = vec![vec![0.0], vec![0.0], vec![1.0], vec![3.0]];
        let sample_dup = Sample::from_dense(&rows_dup, Some(&[-1.0, -1.0, 1.0, -1.0])).unwrap();
        let learner = LogisticRegression::new().regularization(0.01);
        let h = learner.produce(&sample, &[0.5, 0.25, 0.25]);
        let h_dup = learner.produce(&sample_dup, &[0.25; 4]);
        assert!((h.weights()[0] - h_dup.weights()[0]).abs() < 1e-10);
        assert!((h.bias() - h_dup.bias()).abs() < 1e-10);
    }

    #[test]
    fn zero_weight_examples_are_ignored() {
        let rows = vec![vec![-1.0], vec![1.0], vec![100.0]];
        let sample = Sample::from_dense(&rows, Some(&[-1.0, 1.0, -1.0])).unwrap();
        let rows_short = vec![vec![-1.0], vec![1.0]];
        let short = Sample::from_dense(&rows_short, Some(&[-1.0, 1.0])).unwrap();
        let learner = LogisticRegression::new();
        let h = learner.produce(&sample, &[0.5, 0.5, 0.0]);
        let g = learner.produce(&short, &[0.5, 0.5]);
        assert_eq!((h.weights(), h.bias()), (g.weights(), g.bias()));
        // The symmetric problem has b = 0 and separates the two points.
        assert!(h.bias().abs() < 1e-12 && h.weights()[0] > 0.0);
    }

    #[test]
    fn outputs_follow_the_score() {
        let sample = two_feature_sample();
        let names = vec!["feature_0".to_string(), "feature_1".to_string()];
        let make = |o| LinearClassifier::new(names.clone(), vec![1.0, 0.0], -1.0, o);
        // Scores x_0 - 1: 0, 1, -2, -3, -0.5.
        assert_eq!(
            make(LinearOutput::Sign).confidence_all(&sample),
            vec![1.0, 1.0, -1.0, -1.0, -1.0]
        );
        assert_eq!(
            make(LinearOutput::Clamp).confidence_all(&sample),
            vec![0.0, 1.0, -1.0, -1.0, -0.5]
        );
        let tanh = make(LinearOutput::Tanh).confidence_all(&sample);
        assert!((tanh[1] - 0.5f64.tanh()).abs() < 1e-15);
        assert!(
            std::panic::catch_unwind(|| LogisticRegression::new().output(LinearOutput::Clamp))
                .is_err()
        );
    }

    #[test]
    fn cholesky_solves_a_small_system() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let x = cholesky_solve(&a, &[2.0, 1.0]).unwrap();
        // 4x + 2y = 2, 2x + 3y = 1 gives x = 0.5, y = 0.
        assert!((x[0] - 0.5).abs() < 1e-15 && x[1].abs() < 1e-15);
        assert!(cholesky_solve(&[vec![0.0]], &[1.0]).is_none());
    }
}
