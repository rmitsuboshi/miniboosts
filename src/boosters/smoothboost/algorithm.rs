//! Implementation of [`SmoothBoost`] (Servedio, 2001/2003).
//!
//! See the [`SmoothBoost`] struct for the update rule, guarantees, and an example.
use rayon::prelude::*;

use crate::core::CurrentHypothesis;
use crate::core::{Booster, Classifier, Sample, WeakLearner};
use crate::hypotheses::WeightedMajority;

use std::ops::ControlFlow;

/// SmoothBoost: a boosting algorithm that only generates smooth
/// distributions, by Servedio.
///
/// # Algorithm
/// Let `m` be the number of training examples, `y_j ∈ {-1, +1}` the labels,
/// and `h_t(x_j) ∈ [-1, 1]` the weak hypotheses' confidences.
/// The parameters are the target error `κ ∈ (0, 1)`, the assumed weak-learner
/// advantage `γ ∈ (0, 1/2)`, and the margin `θ = γ / (2 + γ)`. This
/// implementation follows Figure 1 of the JMLR version. Start with
/// `M_1(j) = 1` and `N_0(j) = 0`, and repeat for `t = 1, 2, ...`:
///
/// 1. Stop if `Σ_j M_t(j) < κ m`.
/// 2. Set `D_t(j) = M_t(j) / Σ_k M_t(k)` and call the weak learner on `D_t`
///    to obtain `h_t`.
/// 3. Update
///    ```txt
///    N_t(j)     = N_{t-1}(j) + y_j h_t(x_j) - θ,
///    M_{t+1}(j) = 1                        if N_t(j) ≤ 0,
///                 (1 - γ)^{N_t(j) / 2}     otherwise.
///    ```
///    (The paper uses `N_t(j) < 0` for the first case; both cases give `1`
///    at `N_t(j) = 0`.)
///
/// The output is the [`WeightedMajority`] that averages all `T` hypotheses
/// with equal weights, `sign((1/T) Σ_t h_t(x))`.
///
/// Implementation choice: besides the stopping test in step 1, training
/// stops after `ceil(2 / (κ γ² sqrt(1 - γ)))` rounds, the paper's bound on
/// `T`. Under the weak-learning assumption the stopping test fires first, so
/// reaching this budget indicates that the assumption failed.
///
/// # Theoretical guarantee
/// From Servedio (2003), assuming only what each statement lists:
/// - Every `D_t` is `1/(κ m)`-smooth: `D_t(j) ≤ 1/(κ m)` for all `j`
///   (Lemma 1). This holds regardless of the weak learner.
/// - If the stopping test fires, the averaged hypothesis `f` has margin
///   `y_j f(x_j) > θ` on all but fewer than `κ m` examples (Theorem 2),
///   so its training error is below `κ`.
/// - If every `h_t` has advantage at least `γ` under `D_t`, that is,
///   ```txt
///   (1/2) Σ_j D_t(j) |h_t(x_j) - y_j| ≤ 1/2 - γ
///   (equivalently, edge Σ_j D_t(j) y_j h_t(x_j) ≥ 2γ),
///   ```
///   and `θ = γ / (2 + γ)`, then SmoothBoost stops with
///   `T < 2 / (κ γ² sqrt(1 - γ))` (Theorem 3).
///
/// # Limitations
/// - The advantage assumption is not checked at runtime. Choosing a tree
///   depth does not establish it. If training ends at the round budget
///   instead of the stopping test, the margin guarantee does not apply, and
///   nothing signals this.
/// - Smoothness bounds each example's weight but the guarantee concerns the
///   training sample only; `κ` is not a test-error bound.
/// - Confidences outside `[-1, 1]` are not rejected, but the analysis
///   assumes them.
/// - Invalid `kappa` is only detected when training starts (panic);
///   `gamma` is validated immediately.
/// - Labels must be `-1` or `+1` and both classes must be present.
/// - Each round costs `O(m)` time beyond the weak learner, and every
///   hypothesis is stored.
///
/// # Parameters
/// | Builder method | Default | Meaning |
/// | --- | --- | --- |
/// | [`kappa`](SmoothBoost::kappa) | `0.5` | Target training error `κ ∈ (0, 1)`. |
/// | [`gamma`](SmoothBoost::gamma) | `0.25` | Assumed weak-learner advantage `γ ∈ (0, 1/2)`; also sets `θ = γ / (2 + γ)`. |
///
/// # Reference
/// Rocco A. Servedio.
/// *Smooth Boosting and Learning with Malicious Noise.*
/// In Computational Learning Theory (COLT), LNCS 2111, 2001,
/// <https://link.springer.com/chapter/10.1007/3-540-44581-1_31>.
/// Journal version: Journal of Machine Learning Research, 4:633–648, 2003,
/// <https://www.jmlr.org/papers/volume4/servedio03a/servedio03a.pdf>.
///
/// # Examples
/// Train on a held-out split with decision stumps and evaluate on the test
/// part.
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
///
/// // The weak learner sees only the training split.
/// let learner = DecisionTreeBuilder::new(&train).max_depth(1).build();
///
/// // Target training error κ = 0.25 under the assumed advantage γ = 0.2
/// // (edge 0.4), which also sets the margin θ = γ / (2 + γ).
/// let (kappa, gamma) = (0.25, 0.2);
/// let mut booster = SmoothBoost::init(&train).kappa(kappa).gamma(gamma);
/// let f = booster.run(&learner);
///
/// let predictions = f.predict_all(&test);
/// assert!(predictions.iter().all(|&p| p == -1 || p == 1));
/// let correct = predictions
///     .iter()
///     .zip(test.target())
///     .filter(|&(&p, &y)| p as f64 == y)
///     .count();
/// let accuracy = correct as f64 / test.shape().0 as f64;
/// println!("SmoothBoost test accuracy: {accuracy}");
///
/// // The run never exceeds the budget ceil(2 / (κ γ² sqrt(1 - γ))) = 224.
/// let budget = (2.0 / (kappa * gamma * gamma * (1.0f64 - gamma).sqrt())).ceil() as usize;
/// assert_eq!(budget, 224);
/// assert!(!f.hypotheses.is_empty() && f.hypotheses.len() <= budget);
/// // The output averages all hypotheses with equal weights.
/// let w0 = f.weights[0];
/// assert!(f.weights.iter().all(|&w| (w - w0).abs() < 1e-12));
/// ```
pub struct SmoothBoost<'a, F> {
    // Training sample
    sample: &'a Sample,

    // Target training error `κ`.
    kappa: f64,

    // Desired margin for the final hypothesis.
    // To guarantee the convergence rate, `theta` should be
    // `gamma / (2.0 + gamma)`.
    theta: f64,

    // Weak-learner guarantee;
    // for any distribution over the training examples,
    // the weak-learner returns a hypothesis
    // with advantage at least `gamma` (edge at least `2 * gamma`).
    gamma: f64,

    // The number of training examples.
    n_examples: usize,

    // Current round.
    current: usize,

    // Terminated iteration.
    terminated: usize,

    // Iteration budget.
    max_iter: usize,

    hypotheses: Vec<F>,

    // Measures `M_t(j)` and cumulative margin excesses `N_t(j)`.
    m: Vec<f64>,
    n: Vec<f64>,
}

impl<'a, F> SmoothBoost<'a, F> {
    /// Constructs a new instance of `SmoothBoost` for `sample` with
    /// `kappa = 0.5` and weak-learner advantage `gamma = 0.25`.
    ///
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
    /// // Defaults: κ = 0.5, γ = 0.25.
    /// let mut booster = SmoothBoost::init(&sample);
    /// let f = booster.run(&learner);
    /// assert_eq!(f.predict_all(&sample).len(), 6);
    /// ```
    pub fn init(sample: &'a Sample) -> Self {
        let n_examples = sample.shape().0;

        let gamma = 0.25;

        Self {
            sample,

            kappa: 0.5,
            theta: gamma / (2.0 + gamma), // gamma / (2.0 + gamma)
            gamma,

            n_examples,

            current: 0_usize,

            terminated: usize::MAX,
            max_iter: usize::MAX,

            hypotheses: Vec::new(),

            m: Vec::new(),
            n: Vec::new(),
        }
    }

    /// Sets the target training error `kappa`, which must lie in `(0, 1)`.
    /// Training stops once the total measure `Σ_j M_t(j)` falls below
    /// `kappa · m`; smaller values need more rounds (the bound grows as
    /// `1 / kappa`) and allow distributions up to `1 / (kappa m)` per example.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Panics
    /// This method does not validate `kappa`; [`Booster::run`] panics
    /// if `kappa` is not in `(0, 1)`.
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
    /// // Aim for training error below κ = 0.1: if the stopping test fires
    /// // before the budget, fewer than 0.1 · 6 examples have margin ≤ θ.
    /// let mut booster = SmoothBoost::init(&sample).kappa(0.1);
    /// let f = booster.run(&learner);
    /// assert!(!f.hypotheses.is_empty());
    /// ```
    #[inline(always)]
    pub fn kappa(mut self, kappa: f64) -> Self {
        self.kappa = kappa;

        self
    }

    /// Sets the assumed weak-learner advantage `gamma` in `(0, 0.5)`.
    /// `SmoothBoost` assumes the weak learner returns a hypothesis `h`
    /// such that
    /// `0.5 * sum_i D[i] |h(x[i]) - y[i]| <= 0.5 - gamma`
    /// for the given distribution `D`, i.e., an edge of at least `2 * gamma`.
    /// This assumption is required for the iteration bound but is not checked.
    /// `gamma` also determines the margin `theta = gamma / (2 + gamma)` and
    /// the round budget.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Panics
    /// Panics if `gamma` is not in `(0, 0.5)` (including NaN).
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
    /// // Assume edge at least 0.6 (advantage 0.3); θ = 0.3 / 2.3.
    /// let mut booster = SmoothBoost::init(&sample).kappa(0.2).gamma(0.3);
    /// let f = booster.run(&learner);
    /// assert!(!f.hypotheses.is_empty());
    /// ```
    #[inline(always)]
    pub fn gamma(mut self, gamma: f64) -> Self {
        // A positive advantage is needed for a finite iteration bound.
        assert!(gamma > 0.0 && gamma < 0.5);
        self.gamma = gamma;

        self
    }

    /// Sets `theta = gamma / (2 + gamma)`, as in Theorem 3 of the paper.
    fn theta(&mut self) {
        self.theta = self.gamma / (2.0 + self.gamma);
    }

    /// Returns the round budget `ceil(2 / (kappa gamma² sqrt(1 - gamma)))`,
    /// the paper's bound on the number of rounds before the stopping
    /// criterion holds under the weak-learning assumption.
    fn max_loop(&self) -> usize {
        let denom = self.kappa * self.gamma.powi(2) * (1.0 - self.gamma).sqrt();

        (2.0 / denom).ceil() as usize
    }

    fn check_preconditions(&self) {
        // Check `kappa`.
        if !(0.0..1.0).contains(&self.kappa) || self.kappa <= 0.0 {
            panic!(
                "Invalid kappa. \
                 The parameter `kappa` must be in (0.0, 1.0)"
            );
        }

        // Check `gamma`.
        if !(self.theta..0.5).contains(&self.gamma) {
            panic!(
                "Invalid gamma. \
                 The parameter `gamma` must be in [self.theta, 0.5)"
            );
        }
    }
}

impl<F> Booster<F> for SmoothBoost<'_, F>
where
    F: Classifier + Clone,
{
    type Output = WeightedMajority<F>;

    fn name(&self) -> &str {
        "SmoothBoost"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_examples, n_feature) = self.sample.shape();
        let info = Vec::from([
            ("# of examples", format!("{n_examples}")),
            ("# of features", format!("{n_feature}")),
            ("Tolerance (Kappa)", format!("{}", self.kappa)),
            ("Max iteration", format!("{}", self.max_iter)),
            ("Theta", format!("{}", self.theta)),
            ("Gamma (WL guarantee)", format!("{}", self.gamma)),
        ]);
        Some(info)
    }

    fn preprocess(&mut self) {
        self.sample.is_valid_binary_instance();
        self.n_examples = self.sample.shape().0;
        // Set the parameter `theta`.
        self.theta();

        // Check whether the parameter satisfies the pre-conditions.
        self.check_preconditions();

        self.current = 0_usize;
        self.max_iter = self.max_loop();
        self.terminated = self.max_iter;

        self.hypotheses = Vec::new();

        self.m = vec![1.0; self.n_examples];
        // Servedio (2003), Figure 1, line 2: N_0(j) = 0.
        // https://www.jmlr.org/papers/volume4/servedio03a/servedio03a.pdf
        self.n = vec![0.0; self.n_examples];
    }

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize) -> ControlFlow<usize>
    where
        W: WeakLearner<Hypothesis = F>,
    {
        if self.max_iter < iteration {
            return ControlFlow::Break(self.max_iter);
        }

        self.current = iteration;

        let sum = self.m.iter().sum::<f64>();
        // Check the stopping criterion (Figure 1, line 4).
        if sum < self.n_examples as f64 * self.kappa {
            self.terminated = iteration - 1;
            return ControlFlow::Break(iteration);
        }

        // Compute the distribution.
        let dist = self.m.iter().map(|mj| *mj / sum).collect::<Vec<_>>();

        // Call weak learner to obtain a hypothesis.
        self.hypotheses
            .push(weak_learner.produce(self.sample, &dist[..]));
        let h: &F = self.hypotheses.last().unwrap();

        let target = self.sample.target();
        let margins = target
            .iter()
            .enumerate()
            .map(|(i, y)| y * h.confidence(self.sample, i));

        // Update `n`
        self.n.iter_mut().zip(margins).for_each(|(nj, yh)| {
            *nj = *nj + yh - self.theta;
        });

        // Update `m`
        self.m.par_iter_mut().zip(&self.n[..]).for_each(|(mj, nj)| {
            if *nj <= 0.0 {
                *mj = 1.0;
            } else {
                *mj = (1.0 - self.gamma).powf(*nj * 0.5);
            }
        });

        ControlFlow::Continue(())
    }

    fn postprocess(&mut self) -> Self::Output {
        // Figure 1, line 11: average all T hypotheses, not the sample rows.
        let weights = vec![1f64; self.hypotheses.len()];
        WeightedMajority::from_slices(&weights[..], &self.hypotheses[..])
    }
}

impl<H> CurrentHypothesis for SmoothBoost<'_, H>
where
    H: Classifier + Clone,
{
    type Output = WeightedMajority<H>;
    fn current_hypothesis(&self) -> Self::Output {
        // Figure 1, line 11: average all T hypotheses, not the sample rows.
        let weights = vec![1f64; self.hypotheses.len()];
        WeightedMajority::from_slices(&weights[..], &self.hypotheses[..])
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    #[derive(Clone)]
    struct Constant(f64);
    impl Classifier for Constant {
        fn confidence(&self, _: &Sample, _: usize) -> f64 {
            self.0
        }
    }
    impl WeakLearner for Constant {
        type Hypothesis = Self;
        fn produce(&self, _: &Sample, _: &[f64]) -> Self {
            self.clone()
        }
    }

    #[test]
    fn default_training_keeps_every_round_and_starts_at_zero_margin() {
        let sample = Sample::dummy(2);
        let mut booster = SmoothBoost::init(&sample);
        booster.preprocess();
        assert_eq!(booster.n, vec![0.0; 2]);
        // Deliberately poor learner keeps training active for more rounds than rows.
        for iteration in 1..=3 {
            assert!(booster.boost(&Constant(1.0), iteration).is_continue());
        }
        for ensemble in [booster.current_hypothesis(), booster.postprocess()] {
            assert_eq!(ensemble.hypotheses.len(), 3);
            assert!(
                ensemble
                    .weights
                    .iter()
                    .all(|w| (*w - 1.0 / 3.0).abs() < 1e-14)
            );
        }
        assert!((booster.n[0] - 3.0 * (1.0 - booster.theta)).abs() < 1e-14);
    }

    #[test]
    #[should_panic]
    fn zero_advantage_is_rejected() {
        let sample = Sample::dummy(2);
        let _ = SmoothBoost::<Constant>::init(&sample).gamma(0.0);
    }
}
