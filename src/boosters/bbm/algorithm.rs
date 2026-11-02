//! Implementation of boost-by-majority (Freund, 1995) as
//! [`BoostByMajority`]. See the struct for details and an example.
use crate::core::CurrentHypothesis;
use crate::core::{Booster, Classifier, Sample, WeakLearner, tools::helpers};
use crate::hypotheses::MajorityVote;

use std::ops::ControlFlow;

/// Boost-by-majority (BBM) by Freund: a non-adaptive booster that, given a
/// weak-learning advantage `γ` in advance, combines a fixed number `k` of
/// weak hypotheses by an unweighted majority vote.
///
/// # Algorithm
/// Let `k` be the number of rounds, `p = 1/2 + γ`, and `r_n` the number of
/// the first `i` weak hypotheses that predict example `n` correctly. In
/// round `i + 1` (`i = 0, ..., k - 1`), example `n` gets the weight
/// (Equation (1) of Freund (2001), quoting Freund (1995))
/// ```txt
/// α_{r}^{i} = C(k - i - 1, ⌊k/2⌋ - r) p^{⌊k/2⌋ - r} (1 - p)^{⌈k/2⌉ - i - 1 + r},
/// ```
/// with `α = 0` outside `0 ≤ ⌊k/2⌋ - r ≤ k - i - 1`, and the weak learner is
/// called on the normalized weights. This is the probability that, if each
/// remaining hypothesis were correct independently with probability `p`,
/// example `n` would end with exactly `⌊k/2⌋` correct votes: the examples
/// whose final vote is still undecided get the weight. The output is the
/// [`MajorityVote`] of the `k` hypotheses' predicted labels.
///
/// Here `k = ⌈ln(1/ε) / (2γ²)⌉` for the target training error
/// `ε = tolerance`.
///
/// Implementation choices (not part of the original method):
/// - "Correct" means [`Classifier::predict`] equals the label, so
///   confidence-rated hypotheses vote with their sign (`0` counts as `+1`).
/// - The weights are computed in log space with `ln Γ` (from `libm`).
/// - If every example has weight `0` (its final vote is decided whatever
///   the remaining hypotheses predict), training stops early; the majority
///   vote of the hypotheses so far is then correct on exactly the examples
///   whose full vote would be.
///
/// # Theoretical guarantee
/// With the potential `Φ_i(r) = P[r + Bin(k - i, p) ≤ ⌊k/2⌋]` (the
/// probability that an example with `r` correct votes after `i` rounds ends
/// with at most `⌊k/2⌋`), the weights are `α_r^i = Φ_{i+1}(r) - Φ_{i+1}(r+1)`.
/// If every weak hypothesis has weighted error at most `1/2 - γ` under the
/// distribution it was trained on, the average potential does not increase,
/// so the majority vote misclassifies at most a fraction
/// `Φ_0(0) = P[Bin(k, p) ≤ ⌊k/2⌋] ≤ exp(-2γ²k) ≤ ε` of the training examples
/// (ties at `k/2` count as errors in this bound).
///
/// # Limitations
/// - BBM is not adaptive: `γ` must be known in advance, and a weak
///   hypothesis much better than `1/2 - γ` gets the same vote as a barely
///   good one. If the assumption fails, the bound does not apply.
/// - `k` grows like `γ⁻² ln(1/ε)`.
/// - The analysis in Freund (1995) is for the boosting-by-filtering setting;
///   this implementation reweights a fixed sample, for which the potential
///   argument above gives the training-error bound.
/// - Labels must be `-1` or `+1` and both classes must be present.
///
/// # Parameters
/// | Builder method | Default | Meaning |
/// | --- | --- | --- |
/// | [`gamma`](BoostByMajority::gamma) | required | Assumed advantage `γ`: weighted error at most `1/2 - γ`. |
/// | [`tolerance`](BoostByMajority::tolerance) | `0.01` | Target training error `ε`; sets `k`. |
///
/// # Reference
/// Yoav Freund.
/// *Boosting a Weak Learning Algorithm by Majority.*
/// Information and Computation, 121(2):256–285, 1995.
/// <https://doi.org/10.1006/inco.1995.1136>
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
/// // γ = 0.2, ε = 0.1: k = ⌈ln 10 / 0.08⌉ = 29 rounds.
/// let mut booster = BoostByMajority::init(&sample).gamma(0.2).tolerance(0.1);
/// assert_eq!(booster.max_loop(), 29);
/// let f = booster.run(&tree);
/// assert!(f.hypotheses.len() <= 29);
/// assert!(f.confidence_all(&sample).iter().all(|c| (-1.0..=1.0).contains(c)));
/// ```
pub struct BoostByMajority<'a, F> {
    // Training sample
    sample: &'a Sample,

    // Assumed advantage γ.
    gamma: Option<f64>,

    // Target training error ε.
    tolerance: f64,

    // Number of rounds k, set by `preprocess`.
    n_rounds: usize,

    // r_n: correct votes of example n so far.
    correct: Vec<usize>,

    // Hypotheses obtained by the weak learner.
    hypotheses: Vec<F>,

    // Whether every vote was decided before round k.
    decided_early: bool,
}

impl<'a, F> BoostByMajority<'a, F> {
    /// Constructs BBM for `sample` with target training error `0.01`. Set
    /// the advantage with [`BoostByMajority::gamma`] before training.
    ///
    /// Time complexity: `O(1)`.
    ///
    /// # Examples
    /// See [`BoostByMajority`].
    pub fn init(sample: &'a Sample) -> Self {
        Self {
            sample,
            gamma: None,
            tolerance: 0.01,
            n_rounds: 0,
            correct: Vec::new(),
            hypotheses: Vec::new(),
            decided_early: false,
        }
    }

    /// Sets the assumed advantage `γ`: every weak hypothesis is assumed to
    /// have weighted error at most `1/2 - γ`.
    ///
    /// # Panics
    /// Panics unless `0 < gamma < 1/2`.
    ///
    /// # Examples
    /// See [`BoostByMajority`].
    pub fn gamma(mut self, gamma: f64) -> Self {
        assert!(
            gamma > 0.0 && gamma < 0.5,
            "BBM needs an advantage gamma in (0, 1/2)"
        );
        self.gamma = Some(gamma);
        self
    }

    /// Sets the target training error `ε`, which sets the number of rounds
    /// `k = ⌈ln(1/ε) / (2γ²)⌉`.
    ///
    /// # Panics
    /// Panics unless `0 < tolerance < 1`.
    ///
    /// # Examples
    /// See [`BoostByMajority`].
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        assert!(
            tolerance > 0.0 && tolerance < 1.0,
            "BBM tolerance must lie in (0, 1)"
        );
        self.tolerance = tolerance;
        self
    }

    /// Returns the number of rounds `k = ⌈ln(1/ε) / (2γ²)⌉` (at least `1`).
    ///
    /// # Panics
    /// Panics if `γ` is not set.
    ///
    /// # Examples
    /// See [`BoostByMajority`].
    pub fn max_loop(&self) -> usize {
        let gamma = self
            .gamma
            .expect("BBM needs the advantage gamma; set it with `gamma`");
        ((1.0 / self.tolerance).ln() / (2.0 * gamma * gamma))
            .ceil()
            .max(1.0) as usize
    }

    /// Returns whether the last run stopped before round `k` because every
    /// example's final vote was already decided (all weights `0`).
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
    /// // k = 29; the separating stump is correct on every example, so after
    /// // 15 rounds every example has a majority of correct votes.
    /// let mut booster = BoostByMajority::init(&sample).gamma(0.2).tolerance(0.1);
    /// let f = booster.run(&tree);
    /// assert!(booster.decided_early());
    /// assert_eq!(f.hypotheses.len(), 15);
    /// ```
    pub fn decided_early(&self) -> bool {
        self.decided_early
    }
}

/// `ln α_r^i` for `k` rounds, `i` finished rounds, `r` correct votes, and
/// `p = 1/2 + γ`; `-∞` outside the support.
fn log_weight(k: usize, i: usize, r: usize, p: f64) -> f64 {
    let remaining = k - i - 1;
    let half = k / 2;
    if r > half || half - r > remaining {
        return f64::NEG_INFINITY;
    }
    let needed = half - r;
    log_binomial(remaining, needed)
        + needed as f64 * p.ln()
        + (remaining - needed) as f64 * (1.0 - p).ln()
}

/// `ln C(n, j)` via `ln Γ`.
fn log_binomial(n: usize, j: usize) -> f64 {
    let lg = |x: usize| libm::lgamma(x as f64 + 1.0);
    lg(n) - lg(j) - lg(n - j)
}

impl<F> Booster<F> for BoostByMajority<'_, F>
where
    F: Classifier + Clone,
{
    type Output = MajorityVote<F>;

    fn name(&self) -> &str {
        "Boost-by-majority"
    }

    fn info(&self) -> Option<Vec<(&str, String)>> {
        let (n_examples, n_feature) = self.sample.shape();
        let gamma = self.gamma.map_or("-".to_string(), |g| g.to_string());
        let rounds = self
            .gamma
            .map_or("-".to_string(), |_| self.max_loop().to_string());
        Some(vec![
            ("# of examples", n_examples.to_string()),
            ("# of features", n_feature.to_string()),
            ("Gamma", gamma),
            ("Tolerance", self.tolerance.to_string()),
            ("Rounds k", rounds),
        ])
    }

    fn preprocess(&mut self) {
        self.n_rounds = self.max_loop();
        self.sample.is_valid_binary_instance();
        self.correct = vec![0; self.sample.shape().0];
        self.hypotheses = Vec::new();
        self.decided_early = false;
    }

    fn boost<W>(&mut self, weak_learner: &W, iteration: usize) -> ControlFlow<usize>
    where
        W: WeakLearner<Hypothesis = F>,
    {
        let k = self.n_rounds;
        if iteration > k || self.decided_early {
            return ControlFlow::Break(iteration - 1);
        }
        let i = iteration - 1;
        let p = 0.5 + self.gamma.expect("call preprocess before boost");
        let log_weights: Vec<f64> = self
            .correct
            .iter()
            .map(|&r| log_weight(k, i, r, p))
            .collect();
        if log_weights.iter().all(|l| *l == f64::NEG_INFINITY) {
            self.decided_early = true;
            return ControlFlow::Break(iteration - 1);
        }
        let dist = helpers::softmax(&log_weights);
        let h = weak_learner.produce(self.sample, &dist);
        let target = self.sample.target();
        self.correct.iter_mut().enumerate().for_each(|(n, r)| {
            if h.predict(self.sample, n) as f64 == target[n] {
                *r += 1;
            }
        });
        self.hypotheses.push(h);
        ControlFlow::Continue(())
    }

    fn postprocess(&mut self) -> Self::Output {
        MajorityVote::new(self.hypotheses.clone())
    }
}

impl<H> CurrentHypothesis for BoostByMajority<'_, H>
where
    H: Classifier + Clone,
{
    type Output = MajorityVote<H>;
    fn current_hypothesis(&self) -> Self::Output {
        MajorityVote::new(self.hypotheses.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DecisionTreeBuilder;
    use crate::DecisionTreeClassifier;
    use std::io::BufReader;

    /// `Φ_i(r) = P[r + Bin(k - i, p) ≤ ⌊k/2⌋]`.
    fn potential(k: usize, i: usize, r: usize, p: f64) -> f64 {
        let n = k - i;
        let half = k / 2;
        if r > half {
            return 0.0;
        }
        (0..=(half - r).min(n))
            .map(|j| {
                (log_binomial(n, j) + j as f64 * p.ln() + (n - j) as f64 * (1.0 - p).ln()).exp()
            })
            .sum()
    }

    #[test]
    fn weights_are_potential_differences() {
        // α_r^i = Φ_{i+1}(r) - Φ_{i+1}(r + 1) for every support point.
        let (k, p) = (9, 0.6);
        for i in 0..k {
            for r in 0..=i {
                let alpha = log_weight(k, i, r, p).exp();
                let diff = potential(k, i + 1, r, p) - potential(k, i + 1, r + 1, p);
                assert!((alpha - diff).abs() < 1e-14, "i {i}, r {r}");
            }
        }
        // Binomial coefficients: ln C(5, 2) = ln 10.
        assert!((log_binomial(5, 2) - 10f64.ln()).abs() < 1e-14);
    }

    #[test]
    fn the_average_potential_does_not_increase_when_the_edge_holds() {
        let csv = b"x,y,class\n\
            -2,1,-1\n-1,-2,-1\n1,-3,-1\n-3,2,-1\n0,-1,-1\n-2,-1,-1\n\
            2,1,1\n1,2,1\n-1,3,1\n3,-1,1\n0,2,1\n2,2,1\n";
        let sample = Sample::from_reader(BufReader::new(&csv[..]), true)
            .unwrap()
            .set_target("class");
        let tree = DecisionTreeBuilder::new(&sample).max_depth(2).build();
        let gamma = 0.1;
        let mut booster = BoostByMajority::init(&sample).gamma(gamma).tolerance(0.2);
        booster.preprocess();
        let (k, p) = (booster.n_rounds, 0.5 + gamma);
        let average = |b: &BoostByMajority<'_, DecisionTreeClassifier>, i: usize| {
            b.correct
                .iter()
                .map(|&r| potential(k, i, r, p))
                .sum::<f64>()
                / 12.0
        };
        let mut checked = 0;
        for iteration in 1..=k {
            let before = average(&booster, iteration - 1);
            let log_weights: Vec<f64> = booster
                .correct
                .iter()
                .map(|&r| log_weight(k, iteration - 1, r, p))
                .collect();
            if booster.boost(&tree, iteration).is_break() {
                break;
            }
            // Weighted error of the new hypothesis under its distribution.
            let dist = helpers::softmax(&log_weights);
            let h = booster.hypotheses.last().unwrap();
            let error: f64 = (0..12)
                .filter(|&n| h.predict(&sample, n) as f64 != sample.target()[n])
                .map(|n| dist[n])
                .sum();
            if error <= 0.5 - gamma {
                checked += 1;
                assert!(average(&booster, iteration) <= before + 1e-12);
            }
        }
        assert!(checked > 0);
    }

    #[test]
    fn missing_gamma_panics() {
        let sample = Sample::dummy(2);
        let result = std::panic::catch_unwind(|| {
            BoostByMajority::<DecisionTreeClassifier>::init(&sample).preprocess()
        });
        assert!(result.is_err());
        let init = || BoostByMajority::<DecisionTreeClassifier>::init(&sample);
        assert!(std::panic::catch_unwind(|| init().gamma(0.5)).is_err());
        assert!(std::panic::catch_unwind(|| init().tolerance(1.0)).is_err());
    }
}
