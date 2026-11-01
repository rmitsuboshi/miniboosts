use crate::core::{Classifier, Sample, helpers};
use crate::optimization::{ObjectiveFunction, SoftMarginObjective};

/// An objective evaluated on a whole classifier for the
/// `ObjectiveValue` column of a [`Logger`](crate::Logger) log.
/// # Examples
/// A custom objective: the average exponential loss
/// `(1/m) Σ_i exp(-y_i f(x_i))` on the training sample. The snapshots of the
/// built-in boosters are [`WeightedMajority`](crate::WeightedMajority)
/// ensembles whose weights are normalized to `Σ_t |w_t| = 1`, so for
/// [`AdaBoost`](crate::AdaBoost) this is the loss of the normalized ensemble,
/// not of the unnormalized combination that AdaBoost greedily minimizes; it
/// need not decrease monotonically.
/// ```
/// # use std::io::BufReader;
/// # use miniboosts::{prelude::*, Sample};
/// # let csv = b"x,y,class\n\
/// #     -2.0,0.5,-1\n-1.5,-1.0,-1\n-1.0,0.8,-1\n-0.5,-1.5,-1\n\
/// #     0.5,-2.0,-1\n1.0,-1.8,-1\n-1.8,1.2,-1\n0.2,-0.9,-1\n\
/// #     2.0,-0.5,1\n1.5,1.0,1\n1.0,-0.2,1\n0.5,1.5,1\n\
/// #     -0.5,2.0,1\n-1.0,1.8,1\n1.8,0.4,1\n0.3,0.9,1\n";
/// # let sample = Sample::from_reader(BufReader::new(&csv[..]), true)?.set_target("class");
/// # let n = sample.shape().0;
/// # let ix: Vec<usize> = (0..n).filter(|i| i % 4 != 0).chain((0..n).filter(|i| i % 4 == 0)).collect();
/// # let (train, test) = sample.split(&ix, n - n / 4, n);
/// # fn error_rate<H: Classifier>(sample: &Sample, h: &H) -> f64 {
/// #     let wrong = h.predict_all(sample).iter().zip(sample.target())
/// #         .filter(|(p, y)| **p as f64 != **y).count();
/// #     wrong as f64 / sample.shape().0 as f64
/// # }
/// # let path = std::env::temp_dir()
/// #     .join(format!("miniboosts-doc-objective-trait-{}.csv", std::process::id()));
/// use miniboosts::{Logger, LoggingObjective};
///
/// struct ExponentialLoss;
///
/// impl LoggingObjective for ExponentialLoss {
///     fn name(&self) -> String {
///         "Exponential loss".to_string()
///     }
///
///     fn objective_value<H: Classifier>(&self, sample: &Sample, f: &H) -> f64 {
///         let m = sample.shape().0 as f64;
///         f.confidence_all(sample).iter().zip(sample.target())
///             .map(|(fx, y)| (-y * fx).exp())
///             .sum::<f64>() / m
///     }
/// }
///
/// let booster = AdaBoost::init(&train).tolerance(0.1);
/// let tree = DecisionTreeBuilder::new(&train).max_depth(1).build();
/// let mut logger = Logger::new(booster, tree, ExponentialLoss, error_rate, &train, &test)
///     .max_iterations(10)
///     .print_every(usize::MAX);
/// logger.run(&path)?;
/// let log = std::fs::read_to_string(&path)?;
/// std::fs::remove_file(&path)?;
///
/// for line in log.lines().skip(1) {
///     let loss: f64 = line.split(',').next().unwrap().parse()?;
///     // Confidences of a normalized ensemble lie in [-1, 1].
///     assert!((-1f64).exp() - 1e-12 <= loss && loss <= 1f64.exp() + 1e-12);
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait LoggingObjective {
    /// Name shown in the logger's settings panel.
    fn name(&self) -> String;

    /// Objective value of classifier `f` on `sample`. The logger calls this
    /// with the training sample after every boosting step.
    fn objective_value<H: Classifier>(&self, sample: &Sample, f: &H) -> f64;
}

/// The soft-margin objective of a classifier, for logging.
///
/// For a classifier `f` with confidences `f(x_i)`
/// ([`Classifier::confidence`]) it records
/// ```txt
/// min_{d ∈ Δ_{m,ν}} Σ_i d_i y_i f(x_i)
///   = max_{ρ, ξ ≥ 0} { ρ - (1/ν) Σ_i ξ_i : y_i f(x_i) ≥ ρ - ξ_i ∀i },
/// ```
/// the soft-margin LP objective (LPBoost, ERLPBoost, MLPBoost, ...) at the
/// fixed combined hypothesis `f`. Here `Δ_{m,ν}` is the capped simplex
/// `{ d : Σ_i d_i = 1, 0 ≤ d_i ≤ 1/ν }`. It equals the average of the `ν`
/// smallest margins (fractionally weighted for non-integral `ν`).
///
/// This is a thin adapter over
/// [`SoftMarginObjective`], which takes negated
/// margins rather than a classifier. The value is meaningful as the
/// soft-margin objective only when the confidences are those of a convex
/// combination of hypotheses (weights nonnegative and summing to one), as for
/// the ensembles of the soft-margin boosters; it is not rescaled here.
///
/// Time complexity: `O(m log m)` plus `m` confidence evaluations.
///
/// # Panics
/// `objective_value` panics if a margin is NaN. `ν` is not validated; use
/// `1 ≤ ν ≤ m` with the same `ν` as the booster.
///
/// # Examples
/// For margins `(1, 0.2, -0.4, 2)` the value with `ν = 1` is the smallest
/// margin, and with `ν = 2` the mean of the two smallest:
/// ```
/// use miniboosts::{Classifier, LoggingObjective, LoggingSoftMarginObjective, Sample};
///
/// /// A classifier with fixed margins `y_i f(x_i)`, one per example.
/// struct Fixed([f64; 4]);
///
/// impl Classifier for Fixed {
///     fn confidence(&self, sample: &Sample, row: usize) -> f64 {
///         self.0[row] * sample.target()[row]
///     }
/// }
///
/// let sample = Sample::dummy(4);
/// let f = Fixed([1.0, 0.2, -0.4, 2.0]);
///
/// let hard = LoggingSoftMarginObjective::new(1.0);
/// assert!((hard.objective_value(&sample, &f) - (-0.4)).abs() < 1e-12);
///
/// let soft = LoggingSoftMarginObjective::new(2.0);
/// assert!((soft.objective_value(&sample, &f) - (-0.1)).abs() < 1e-12);
/// ```
pub struct LoggingSoftMarginObjective(SoftMarginObjective);

impl LoggingSoftMarginObjective {
    /// Creates the objective with capping parameter `ν`.
    ///
    /// # Examples
    /// ```
    /// use miniboosts::{LoggingObjective, LoggingSoftMarginObjective};
    ///
    /// let objective = LoggingSoftMarginObjective::new(2.0);
    /// // The name shown in the logger's settings panel.
    /// assert_eq!(objective.name(), "Soft Margin Objective");
    /// ```
    pub fn new(nu: f64) -> Self {
        let obj = SoftMarginObjective::new(nu);
        Self(obj)
    }
}

impl LoggingObjective for LoggingSoftMarginObjective {
    fn name(&self) -> String {
        self.0.name().to_string()
    }

    fn objective_value<H: Classifier>(&self, sample: &Sample, f: &H) -> f64 {
        let neg_margins = helpers::margins(sample, f)
            .map(|yf| -yf)
            .collect::<Vec<_>>();
        self.0.objective_value(&neg_margins[..])
    }
}
