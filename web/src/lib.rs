//! Thin WebAssembly boundary for the interactive demo: JavaScript supplies
//! labeled 2-D points, Rust trains a booster with the chosen weak learner and
//! returns the ensemble's confidence on a square grid over `[-1, 1]^2`.
use js_sys::Reflect;
use miniboosts::{
    AdditiveModel, ClassMember, Classifier, DecisionTree, DecisionTreeClassifier, FiniteClass,
    GbmLoss, LinearClassifier, LinearOutput, MajorityVote, NaiveAggregation, NaiveBayesClassifier,
    ObjectiveFunction, RbfHypothesis, RegressionTree, RegressionTreeBuilder,
    RegressionTreeRegressor, Regressor, Sample, ScoreLink, SoftMarginObjective, WeightedMajority,
    core::{CurrentHypothesis, helpers::log_weighted_edge_coefficient},
    prelude::*,
};
use std::{
    cell::{Cell, RefCell},
    ops::ControlFlow,
};

/// Side length of the coarse grid stored for each timeline frame.
const FRAME_RESOLUTION: usize = 41;
/// At most this many rounds keep training-sample confidences (loss curve).
const CURVE_CAP: usize = 200;
/// At most this many rounds keep a grid and the example weights (slider).
const FRAME_CAP: usize = 60;
use wasm_bindgen::prelude::*;

/// Result of one training run.
#[wasm_bindgen]
pub struct Fit {
    grid: Vec<f64>,
    rounds: usize,
    stop_reason: String,
    train_error: f64,
    test_error: f64,
    note: String,
    timeline: Timeline,
}

#[wasm_bindgen]
impl Fit {
    /// Ensemble confidence on the grid, row-major from `(x, y) = (-1, 1)`
    /// (top-left) to `(1, -1)` (bottom-right).
    #[wasm_bindgen(getter)]
    pub fn grid(&self) -> Vec<f64> {
        self.grid.clone()
    }
    /// Number of weak hypotheses produced. A round that stops before calling
    /// the weak learner is not counted.
    #[wasm_bindgen(getter)]
    pub fn rounds(&self) -> usize {
        self.rounds
    }
    /// `"algorithm_stop"`, `"max_iterations"`, for SmoothBoost
    /// `"round_budget"` (the paper's round bound was reached without the
    /// stopping test firing, so its margin guarantee does not apply), or for
    /// BrownBoost `"numerical_stop"` (stopped before the remaining time ran
    /// out; see the note).
    #[wasm_bindgen(getter, js_name = stopReason)]
    pub fn stop_reason(&self) -> String {
        self.stop_reason.clone()
    }
    /// Fraction of misclassified training points.
    #[wasm_bindgen(getter, js_name = trainError)]
    pub fn train_error(&self) -> f64 {
        self.train_error
    }
    /// Fraction of misclassified test points, or NaN without test points.
    #[wasm_bindgen(getter, js_name = testError)]
    pub fn test_error(&self) -> f64 {
        self.test_error
    }
    /// Algorithm-specific summary (empty for most algorithms).
    #[wasm_bindgen(getter)]
    pub fn note(&self) -> String {
        self.note.clone()
    }
    /// Rounds `t` (number of weak hypotheses produced) whose snapshot
    /// confidences on the training points are in `curveConfidences`.
    #[wasm_bindgen(getter, js_name = curveRounds)]
    pub fn curve_rounds(&self) -> Vec<u32> {
        self.timeline
            .curve
            .items
            .iter()
            .map(|(t, _)| *t as u32)
            .collect()
    }
    /// Confidences `f_t(x_i)` of the snapshot after each round in
    /// `curveRounds`, one row of `n` values per round. Snapshots are the
    /// boosters' combined hypotheses, with weights normalized to `Σ|w| = 1`.
    #[wasm_bindgen(getter, js_name = curveConfidences)]
    pub fn curve_confidences(&self) -> Vec<f64> {
        self.timeline
            .curve
            .items
            .iter()
            .flat_map(|(_, p)| p.conf.iter().copied())
            .collect()
    }
    /// Confidences of the snapshot after each round in `curveRounds` on the
    /// test points, one row per round (empty without test points). The
    /// test points are never used for training.
    #[wasm_bindgen(getter, js_name = curveTestConfidences)]
    pub fn curve_test_confidences(&self) -> Vec<f64> {
        self.timeline
            .curve
            .items
            .iter()
            .flat_map(|(_, p)| p.test_conf.iter().copied())
            .collect()
    }
    /// The algorithm's own objective (see `Objective`) after each round in
    /// `curveRounds`, or NaN for algorithms without one.
    #[wasm_bindgen(getter, js_name = curveObjective)]
    pub fn curve_objective(&self) -> Vec<f64> {
        self.timeline
            .curve
            .items
            .iter()
            .map(|(_, p)| p.objective)
            .collect()
    }
    /// Rounds kept as timeline frames.
    #[wasm_bindgen(getter, js_name = frameRounds)]
    pub fn frame_rounds(&self) -> Vec<u32> {
        self.timeline
            .frames
            .items
            .iter()
            .map(|(t, _)| *t as u32)
            .collect()
    }
    /// Snapshot confidences on the `frameResolution²` grid for each frame,
    /// in the same layout as `grid`.
    #[wasm_bindgen(getter, js_name = frameGrids)]
    pub fn frame_grids(&self) -> Vec<f32> {
        self.timeline
            .frames
            .items
            .iter()
            .flat_map(|(_, f)| f.grid.iter().copied())
            .collect()
    }
    /// For each frame, the distribution over the training points that the
    /// booster passed to the weak learner in that round.
    #[wasm_bindgen(getter, js_name = frameDists)]
    pub fn frame_dists(&self) -> Vec<f64> {
        self.timeline
            .frames
            .items
            .iter()
            .flat_map(|(_, f)| f.dist.iter().copied())
            .collect()
    }
    /// Side length of the frame grids.
    #[wasm_bindgen(getter, js_name = frameResolution)]
    pub fn frame_resolution(&self) -> usize {
        FRAME_RESOLUTION
    }
    /// Name of a step after the last round shown as the final frame, such
    /// as `"sparsification"`, or empty.
    #[wasm_bindgen(getter, js_name = finalStep)]
    pub fn final_step(&self) -> String {
        self.timeline.final_step.into()
    }
}

/// Installs a panic hook so that Rust panics show their message in the
/// browser console. A panic aborts the wasm instance; the caller should
/// discard it (e.g. restart the worker).
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Trains `algorithm` with the weak learner `learner` on the points
/// `(xs[i], ys[i])` with labels `labels[i] ∈ {-1, +1}` and evaluates the
/// result on a `resolution × resolution` grid and on the held-out test
/// points `(test_xs[i], test_ys[i])` with labels `test_labels[i]` (possibly
/// none), which are not used for training.
///
/// `booster` holds the booster parameters: `maxIterations`, `tolerance`,
/// `nu` (capping, `1 ≤ nu ≤ n`), `gamma` and `kappa` (SmoothBoost),
/// `nHypotheses` (`T`, `0` for none) and `gap` (`v`, NaN for none) for
/// SparsiBoost, and `brownTolerance` (`ε`) and `edgeThreshold` (`ν`) for
/// BrownBoost. `weak` holds the learner parameters: `maxDepth` and `splitBy`
/// (decision tree), `regularization` and `output` (logistic regression),
/// `varSmoothing` (naive Bayes), and `bandwidth` (RBF). Only the parameters
/// of the chosen booster and learner are read.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn train(
    algorithm: &str,
    learner: &str,
    xs: &[f64],
    ys: &[f64],
    labels: &[f64],
    test_xs: &[f64],
    test_ys: &[f64],
    test_labels: &[f64],
    booster: &JsValue,
    weak: &JsValue,
    resolution: usize,
) -> Result<Fit, JsError> {
    let n = labels.len();
    if xs.len() != n || ys.len() != n {
        return Err(JsError::new("coordinate and label arrays differ in length"));
    }
    if labels.iter().any(|&y| y != -1.0 && y != 1.0)
        || !labels.contains(&-1.0)
        || !labels.contains(&1.0)
    {
        return Err(JsError::new("place at least one point of each class"));
    }
    let n_test = test_labels.len();
    if test_xs.len() != n_test || test_ys.len() != n_test {
        return Err(JsError::new(
            "test coordinate and label arrays differ in length",
        ));
    }
    if test_labels.iter().any(|&y| y != -1.0 && y != 1.0) {
        return Err(JsError::new("test labels must be -1 or +1"));
    }
    if !(2..=400).contains(&resolution) {
        return Err(JsError::new("resolution must lie in [2, 400]"));
    }

    let rows = xs
        .iter()
        .zip(ys)
        .map(|(&x, &y)| vec![x, y])
        .collect::<Vec<_>>();
    let sample = Sample::from_dense(&rows, Some(labels)).map_err(|e| JsError::new(&e))?;
    // Snapshots are evaluated on the training rows followed by the test rows.
    let mut all_rows = rows;
    all_rows.extend(test_xs.iter().zip(test_ys).map(|(&x, &y)| vec![x, y]));
    let all_labels = [labels, test_labels].concat();
    let points = Sample::from_dense(&all_rows, Some(&all_labels)).map_err(|e| JsError::new(&e))?;
    let p = Params(booster);
    let w = Params(weak);

    let frame_sample = square_grid(FRAME_RESOLUTION)?;
    let ctx = Context {
        sample: &sample,
        points: &points,
        frame_sample: &frame_sample,
    };
    // Gradient boosting and LogitBoost fit real-valued weak hypotheses by
    // least squares, so they take regression trees; the other boosters take
    // classifiers.
    let outcome = if matches!(algorithm, "gradient_boosting" | "logitboost") {
        if learner != "regression_tree" {
            return Err(JsError::new(
                "gradient boosting and LogitBoost need the regression-tree weak learner",
            ));
        }
        let tree = RegressionTreeBuilder::new(&sample)
            .max_depth(w.count("maxDepth", 1)?)
            .build();
        fit_regression(algorithm, &ctx, &tree, &p)?
    } else {
        if learner == "regression_tree" {
            return Err(JsError::new(
                "regression trees are only used by gradient boosting and LogitBoost",
            ));
        }
        let learner = match learner {
            "decision_tree" => {
                let split = match w.text("splitBy")?.as_str() {
                    "entropy" => SplitBy::Entropy,
                    "edge" => SplitBy::Edge,
                    "gini" => SplitBy::Gini,
                    "twoing" => SplitBy::Twoing,
                    _ => return Err(JsError::new("unknown split criterion")),
                };
                Learner::Tree(
                    DecisionTreeBuilder::new(&sample)
                        .max_depth(w.count("maxDepth", 1)?)
                        .split_by(split)
                        .build(),
                )
            }
            "logistic_regression" => {
                let output = match w.text("output")?.as_str() {
                    "sign" => LinearOutput::Sign,
                    "tanh" => LinearOutput::Tanh,
                    _ => return Err(JsError::new("unknown logistic-regression output")),
                };
                Learner::Logistic(
                    LogisticRegression::new()
                        .regularization(w.positive("regularization")?)
                        .output(output),
                )
            }
            "bounded_linear" => Learner::Linear(BoundedLinear::new()),
            "naive_bayes" => {
                Learner::Bayes(NaiveBayes::new().var_smoothing(w.positive("varSmoothing")?))
            }
            "rbf" => Learner::Rbf(
                RbfClassBuilder::new(&sample)
                    .bandwidth(w.positive("bandwidth")?)
                    .build(),
            ),
            _ => return Err(JsError::new("unknown weak learner")),
        };
        fit(algorithm, &ctx, &learner, &p)?
    };
    let Outcome {
        model,
        rounds,
        stop: stop_reason,
        note,
        timeline,
    } = outcome;

    let predictions = model.predict_all(&points);
    let error = |predicted: &[i64], labels: &[f64]| {
        predicted
            .iter()
            .zip(labels)
            .filter(|&(&p, &y)| p as f64 != y)
            .count() as f64
            / labels.len() as f64
    };
    let train_error = error(&predictions[..n], labels);
    let test_error = error(&predictions[n..], test_labels);

    let grid = model.confidence_all(&square_grid(resolution)?);
    if grid.iter().any(|v| !v.is_finite()) {
        return Err(JsError::new("ensemble produced non-finite confidence"));
    }

    Ok(Fit {
        grid,
        rounds,
        stop_reason: stop_reason.into(),
        train_error,
        test_error,
        note,
        timeline,
    })
}

/// The points of a `resolution × resolution` grid over `[-1, 1]^2`,
/// row-major from `(-1, 1)` (top-left) to `(1, -1)` (bottom-right).
fn square_grid(resolution: usize) -> Result<Sample, JsError> {
    let step = 2.0 / (resolution - 1) as f64;
    let rows = (0..resolution)
        .flat_map(|r| {
            let y = 1.0 - r as f64 * step;
            (0..resolution).map(move |c| vec![-1.0 + c as f64 * step, y])
        })
        .collect::<Vec<_>>();
    Sample::from_dense(&rows, None).map_err(|e| JsError::new(&e))
}

/// One of the demo's weak learners. Dispatching through an enum compiles
/// each booster once rather than once per learner, which keeps the wasm
/// module small.
enum Learner<'a> {
    Tree(DecisionTree<'a>),
    Logistic(LogisticRegression),
    Linear(BoundedLinear),
    Bayes(NaiveBayes),
    Rbf(FiniteClass<RbfHypothesis>),
}

/// A hypothesis returned by [`Learner`].
#[derive(Clone, PartialEq)]
enum Hypothesis {
    Tree(DecisionTreeClassifier),
    Linear(LinearClassifier),
    Bayes(NaiveBayesClassifier),
    Rbf(ClassMember<RbfHypothesis>),
}

impl Classifier for Hypothesis {
    fn confidence(&self, sample: &Sample, row: usize) -> f64 {
        match self {
            Self::Tree(h) => h.confidence(sample, row),
            Self::Linear(h) => h.confidence(sample, row),
            Self::Bayes(h) => h.confidence(sample, row),
            Self::Rbf(h) => h.confidence(sample, row),
        }
    }
}

impl WeakLearner for Learner<'_> {
    type Hypothesis = Hypothesis;
    fn produce(&self, sample: &Sample, dist: &[f64]) -> Hypothesis {
        match self {
            Self::Tree(l) => Hypothesis::Tree(l.produce(sample, dist)),
            Self::Logistic(l) => Hypothesis::Linear(l.produce(sample, dist)),
            Self::Linear(l) => Hypothesis::Linear(l.produce(sample, dist)),
            Self::Bayes(l) => Hypothesis::Bayes(l.produce(sample, dist)),
            Self::Rbf(l) => Hypothesis::Rbf(l.produce(sample, dist)),
        }
    }
}

/// Named numeric and string parameters read from a JavaScript object.
struct Params<'a>(&'a JsValue);

impl Params<'_> {
    /// Returns the number under `key`, or NaN if it is missing.
    fn num(&self, key: &str) -> f64 {
        Reflect::get(self.0, &JsValue::from_str(key))
            .ok()
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::NAN)
    }

    fn text(&self, key: &str) -> Result<String, JsError> {
        Reflect::get(self.0, &JsValue::from_str(key))
            .ok()
            .and_then(|v| v.as_string())
            .ok_or_else(|| JsError::new(&format!("{key} is missing")))
    }

    fn positive(&self, key: &str) -> Result<f64, JsError> {
        let v = self.num(key);
        if v.is_finite() && v > 0.0 {
            Ok(v)
        } else {
            Err(JsError::new(&format!("{key} must be positive")))
        }
    }

    /// Returns the integer under `key`, which must be at least `min`.
    fn count(&self, key: &str, min: usize) -> Result<usize, JsError> {
        let v = self.num(key);
        if v.is_finite() && v.fract() == 0.0 && v >= min as f64 {
            Ok(v as usize)
        } else {
            Err(JsError::new(&format!("{key} must be an integer ≥ {min}")))
        }
    }
}

struct Outcome {
    model: Box<dyn Classifier>,
    rounds: usize,
    stop: &'static str,
    note: String,
    timeline: Timeline,
}

/// The training sample, the points snapshots are evaluated on (the training
/// rows followed by the test rows), and the coarse grid for timeline frames.
struct Context<'a> {
    sample: &'a Sample,
    points: &'a Sample,
    frame_sample: &'a Sample,
}

impl Context<'_> {
    fn n_train(&self) -> usize {
        self.sample.shape().0
    }
}

/// Entries kept for at most `cap` rounds. When full, every other entry is
/// dropped and the stride doubles, so the kept rounds stay evenly spaced
/// whatever the final number of rounds.
struct Thinned<T> {
    cap: usize,
    stride: usize,
    items: Vec<(usize, T)>,
}

impl<T> Thinned<T> {
    fn new(cap: usize) -> Self {
        Self {
            cap,
            stride: 1,
            items: Vec::new(),
        }
    }

    fn wants(&self, round: usize) -> bool {
        round % self.stride == 0
    }

    fn push(&mut self, round: usize, item: T) {
        self.items.push((round, item));
        if self.items.len() > self.cap {
            self.stride *= 2;
            let stride = self.stride;
            self.items.retain(|(r, _)| r % stride == 0);
        }
    }
}

struct Frame {
    grid: Vec<f32>,
    dist: Vec<f64>,
}

/// The snapshot's confidences on the training and test points and its
/// objective on the training points.
struct CurvePoint {
    conf: Vec<f64>,
    test_conf: Vec<f64>,
    objective: f64,
}

/// Snapshots of the combined hypothesis after selected rounds.
struct Timeline {
    curve: Thinned<CurvePoint>,
    frames: Thinned<Frame>,
    /// If nonempty, the last curve point and frame show this extra step
    /// after the last round (e.g. SparsiBoost's sparsification) and share
    /// the round number of the entry before them.
    final_step: &'static str,
}

/// The quantity an algorithm optimizes, evaluated after each round for the
/// loss chart from the snapshot's confidences `f(x_i)` on the training
/// points. Snapshots are normalized to `Σ_s |w_s| = 1`, so the AdaBoost-type
/// losses rescale them by `S = Σ_s |α_s|` to the combination `F = Σ_s α_s h_s`
/// that the algorithm minimizes over.
#[derive(Clone, Copy)]
enum Objective {
    None,
    /// AdaBoost: `(1/m) Σ_i exp(-y_i F(x_i))`.
    Exponential,
    /// MadaBoost (Domingo and Watanabe): `(1/m) Σ_i φ(y_i F(x_i))` with
    /// `φ(z) = exp(-z)` for `z ≥ 0` and `1 - z` for `z < 0`, whose
    /// derivative gives MadaBoost's capped weights `min(1, exp(-z))`.
    Mada,
    /// `min_{d ∈ Δ_{m,ν}} Σ_i d_i y_i f(x_i)`, maximized by the soft-margin
    /// boosters; `ν = 1` is the minimum margin.
    SoftMargin(f64),
    /// SmoothBoost (Servedio 2003, Figure 1): `(1/m) Σ_j M_{t+1}(j)` with
    /// `N_t(j) = t (y_j f(x_j) - θ)` for the average `f` of `t` hypotheses;
    /// training stops once it drops below `κ`.
    Smooth {
        gamma: f64,
    },
    /// `(1/m) Σ_i ln(1 + exp(-2 y_i F(x_i)))`, the binomial deviance that
    /// LogitBoost and logistic gradient boosting minimize (`F` is half the
    /// log-odds).
    Deviance,
    /// `(1/m) Σ_i exp(-y_i F(x_i))`: gradient boosting with the exponential
    /// loss.
    ScoreExponential,
    /// `(1/m) Σ_i (y_i - F(x_i))² / 2`: gradient boosting with the squared
    /// loss.
    Squared,
}

impl Objective {
    fn needs_alpha(self) -> bool {
        matches!(self, Self::Exponential | Self::Mada)
    }

    /// `conf` are the snapshot's confidences and `score` its score `F`
    /// (only the last three variants read `score`).
    fn value(self, labels: &[f64], conf: &[f64], score: &[f64], alpha_sum: f64, t: usize) -> f64 {
        let m = labels.len() as f64;
        let mean = |loss: &dyn Fn(f64, f64) -> f64| {
            labels
                .iter()
                .zip(score)
                .map(|(&y, &f)| loss(y, f))
                .sum::<f64>()
                / m
        };
        let margins = labels.iter().zip(conf).map(|(y, f)| y * f);
        // `y F(x) = S y f(x)`; an infinite `S` (a hypothesis with edge ±1)
        // sends every nonzero margin to ±∞.
        let scaled = |yf: f64| {
            if alpha_sum.is_finite() || yf == 0.0 {
                alpha_sum * yf
            } else {
                yf.signum() * f64::INFINITY
            }
        };
        match self {
            Self::None => f64::NAN,
            // ln(1 + e^z) without overflow.
            Self::Deviance => mean(&|y, f| {
                let z = -2.0 * y * f;
                z.max(0.0) + (-z.abs()).exp().ln_1p()
            }),
            Self::ScoreExponential => mean(&|y, f| (-y * f).exp()),
            Self::Squared => mean(&|y, f| (y - f).powi(2) / 2.0),
            Self::Exponential => margins.map(|yf| (-scaled(yf)).exp()).sum::<f64>() / m,
            Self::Mada => {
                margins
                    .map(|yf| {
                        let z = scaled(yf);
                        if z >= 0.0 { (-z).exp() } else { 1.0 - z }
                    })
                    .sum::<f64>()
                    / m
            }
            Self::SoftMargin(nu) => {
                let negated = margins.map(|yf| -yf).collect::<Vec<_>>();
                SoftMarginObjective::new(nu).objective_value(&negated)
            }
            Self::Smooth { gamma } => {
                let theta = gamma / (2.0 + gamma);
                margins
                    .map(|yf| {
                        let n = t as f64 * (yf - theta);
                        if n <= 0.0 {
                            1.0
                        } else {
                            (1.0 - gamma).powf(n / 2.0)
                        }
                    })
                    .sum::<f64>()
                    / m
            }
        }
    }
}

/// Trains `algorithm` with `learner`, validating the booster parameters it
/// uses (the library asserts the same conditions; checking here turns them
/// into errors instead of panics).
fn fit(algorithm: &str, ctx: &Context, learner: &Learner, p: &Params) -> Result<Outcome, JsError> {
    let s = ctx.sample;
    let limit = p.count("maxIterations", 1)?;
    let n = s.shape().0;
    let tolerance = || p.positive("tolerance");
    let nu = || {
        let nu = p.num("nu");
        if (1.0..=n as f64).contains(&nu) {
            Ok(nu)
        } else {
            Err(JsError::new(
                "nu must lie between 1 and the number of points",
            ))
        }
    };
    let l = learner;
    Ok(match algorithm {
        "adaboost" => run(
            AdaBoost::init(s).tolerance(tolerance()?),
            ctx,
            l,
            limit,
            true,
            Objective::Exponential,
            no_check,
        )?
        .boxed(),
        "adaboostv" => run(
            AdaBoostV::init(s).tolerance(tolerance()?),
            ctx,
            l,
            limit,
            true,
            Objective::SoftMargin(1.0),
            no_check,
        )?
        .boxed(),
        "madaboost" => run(
            MadaBoost::init(s).tolerance(tolerance()?),
            ctx,
            l,
            limit,
            true,
            Objective::Mada,
            no_check,
        )?
        .boxed(),
        "smoothboost" => {
            let (gamma, kappa) = (p.num("gamma"), p.num("kappa"));
            if !(0.0 < gamma && gamma < 0.5) || !(0.0 < kappa && kappa < 1.0) {
                return Err(JsError::new(
                    "gamma must lie in (0, 0.5) and kappa in (0, 1)",
                ));
            }
            run(
                SmoothBoost::init(s).gamma(gamma).kappa(kappa),
                ctx,
                l,
                limit,
                true,
                Objective::Smooth { gamma },
                |b: &SmoothBoost<_>| Ok(b.reached_round_budget().then_some("round_budget")),
            )?
            .boxed()
        }
        "lpboost" => run(
            LpBoost::init(s).nu(nu()?).tolerance(tolerance()?),
            ctx,
            l,
            limit,
            true,
            Objective::SoftMargin(nu()?),
            no_check,
        )?
        .boxed(),
        "erlpboost" => run(
            ErlpBoost::init(s).nu(nu()?).tolerance(tolerance()?),
            ctx,
            l,
            limit,
            true,
            Objective::SoftMargin(nu()?),
            no_check,
        )?
        .boxed(),
        "corrective_erlpboost" => run(
            CorrectiveErlpBoost::init(s)
                .nu(nu()?)
                .tolerance(tolerance()?),
            ctx,
            l,
            limit,
            true,
            Objective::SoftMargin(nu()?),
            no_check,
        )?
        .boxed(),
        "mlpboost" => run(
            MlpBoost::init(s).nu(nu()?).tolerance(tolerance()?),
            ctx,
            l,
            limit,
            true,
            Objective::SoftMargin(nu()?),
            no_check,
        )?
        .boxed(),
        "softboost" => run(
            SoftBoost::init(s).nu(nu()?).tolerance(tolerance()?),
            ctx,
            l,
            limit,
            true,
            Objective::SoftMargin(nu()?),
            |b: &SoftBoost<_>| {
                b.numerical_stop_reason()
                    .map_or(Ok(None), |r| Err(r.to_owned()))
            },
        )?
        .boxed(),
        "totalboost" => run(
            TotalBoost::init(s).tolerance(tolerance()?),
            ctx,
            l,
            limit,
            true,
            Objective::SoftMargin(1.0),
            |b: &TotalBoost<_>| {
                b.numerical_stop_reason()
                    .map_or(Ok(None), |r| Err(r.to_owned()))
            },
        )?
        .boxed(),
        "arcgv" => {
            let t = run(
                ArcGv::init(s).force_quit_at(limit),
                ctx,
                l,
                limit,
                true,
                Objective::SoftMargin(1.0),
                no_check,
            )?;
            let note = if t.booster.reached_zero_step() {
                "Stopped at a zero step: the weak hypothesis's edge was at most the \
                 current minimum margin, so it was not added."
                    .to_owned()
            } else {
                String::new()
            };
            t.with_note(None, note)
        }
        "bbm" => {
            let (gamma, epsilon) = (p.num("bbmGamma"), p.num("bbmTolerance"));
            if !(0.0 < gamma && gamma < 0.5) || !(0.0 < epsilon && epsilon < 1.0) {
                return Err(JsError::new("BBM needs 0 < γ < 1/2 and 0 < ε < 1"));
            }
            let booster = BoostByMajority::init(s).gamma(gamma).tolerance(epsilon);
            let k = booster.max_loop();
            let t = run(booster, ctx, l, limit, true, Objective::None, no_check)?;
            let early = if t.booster.decided_early() {
                "; stopped early because every example's final vote was decided"
            } else {
                ""
            };
            let note = format!("k = ⌈ln(1/ε) / (2γ²)⌉ = {k} rounds planned{early}.");
            t.with_note(None, note)
        }
        "robustboost" => {
            let (epsilon, theta, sigma) = (
                p.num("robustTolerance"),
                p.num("robustTheta"),
                p.num("robustSigma"),
            );
            if !(0.0 < epsilon && epsilon < 1.0)
                || !(theta.is_finite() && theta >= 0.0)
                || !(sigma.is_finite() && sigma > 0.0)
            {
                return Err(JsError::new(
                    "RobustBoost needs 0 < ε < 1, θ ≥ 0, and σ_f > 0",
                ));
            }
            let booster = RobustBoost::init(s)
                .tolerance(epsilon)
                .theta(theta)
                .sigma_f(sigma);
            let t = run(booster, ctx, l, limit, true, Objective::None, no_check)?;
            let remaining = t.booster.remaining_time().unwrap_or(1.0);
            // The guarantee needs the clock to reach 1 (remaining time 0).
            let (stop, note) = match t.booster.numerical_stop_reason() {
                Some(reason) => (
                    Some("numerical_stop"),
                    format!(
                        "Stopped early with remaining time {remaining:.3}: {reason}. \
                         The guarantee does not apply."
                    ),
                ),
                None if remaining <= 0.0 => (
                    None,
                    "The clock reached 1, so RobustBoost's guarantee applies.".to_owned(),
                ),
                None => (
                    None,
                    format!(
                        "Remaining time {remaining:.3}; the guarantee needs the clock to reach 1."
                    ),
                ),
            };
            t.with_note(stop, note)
        }
        "graph_separation" => run(
            GraphSeparationBoosting::init(s),
            ctx,
            l,
            limit,
            true,
            Objective::None,
            no_check,
        )?
        .boxed(),
        "sparsiboost" => {
            let t = p.num("nHypotheses");
            let gap = p.num("gap");
            if !(gap.is_nan() || (gap.is_finite() && gap > 0.0)) {
                return Err(JsError::new("the target gap v must be positive"));
            }
            // `T` counts as set only if it is at least 1 (NaN and 0 mean unset).
            let has_t = t >= 1.0;
            if !has_t && gap.is_nan() {
                return Err(JsError::new(
                    "SparsiBoost needs a target size T, a target gap v, or both",
                ));
            }
            let mut booster = SparsiBoost::init(s);
            if has_t {
                booster = booster.n_hypotheses(t as usize);
            }
            if !gap.is_nan() {
                booster = booster.tolerance(gap);
            }
            let planned = booster.max_loop();
            // SparsiBoost runs AdaBoostV with these parameters and sparsifies
            // its ensemble once, after the last round. Its own snapshots would
            // redo that O(m R² + R³) step every round, so the timeline comes
            // from the same AdaBoostV run, followed by the sparsified model.
            let stage = run(
                sparsiboost_stage(s, &booster),
                ctx,
                l,
                limit,
                true,
                Objective::SoftMargin(1.0),
                no_check,
            )?;
            let t = run(
                booster,
                ctx,
                l,
                limit,
                false,
                Objective::SoftMargin(1.0),
                no_check,
            )?;
            let mut timeline = stage.timeline;
            timeline.curve.items.extend(t.timeline.curve.items);
            timeline.frames.items.extend(t.timeline.frames.items);
            timeline.final_step = "sparsification";
            let kept = t.model.weights.iter().filter(|w| **w != 0.0).count();
            let change = t.booster.sparsification_error().unwrap_or(f64::NAN);
            let within = match t.booster.sparsification_within_tolerance() {
                Some(true) => " (within v/2)",
                Some(false) => " (exceeds v/2: the cap T forced extra halving)",
                None => "",
            };
            let note = format!(
                "AdaBoostV: {} of {planned} planned rounds; sparsified to {kept} \
                 hypotheses; largest training-margin change {change:.3}{within}",
                t.rounds
            );
            Outcome {
                model: Box::new(t.model),
                rounds: t.rounds,
                stop: t.stop,
                note,
                timeline,
            }
        }
        "brownboost" => {
            let epsilon = p.num("brownTolerance");
            let threshold = p.num("edgeThreshold");
            if !(0.0 < epsilon && epsilon < 1.0) || !(0.0..1.0).contains(&threshold) {
                return Err(JsError::new("BrownBoost needs 0 < ε < 1 and 0 ≤ ν < 1"));
            }
            let booster = BrownBoost::init(s)
                .tolerance(epsilon)
                .edge_threshold(threshold);
            let total = booster.total_time();
            let t = run(booster, ctx, l, limit, true, Objective::None, no_check)?;
            let remaining = t.booster.remaining_time().unwrap_or(total);
            // Freund (2001), Theorem 2 applies only once the remaining time
            // runs out; a numerical stop or the round limit voids it.
            let (stop, note) = match t.booster.numerical_stop_reason() {
                Some(reason) => (
                    "numerical_stop",
                    format!(
                        "Stopped early with remaining time s = {remaining:.3} of \
                         c = {total:.3}: {reason}. The training-error guarantee does \
                         not apply."
                    ),
                ),
                None if remaining <= 0.0 => (
                    t.stop,
                    format!(
                        "Remaining time used up (c = {total:.3}), so the training \
                         error is at most ε = {epsilon} (Theorem 2)."
                    ),
                ),
                None => (
                    t.stop,
                    format!(
                        "Remaining time s = {remaining:.3} of c = {total:.3}; the \
                         training-error guarantee needs s ≤ 0."
                    ),
                ),
            };
            Outcome {
                model: Box::new(t.model),
                rounds: t.rounds,
                stop,
                note,
                timeline: t.timeline,
            }
        }
        _ => return Err(JsError::new("unknown boosting algorithm")),
    })
}

/// The AdaBoostV run inside `booster` (see `SparsiBoost::preprocess`): margin
/// accuracy `adaboostv_tolerance()` for at most `max_loop()` rounds.
fn sparsiboost_stage<'a, H>(s: &'a Sample, booster: &SparsiBoost<'a, H>) -> AdaBoostV<'a, H>
where
    H: Classifier + Clone,
{
    AdaBoostV::init(s)
        .tolerance(booster.adaboostv_tolerance())
        .force_quit_at(booster.max_loop())
}

/// Trains gradient boosting or LogitBoost with regression trees.
fn fit_regression(
    algorithm: &str,
    ctx: &Context,
    tree: &RegressionTree,
    p: &Params,
) -> Result<Outcome, JsError> {
    let s = ctx.sample;
    let limit = p.count("maxIterations", 1)?;
    let stopped = |reason: Option<&str>| match reason {
        Some(reason) => (
            Some("numerical_stop"),
            format!("Stopped before the round limit: {reason}."),
        ),
        None => (None, String::new()),
    };
    Ok(match algorithm {
        "gradient_boosting" => {
            let (loss, objective) = match p.text("gbmLoss")?.as_str() {
                "logistic" => (GbmLoss::Logistic, Objective::Deviance),
                "exponential" => (GbmLoss::Exponential, Objective::ScoreExponential),
                "squared" => (GbmLoss::Squared, Objective::Squared),
                _ => return Err(JsError::new("unknown gradient-boosting loss")),
            };
            let shrinkage = p.num("shrinkage");
            if !(0.0 < shrinkage && shrinkage <= 1.0) {
                return Err(JsError::new("the shrinkage ν must lie in (0, 1]"));
            }
            let booster = GradientBoosting::init(s)
                .loss(loss)
                .shrinkage(shrinkage)
                .force_quit_at(limit);
            let t = run(booster, ctx, tree, limit, true, objective, no_check)?;
            let (stop, note) = stopped(t.booster.numerical_stop_reason());
            t.with_note(stop, note)
        }
        "logitboost" => {
            let z_max = p.num("maxResponse");
            if !(z_max.is_finite() && z_max >= 1.0) {
                return Err(JsError::new("the response bound z_max must be at least 1"));
            }
            let booster = LogitBoost::init(s).max_response(z_max).force_quit_at(limit);
            let t = run(
                booster,
                ctx,
                tree,
                limit,
                true,
                Objective::Deviance,
                no_check,
            )?;
            let (stop, note) = stopped(t.booster.numerical_stop_reason());
            t.with_note(stop, note)
        }
        _ => return Err(JsError::new("unknown boosting algorithm")),
    })
}

fn no_check<B>(_: &B) -> Result<Option<&'static str>, String> {
    Ok(None)
}

/// Counts the weak-learner calls, since boosters may stop at the start of a
/// round without producing a hypothesis.
struct Counting<'a, W: WeakLearner> {
    inner: &'a W,
    calls: Cell<usize>,
    /// The distribution of the latest call, shown as example weights.
    dist: RefCell<Vec<f64>>,
    /// If set, `alpha_sum` accumulates `|α_t|` with the coefficient
    /// `α_t = atanh(Σ_i d_i y_i h_t(x_i))` that AdaBoost and MadaBoost
    /// compute (`log_weighted_edge_coefficient` on the same distribution).
    track_alpha: bool,
    alpha_sum: Cell<f64>,
    /// If set, every hypothesis with its predictions on these points and
    /// this frame grid, for composing snapshots cheaply.
    eval: Option<(&'a Sample, &'a Sample)>,
    cache: RefCell<Cache<W::Hypothesis>>,
}

/// The hypotheses produced so far, in order, with their predictions on the
/// evaluation points (`train`: training rows, then test rows) and the grid.
struct Cache<H> {
    hypotheses: Vec<H>,
    train: Vec<Vec<f64>>,
    // f64, not f32: regression trees can output values such as 1e-162
    // that gradient boosting multiplies by coefficients such as 1e160.
    grid: Vec<Vec<f64>>,
}

impl<H> Default for Cache<H> {
    fn default() -> Self {
        Self {
            hypotheses: Vec::new(),
            train: Vec::new(),
            grid: Vec::new(),
        }
    }
}

/// What snapshots are composed from: the confidences of classifiers and the
/// real-valued outputs of regressors.
trait Scores {
    fn scores(&self, sample: &Sample) -> Vec<f64>;
}

impl Scores for Hypothesis {
    fn scores(&self, sample: &Sample) -> Vec<f64> {
        self.confidence_all(sample)
    }
}

impl Scores for RegressionTreeRegressor {
    fn scores(&self, sample: &Sample) -> Vec<f64> {
        Regressor::predict_all(self, sample)
    }
}

impl<'a, W> WeakLearner for Counting<'a, W>
where
    W: WeakLearner,
    W::Hypothesis: Scores + Clone,
{
    type Hypothesis = W::Hypothesis;
    fn produce(&self, sample: &Sample, dist: &[f64]) -> Self::Hypothesis {
        self.calls.set(self.calls.get() + 1);
        dist.clone_into(&mut self.dist.borrow_mut());
        let h = self.inner.produce(sample, dist);
        let train = h.scores(sample);
        if self.track_alpha {
            let margins = sample
                .target()
                .iter()
                .zip(&train)
                .map(|(y, c)| y * c)
                .collect::<Vec<_>>();
            let logs = dist.iter().map(|d| d.ln()).collect::<Vec<_>>();
            let alpha = log_weighted_edge_coefficient(&logs, &margins);
            self.alpha_sum.set(self.alpha_sum.get() + alpha.abs());
        }
        if let Some((points, frame_sample)) = self.eval {
            let train = h.scores(points);
            let grid = h.scores(frame_sample);
            let mut cache = self.cache.borrow_mut();
            cache.hypotheses.push(h.clone());
            cache.train.push(train);
            cache.grid.push(grid);
        }
        h
    }
}

/// A snapshot evaluated on the evaluation points (training rows, then test
/// rows): its confidences `conf` and the score `F` its algorithm's loss is
/// defined on (equal to `conf` for votes), and, if requested, its
/// confidences on the frame grid.
struct Eval {
    conf: Vec<f64>,
    score: Vec<f64>,
    grid: Option<Vec<f64>>,
}

/// Evaluates a snapshot, composing it from the cached predictions of the
/// weak hypotheses when it is built from the first ones in production order
/// (summed in the same order as its own `confidence`); any other snapshot is
/// evaluated directly.
trait Snapshot<H> {
    fn evaluate(&self, cache: &Cache<H>, ctx: &Context, grid: bool) -> Eval;
}

fn is_cached_prefix<H: PartialEq>(hypotheses: &[H], cache: &Cache<H>) -> bool {
    hypotheses.len() <= cache.hypotheses.len()
        && hypotheses
            .iter()
            .zip(&cache.hypotheses)
            .all(|(a, b)| a == b)
}

/// `Σ_j w_j r_j[i]` for every `i`, summed over `j` in order.
fn combine<R: Copy + Into<f64>>(weights: &[f64], rows: &[Vec<R>], len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| {
            weights
                .iter()
                .zip(rows)
                .map(|(w, r)| w * r[i].into())
                .sum::<f64>()
        })
        .collect()
}

fn direct<F: Classifier>(f: &F, ctx: &Context, grid: bool) -> Eval {
    let conf = f.confidence_all(ctx.points);
    Eval {
        score: conf.clone(),
        conf,
        grid: grid.then(|| f.confidence_all(ctx.frame_sample)),
    }
}

impl<H: Classifier + PartialEq> Snapshot<H> for WeightedMajority<H> {
    /// `Σ_j w_j h_j(x)`, as `WeightedMajority::confidence`.
    fn evaluate(&self, cache: &Cache<H>, ctx: &Context, grid: bool) -> Eval {
        let k = self.hypotheses.len();
        if !is_cached_prefix(&self.hypotheses, cache) {
            return direct(self, ctx, grid);
        }
        let conf = combine(&self.weights, &cache.train[..k], ctx.points.shape().0);
        Eval {
            score: conf.clone(),
            conf,
            grid: grid
                .then(|| combine(&self.weights, &cache.grid[..k], ctx.frame_sample.shape().0)),
        }
    }
}

impl<H: Classifier + PartialEq> Snapshot<H> for MajorityVote<H> {
    /// `(1/T) Σ_j sign(h_j(x))` with `sign(0) = +1`, as
    /// `MajorityVote::confidence`.
    fn evaluate(&self, cache: &Cache<H>, ctx: &Context, grid: bool) -> Eval {
        let k = self.hypotheses.len();
        if k == 0 || !is_cached_prefix(&self.hypotheses, cache) {
            return direct(self, ctx, grid);
        }
        fn vote<R: Copy + Into<f64>>(rows: &[Vec<R>], len: usize) -> Vec<f64> {
            (0..len)
                .map(|i| {
                    let votes: i64 = rows
                        .iter()
                        .map(|r| if r[i].into() >= 0.0 { 1 } else { -1 })
                        .sum();
                    votes as f64 / rows.len() as f64
                })
                .collect()
        }
        let conf = vote(&cache.train[..k], ctx.points.shape().0);
        Eval {
            score: conf.clone(),
            conf,
            grid: grid.then(|| vote(&cache.grid[..k], ctx.frame_sample.shape().0)),
        }
    }
}

impl<H: Classifier> Snapshot<H> for NaiveAggregation<H> {
    fn evaluate(&self, _: &Cache<H>, ctx: &Context, grid: bool) -> Eval {
        direct(self, ctx, grid)
    }
}

impl<H: Regressor + PartialEq> Snapshot<H> for AdditiveModel<H> {
    /// The score `F(x) = c_0 + Σ_j c_j f_j(x)`, as `AdditiveModel`'s
    /// `Regressor::predict`, and its confidence through the model's link.
    fn evaluate(&self, cache: &Cache<H>, ctx: &Context, grid: bool) -> Eval {
        let link = |f: f64| match self.link {
            ScoreLink::Tanh => f.tanh(),
            ScoreLink::Clamp => f.clamp(-1.0, 1.0),
        };
        let k = self.hypotheses.len();
        if !is_cached_prefix(&self.hypotheses, cache) {
            return Eval {
                score: Regressor::predict_all(self, ctx.points),
                conf: self.confidence_all(ctx.points),
                grid: grid.then(|| self.confidence_all(ctx.frame_sample)),
            };
        }
        let score = combine(&self.coefficients, &cache.train[..k], ctx.points.shape().0)
            .into_iter()
            .map(|s| self.intercept + s)
            .collect::<Vec<_>>();
        Eval {
            conf: score.iter().map(|&f| link(f)).collect(),
            score,
            grid: grid.then(|| {
                combine(
                    &self.coefficients,
                    &cache.grid[..k],
                    ctx.frame_sample.shape().0,
                )
                .into_iter()
                .map(|s| link(self.intercept + s))
                .collect()
            }),
        }
    }
}

/// A finished run: the booster (for algorithm-specific diagnostics), its
/// output, the number of weak hypotheses produced, and the stop reason.
struct Trained<B, F> {
    booster: B,
    model: F,
    rounds: usize,
    stop: &'static str,
    timeline: Timeline,
}

impl<B, F: Classifier + 'static> Trained<B, F> {
    fn boxed(self) -> Outcome {
        self.with_note(None, String::new())
    }

    /// The outcome with a note and, if given, a refined stop reason.
    fn with_note(self, stop: Option<&'static str>, note: String) -> Outcome {
        Outcome {
            model: Box::new(self.model),
            rounds: self.rounds,
            stop: stop.unwrap_or(self.stop),
            note,
            timeline: self.timeline,
        }
    }
}

/// Runs at most `limit` rounds of `booster`. Before postprocessing, `check`
/// either reports a numerical failure detected by the booster, which is
/// returned as an error instead of a model, or optionally refines the stop
/// reason of an algorithm stop (e.g. SmoothBoost's `"round_budget"`).
///
/// If `record` is set, the booster's snapshot (`current_hypothesis`) after
/// each round that produced a hypothesis is evaluated for the timeline, on
/// a thinned set of rounds, together with `objective`. The last round is
/// always kept and shows the final model.
#[allow(clippy::too_many_arguments)]
fn run<B, W, C>(
    mut booster: B,
    ctx: &Context,
    learner: &W,
    limit: usize,
    record: bool,
    objective: Objective,
    check: C,
) -> Result<Trained<B, <B as Booster<W::Hypothesis>>::Output>, JsError>
where
    W: WeakLearner,
    W::Hypothesis: Scores + Clone,
    B: Booster<W::Hypothesis> + CurrentHypothesis,
    <B as Booster<W::Hypothesis>>::Output: Classifier + Snapshot<W::Hypothesis>,
    <B as CurrentHypothesis>::Output: Snapshot<W::Hypothesis>,
    C: Fn(&B) -> Result<Option<&'static str>, String>,
{
    let learner = Counting {
        inner: learner,
        calls: Cell::new(0),
        dist: RefCell::new(Vec::new()),
        track_alpha: objective.needs_alpha(),
        alpha_sum: Cell::new(0.0),
        eval: record.then_some((ctx.points, ctx.frame_sample)),
        cache: RefCell::new(Cache::default()),
    };
    let labels = ctx.sample.target();
    let n = ctx.n_train();
    let point = |e: Eval, t: usize| -> Result<CurvePoint, JsError> {
        let score = finite(e.score)?;
        let mut conf = finite(e.conf)?;
        let test_conf = conf.split_off(n);
        Ok(CurvePoint {
            objective: objective.value(labels, &conf, &score[..n], learner.alpha_sum.get(), t),
            conf,
            test_conf,
        })
    };
    let frame = |grid: Vec<f64>| -> Result<Frame, JsError> {
        Ok(Frame {
            grid: finite(grid)?.into_iter().map(|v| v as f32).collect(),
            dist: learner.dist.borrow().clone(),
        })
    };
    let mut timeline = Timeline {
        curve: Thinned::new(CURVE_CAP),
        frames: Thinned::new(FRAME_CAP),
        final_step: "",
    };
    booster.preprocess();
    let mut stop = "max_iterations";
    let mut recorded = 0;
    for round in 1..=limit {
        let flow = booster.boost(&learner, round);
        let t = learner.calls.get();
        if record && t > recorded {
            let (curve, want_frame) = (timeline.curve.wants(t), timeline.frames.wants(t));
            if curve || want_frame {
                let mut e =
                    booster
                        .current_hypothesis()
                        .evaluate(&learner.cache.borrow(), ctx, want_frame);
                if let Some(grid) = e.grid.take() {
                    timeline.frames.push(t, frame(grid)?);
                }
                if curve {
                    timeline.curve.push(t, point(e, t)?);
                }
            }
            recorded = t;
        }
        if let ControlFlow::Break(_) = flow {
            stop = "algorithm_stop";
            break;
        }
    }
    let rounds = learner.calls.get();
    match check(&booster) {
        Err(reason) => return Err(JsError::new(&reason)),
        Ok(Some(reason)) if stop == "algorithm_stop" => stop = reason,
        Ok(_) => {}
    }
    let model = booster.postprocess();
    // The last round shows the final model, which can differ from the
    // snapshot (e.g. SparsiBoost sparsifies only in postprocessing).
    if rounds > 0 {
        let mut e = model.evaluate(&learner.cache.borrow(), ctx, true);
        let last = frame(e.grid.take().unwrap_or_default())?;
        timeline.curve.items.retain(|(t, _)| *t != rounds);
        timeline.curve.items.push((rounds, point(e, rounds)?));
        timeline.frames.items.retain(|(t, _)| *t != rounds);
        timeline.frames.items.push((rounds, last));
    }
    Ok(Trained {
        booster,
        model,
        rounds,
        stop,
        timeline,
    })
}

fn finite(values: Vec<f64>) -> Result<Vec<f64>, JsError> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(values)
    } else {
        Err(JsError::new("a snapshot produced non-finite confidence"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 24 points on two interleaved arcs, labels alternating.
    fn sample() -> Sample {
        let mut rows = Vec::new();
        let mut labels = Vec::new();
        for i in 0..24 {
            let label = if i % 2 == 0 { 1.0 } else { -1.0 };
            let t = std::f64::consts::PI * (i as f64 + 0.5) / 24.0;
            let (x, y) = (t.cos() - 0.5 * label, label * (t.sin() - 0.25));
            rows.push(vec![0.6 * x, 0.6 * y]);
            labels.push(label);
        }
        Sample::from_dense(&rows, Some(&labels)).unwrap()
    }

    fn counting<'a, W: WeakLearner>(inner: &'a W) -> Counting<'a, W> {
        Counting {
            inner,
            calls: Cell::new(0),
            dist: RefCell::new(Vec::new()),
            track_alpha: true,
            alpha_sum: Cell::new(0.0),
            eval: None,
            cache: RefCell::new(Cache::default()),
        }
    }

    /// Snapshots composed from cached predictions must equal the direct
    /// evaluation `confidence_all` exactly (same terms, same summation order).
    fn check_cached_snapshots<B, W>(mut booster: B, learner: &W)
    where
        W: WeakLearner,
        W::Hypothesis: Scores + Clone,
        B: Booster<W::Hypothesis> + CurrentHypothesis,
        <B as CurrentHypothesis>::Output: Snapshot<W::Hypothesis> + Classifier,
    {
        let s = sample();
        let frame = square_grid(9).unwrap();
        let ctx = Context {
            sample: &s,
            points: &s,
            frame_sample: &frame,
        };
        let mut counting = counting(learner);
        counting.track_alpha = false;
        counting.eval = Some((&s, &frame));
        booster.preprocess();
        let mut composed = 0;
        for round in 1..=10 {
            let flow = booster.boost(&counting, round);
            let f = booster.current_hypothesis();
            let e = f.evaluate(&counting.cache.borrow(), &ctx, true);
            assert_eq!(e.conf, f.confidence_all(&s));
            assert_eq!(e.grid.unwrap(), f.confidence_all(&frame));
            composed += 1;
            if flow.is_break() {
                break;
            }
        }
        assert!(composed >= 3);
    }

    #[test]
    fn cached_snapshots_match_direct_evaluation() {
        let s = sample();
        let tree = Learner::Tree(DecisionTreeBuilder::new(&s).max_depth(2).build());
        check_cached_snapshots(AdaBoost::init(&s).tolerance(0.1), &tree);
        check_cached_snapshots(LpBoost::init(&s).nu(2.0).tolerance(0.01), &tree);
        check_cached_snapshots(MlpBoost::init(&s).nu(2.0).tolerance(0.01), &tree);
        check_cached_snapshots(BoostByMajority::init(&s).gamma(0.1).tolerance(0.1), &tree);
        let regression = RegressionTreeBuilder::new(&s).max_depth(2).build();
        // Squared loss: since 0.5.2 the logistic and exponential losses stop
        // after one round when a tree separates this small sample (the line
        // search has no minimizer), which leaves too few rounds to compare.
        check_cached_snapshots(
            GradientBoosting::init(&s)
                .loss(GbmLoss::Squared)
                .shrinkage(0.5)
                .force_quit_at(10),
            &regression,
        );
        check_cached_snapshots(LogitBoost::init(&s).force_quit_at(10), &regression);
    }

    #[test]
    fn gradient_boosting_objective_is_its_training_loss() {
        // With line search, each gradient-boosting round does not increase
        // the training loss L(y, F) (the library's own regression test checks
        // the same), so the objective computed from the scores F must not
        // increase either.
        let s = sample();
        let tree = RegressionTreeBuilder::new(&s).max_depth(1).build();
        for (loss, objective) in [
            (GbmLoss::Logistic, Objective::Deviance),
            (GbmLoss::Exponential, Objective::ScoreExponential),
            (GbmLoss::Squared, Objective::Squared),
        ] {
            let frame = square_grid(5).unwrap();
            let ctx = Context {
                sample: &s,
                points: &s,
                frame_sample: &frame,
            };
            let mut booster = GradientBoosting::init(&s).loss(loss).force_quit_at(8);
            let mut counting = counting(&tree);
            counting.track_alpha = false;
            booster.preprocess();
            let mut values = Vec::new();
            for round in 1..=8 {
                if booster.boost(&counting, round).is_break() {
                    break;
                }
                let e =
                    booster
                        .current_hypothesis()
                        .evaluate(&counting.cache.borrow(), &ctx, false);
                values.push(objective.value(s.target(), &e.conf, &e.score, f64::NAN, round));
            }
            assert!(values.len() >= 3);
            // Tolerance: rounding in sums of 24 losses of size O(1).
            assert!(
                values.windows(2).all(|w| w[1] <= w[0] + 1e-12),
                "{loss:?}: {values:?}"
            );
            assert!(values.last() < values.first(), "{loss:?}: {values:?}");
        }
    }

    /// Checks that the distribution of round `t + 1` is proportional to
    /// `weight(S_t y_i f_t(x_i))`, i.e. that `S_t f_t` is the unnormalized
    /// combination `F_t` the booster reweights by, and returns the
    /// objective values for rounds `1..`.
    fn check_rescaling<B>(
        mut booster: B,
        objective: Objective,
        weight: impl Fn(f64) -> f64,
    ) -> Vec<f64>
    where
        B: Booster<Hypothesis> + CurrentHypothesis,
        <B as CurrentHypothesis>::Output: Classifier,
    {
        let s = sample();
        let tree = DecisionTreeBuilder::new(&s).max_depth(1).build();
        let learner = Learner::Tree(tree);
        let counting = counting(&learner);
        let labels = s.target().to_vec();
        booster.preprocess();
        let mut previous: Option<Vec<f64>> = None;
        let mut values = Vec::new();
        for round in 1..=8 {
            let flow = booster.boost(&counting, round);
            if let Some(expected) = previous.take() {
                // D_t was passed in this round; compare with weights of F_{t-1}.
                let dist = counting.dist.borrow();
                let z: f64 = expected.iter().sum();
                for (d, e) in dist.iter().zip(&expected) {
                    // Tolerance: both sides are normalized exponentials of
                    // margins of size O(10); relative rounding ~1e-15 each.
                    assert!(
                        (d - e / z).abs() <= 1e-9 * (e / z).max(1e-12),
                        "{d} vs {}",
                        e / z
                    );
                }
            }
            if flow.is_break() {
                break;
            }
            let f = booster.current_hypothesis().confidence_all(&s);
            let total = counting.alpha_sum.get();
            values.push(objective.value(&labels, &f, &f, total, round));
            previous = Some(
                labels
                    .iter()
                    .zip(&f)
                    .map(|(y, c)| weight(total * y * c))
                    .collect(),
            );
        }
        assert!(values.len() >= 3, "too few rounds to compare");
        values
    }

    #[test]
    fn adaboost_objective_uses_the_unnormalized_combination() {
        let s = sample();
        let values = check_rescaling(
            AdaBoost::init(&s).tolerance(0.1),
            Objective::Exponential,
            |z| (-z).exp(),
        );
        // AdaBoost's exponential loss is the product of the normalizers
        // Z_t ≤ √(1 - γ_t²) < 1, so it decreases in every round.
        assert!(values.windows(2).all(|w| w[1] < w[0]), "{values:?}");
    }

    #[test]
    fn madaboost_objective_uses_the_unnormalized_combination() {
        let s = sample();
        check_rescaling(MadaBoost::init(&s).tolerance(0.1), Objective::Mada, |z| {
            (-z).exp().min(1.0)
        });
    }

    #[test]
    fn smoothboost_objective_crosses_kappa_when_it_stops() {
        let s = sample();
        let (gamma, kappa) = (0.25, 0.5);
        let tree = DecisionTreeBuilder::new(&s).max_depth(2).build();
        let learner = Learner::Tree(tree);
        let counting = counting(&learner);
        let mut booster = SmoothBoost::init(&s).gamma(gamma).kappa(kappa);
        booster.preprocess();
        let objective = Objective::Smooth { gamma };
        let mut values = Vec::new();
        for round in 1.. {
            if booster.boost(&counting, round).is_break() {
                break;
            }
            let f = booster.current_hypothesis().confidence_all(&s);
            values.push(objective.value(s.target(), &f, &f, f64::NAN, round));
        }
        assert!(!booster.reached_round_budget());
        // Figure 1, line 4: the loop stops at the first round whose
        // (1/m) Σ M(j) is below κ.
        let (last, before) = values.split_last().unwrap();
        assert!(*last < kappa, "{values:?}");
        assert!(before.iter().all(|v| *v >= kappa), "{values:?}");
    }

    #[test]
    fn sparsiboost_stage_is_the_adaboostv_run_inside_sparsiboost() {
        let s = sample();
        let learner = Learner::Tree(DecisionTreeBuilder::new(&s).max_depth(1).build());
        for mut booster in [
            SparsiBoost::init(&s).n_hypotheses(3),
            SparsiBoost::init(&s).tolerance(0.6),
        ] {
            let mut stage = sparsiboost_stage(&s, &booster);
            let sparse = booster.run(&learner);
            let dense = stage.run(&learner);
            // SparsiBoost reweights AdaBoostV's hypotheses, and
            // `WeightedMajority::from_slices` drops those of weight zero, so
            // its hypotheses are a subsequence of AdaBoostV's.
            assert!(dense.hypotheses.len() > 1);
            let mut rest = dense.hypotheses.iter();
            assert!(sparse.hypotheses.iter().all(|h| rest.any(|d| d == h)));
            // Its reported margin change is measured against AdaBoostV's
            // normalized weights, so it must equal the change from `dense`.
            let change = s
                .target()
                .iter()
                .zip(
                    sparse
                        .confidence_all(&s)
                        .iter()
                        .zip(dense.confidence_all(&s)),
                )
                .map(|(y, (a, b))| (y * a - y * b).abs())
                .fold(0.0, f64::max);
            let reported = booster.sparsification_error().unwrap();
            // Tolerance: both sides sum O(10) products of weights ≤ 1.
            assert!((change - reported).abs() < 1e-12, "{change} vs {reported}");
        }
    }

    #[test]
    fn test_points_are_evaluated_but_not_trained_on() {
        let s = sample();
        // Held-out points: the training points shifted slightly.
        let test_rows = (0..s.shape().0)
            .map(|i| vec![s["feature_0"][i] + 0.05, s["feature_1"][i] - 0.03])
            .collect::<Vec<_>>();
        let test = Sample::from_dense(&test_rows, None).unwrap();
        let mut all_rows = (0..s.shape().0)
            .map(|i| vec![s["feature_0"][i], s["feature_1"][i]])
            .collect::<Vec<_>>();
        all_rows.extend(test_rows.iter().cloned());
        let points = Sample::from_dense(&all_rows, None).unwrap();
        let frame = square_grid(5).unwrap();
        let ctx = Context {
            sample: &s,
            points: &points,
            frame_sample: &frame,
        };
        let tree = Learner::Tree(DecisionTreeBuilder::new(&s).max_depth(1).build());
        let t = run(
            AdaBoost::init(&s).tolerance(0.1),
            &ctx,
            &tree,
            // Above AdaBoost's own budget ⌈ln 24 / 0.1²⌉ = 318, so both runs
            // stop at the same round.
            1000,
            true,
            Objective::Exponential,
            no_check,
        )
        .unwrap();
        // Training the same booster without test points gives the same model.
        let alone = AdaBoost::init(&s).tolerance(0.1).run(&tree);
        assert_eq!(t.model.confidence_all(&s), alone.confidence_all(&s));
        let (_, last) = t.timeline.curve.items.last().unwrap();
        assert_eq!(last.conf, t.model.confidence_all(&s));
        assert_eq!(last.test_conf, t.model.confidence_all(&test));
        assert!(
            t.timeline
                .curve
                .items
                .iter()
                .all(|(_, p)| p.test_conf.len() == test.shape().0)
        );
    }

    #[test]
    fn soft_margin_with_nu_one_is_the_minimum_margin() {
        let labels = [1.0, -1.0, 1.0];
        let conf = [0.5, -0.25, -0.1];
        let value = Objective::SoftMargin(1.0).value(&labels, &conf, &conf, f64::NAN, 1);
        assert!((value - -0.1).abs() < 1e-12);
    }
}
