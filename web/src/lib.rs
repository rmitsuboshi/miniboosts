//! Thin WebAssembly boundary for the interactive demo: JavaScript supplies
//! labeled 2-D points, Rust trains a booster with the chosen weak learner and
//! returns the ensemble's confidence on a square grid over `[-1, 1]^2`.
use js_sys::Reflect;
use miniboosts::{
    ClassMember, Classifier, DecisionTree, DecisionTreeClassifier, FiniteClass, LinearClassifier,
    LinearOutput, NaiveBayesClassifier, RbfHypothesis, Sample, prelude::*,
};
use std::{cell::Cell, ops::ControlFlow};
use wasm_bindgen::prelude::*;

/// Result of one training run.
#[wasm_bindgen]
pub struct Fit {
    grid: Vec<f64>,
    rounds: usize,
    stop_reason: String,
    train_error: f64,
    note: String,
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
    /// Algorithm-specific summary (empty for most algorithms).
    #[wasm_bindgen(getter)]
    pub fn note(&self) -> String {
        self.note.clone()
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
/// result on a `resolution × resolution` grid.
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
    if !(2..=400).contains(&resolution) {
        return Err(JsError::new("resolution must lie in [2, 400]"));
    }

    let rows = xs
        .iter()
        .zip(ys)
        .map(|(&x, &y)| vec![x, y])
        .collect::<Vec<_>>();
    let sample = Sample::from_dense(&rows, Some(labels)).map_err(|e| JsError::new(&e))?;
    let p = Params(booster);
    let w = Params(weak);

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
    let (model, rounds, stop_reason, note) = fit(algorithm, &sample, &learner, &p)?;

    let train_error = model
        .predict_all(&sample)
        .iter()
        .zip(labels)
        .filter(|&(&p, &y)| p as f64 != y)
        .count() as f64
        / n as f64;

    let step = 2.0 / (resolution - 1) as f64;
    let grid_rows = (0..resolution)
        .flat_map(|r| {
            let y = 1.0 - r as f64 * step;
            (0..resolution).map(move |c| vec![-1.0 + c as f64 * step, y])
        })
        .collect::<Vec<_>>();
    let grid_sample = Sample::from_dense(&grid_rows, None).map_err(|e| JsError::new(&e))?;
    let grid = model.confidence_all(&grid_sample);
    if grid.iter().any(|v| !v.is_finite()) {
        return Err(JsError::new("ensemble produced non-finite confidence"));
    }

    Ok(Fit {
        grid,
        rounds,
        stop_reason: stop_reason.into(),
        train_error,
        note,
    })
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
#[derive(Clone)]
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

type Outcome = (Box<dyn Classifier>, usize, &'static str, String);

/// Trains `algorithm` with `learner`, validating the booster parameters it
/// uses (the library asserts the same conditions; checking here turns them
/// into errors instead of panics).
fn fit(algorithm: &str, s: &Sample, learner: &Learner, p: &Params) -> Result<Outcome, JsError> {
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
            l,
            limit,
            no_check,
        )?
        .boxed(),
        "adaboostv" => run(
            AdaBoostV::init(s).tolerance(tolerance()?),
            l,
            limit,
            no_check,
        )?
        .boxed(),
        "madaboost" => run(
            MadaBoost::init(s).tolerance(tolerance()?),
            l,
            limit,
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
                l,
                limit,
                |b: &SmoothBoost<_>| Ok(b.reached_round_budget().then_some("round_budget")),
            )?
            .boxed()
        }
        "lpboost" => run(
            LpBoost::init(s).nu(nu()?).tolerance(tolerance()?),
            l,
            limit,
            no_check,
        )?
        .boxed(),
        "erlpboost" => run(
            ErlpBoost::init(s).nu(nu()?).tolerance(tolerance()?),
            l,
            limit,
            no_check,
        )?
        .boxed(),
        "corrective_erlpboost" => run(
            CorrectiveErlpBoost::init(s)
                .nu(nu()?)
                .tolerance(tolerance()?),
            l,
            limit,
            no_check,
        )?
        .boxed(),
        "mlpboost" => run(
            MlpBoost::init(s).nu(nu()?).tolerance(tolerance()?),
            l,
            limit,
            no_check,
        )?
        .boxed(),
        "softboost" => run(
            SoftBoost::init(s).nu(nu()?).tolerance(tolerance()?),
            l,
            limit,
            |b: &SoftBoost<_>| {
                b.numerical_stop_reason()
                    .map_or(Ok(None), |r| Err(r.to_owned()))
            },
        )?
        .boxed(),
        "totalboost" => run(
            TotalBoost::init(s).tolerance(tolerance()?),
            l,
            limit,
            |b: &TotalBoost<_>| {
                b.numerical_stop_reason()
                    .map_or(Ok(None), |r| Err(r.to_owned()))
            },
        )?
        .boxed(),
        "graph_separation" => run(GraphSeparationBoosting::init(s), l, limit, no_check)?.boxed(),
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
            let t = run(booster, l, limit, no_check)?;
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
            (
                Box::new(t.model) as Box<dyn Classifier>,
                t.rounds,
                t.stop,
                note,
            )
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
            let t = run(booster, l, limit, no_check)?;
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
            (
                Box::new(t.model) as Box<dyn Classifier>,
                t.rounds,
                stop,
                note,
            )
        }
        _ => return Err(JsError::new("unknown boosting algorithm")),
    })
}

fn no_check<B>(_: &B) -> Result<Option<&'static str>, String> {
    Ok(None)
}

/// Counts the weak-learner calls, since boosters may stop at the start of a
/// round without producing a hypothesis.
struct Counting<'a, W> {
    inner: &'a W,
    calls: Cell<usize>,
}

impl<W: WeakLearner> WeakLearner for Counting<'_, W> {
    type Hypothesis = W::Hypothesis;
    fn produce(&self, sample: &Sample, dist: &[f64]) -> Self::Hypothesis {
        self.calls.set(self.calls.get() + 1);
        self.inner.produce(sample, dist)
    }
}

/// A finished run: the booster (for algorithm-specific diagnostics), its
/// output, the number of weak hypotheses produced, and the stop reason.
struct Trained<B, F> {
    booster: B,
    model: F,
    rounds: usize,
    stop: &'static str,
}

impl<B, F: Classifier + 'static> Trained<B, F> {
    fn boxed(self) -> Outcome {
        (Box::new(self.model), self.rounds, self.stop, String::new())
    }
}

/// Runs at most `limit` rounds of `booster`. Before postprocessing, `check`
/// either reports a numerical failure detected by the booster, which is
/// returned as an error instead of a model, or optionally refines the stop
/// reason of an algorithm stop (e.g. SmoothBoost's `"round_budget"`).
fn run<B, W, C>(
    mut booster: B,
    learner: &W,
    limit: usize,
    check: C,
) -> Result<Trained<B, B::Output>, JsError>
where
    W: WeakLearner,
    B: Booster<W::Hypothesis>,
    C: Fn(&B) -> Result<Option<&'static str>, String>,
{
    let learner = Counting {
        inner: learner,
        calls: Cell::new(0),
    };
    booster.preprocess();
    let mut stop = "max_iterations";
    for round in 1..=limit {
        if let ControlFlow::Break(_) = booster.boost(&learner, round) {
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
    Ok(Trained {
        booster,
        model,
        rounds,
        stop,
    })
}
