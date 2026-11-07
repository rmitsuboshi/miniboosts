//! Thin Python boundary: Rust owns boosting and aggregation; Python supplies
//! independently fitted weak hypotheses as batches of confidence values (or,
//! for boosters with regression weak learners, of real-valued predictions).
use miniboosts::{
    Classifier, FwUpdateRule, RegressionTreeRegressor, Regressor, Sample, prelude::*,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use std::{
    cell::{Cell, RefCell},
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    sync::{Arc, RwLock},
};

type Scores = Arc<RwLock<Vec<Vec<f64>>>>;
type BoxedClassifier = Box<dyn Classifier + Send + Sync>;

// `BrownBoost::numerical_stop_reason` when a weak hypothesis has correlation
// at most `ν` (Freund 2001, Eq. 5 has only the solution α = t = 0). This is a
// weak-learning failure, not a solver failure; a test pins the text.
const BROWNBOOST_EDGE_STOP: &str =
    "the weak hypothesis has weighted correlation at most the edge threshold";
// Likewise for RobustBoost (Freund 2009): a nonpositive correlation admits no
// step with Δm > 0.
const ROBUSTBOOST_EDGE_STOP: &str = "the weak hypothesis has nonpositive weighted correlation";
// Gradient boosting: the line search step ρ is undefined for h = 0.
const GBM_ZERO_STOP: &str = "the weak hypothesis is zero on the training sample";
// Gradient boosting: the logistic/exponential loss decreases for every step
// along h (no example's margin gets worse), so argmin_ρ does not exist.
const GBM_NO_MINIMIZER_STOP: &str =
    "the line search has no minimizer: the loss decreases for every step size along h";
// LogitBoost: every working weight p(1 - p) underflowed, so no example is
// left to fit (the scores of all examples have saturated).
const LOGITBOOST_UNDERFLOW_STOP: &str = "every LogitBoost weight underflowed to 0";

#[derive(Clone, Debug)]
struct Hypothesis {
    index: usize,
    scores: Scores,
}
impl PartialEq for Hypothesis {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && Arc::ptr_eq(&self.scores, &other.scores)
    }
}
impl Classifier for Hypothesis {
    fn confidence(&self, _: &Sample, row: usize) -> f64 {
        self.scores.read().expect("confidence cache poisoned")[self.index][row]
    }
}
// The same cache serves regression weak learners; only validation differs.
impl Regressor for Hypothesis {
    fn predict(&self, _: &Sample, row: usize) -> f64 {
        self.scores.read().expect("confidence cache poisoned")[self.index][row]
    }
}

struct CallbackLearner {
    callback: Py<PyAny>,
    scores: Scores,
    error: RefCell<Option<PyErr>>,
    round: Cell<usize>,
    // Classification confidences lie in [-1, 1]; regression predictions only
    // need to be finite.
    bounded: bool,
}
impl WeakLearner for CallbackLearner {
    type Hypothesis = Hypothesis;
    fn produce(&self, sample: &Sample, dist: &[f64]) -> Hypothesis {
        let result = Python::attach(|py| {
            py.check_signals()?;
            let values: Vec<f64> = self
                .callback
                .call1(
                    py,
                    (dist.to_vec(), self.round.get(), sample.target().to_vec()),
                )?
                .extract(py)?;
            validate_values(&values, sample.shape().0, self.bounded)?;
            Ok::<_, PyErr>(values)
        });
        let values = match result {
            Ok(values) => values,
            Err(error) => {
                *self.error.borrow_mut() = Some(error);
                // Trait has no Result channel. Unwind only to the local catch
                // boundary, preserving the original Python exception/traceback.
                resume_unwind(Box::new(()));
            }
        };
        let mut scores = self.scores.write().expect("confidence cache poisoned");
        let index = scores.len();
        scores.push(values);
        Hypothesis {
            index,
            scores: self.scores.clone(),
        }
    }
}

fn validate_scores(values: &[f64], n: usize) -> PyResult<()> {
    if values.len() != n {
        return Err(PyValueError::new_err(
            "weak confidence must have one value per row",
        ));
    }
    for (row, &v) in values.iter().enumerate() {
        if !v.is_finite() || !(-1.0..=1.0).contains(&v) {
            return Err(PyValueError::new_err(format!(
                "invalid weak confidence at row {row}: {v}; expected finite [-1, 1]"
            )));
        }
    }
    Ok(())
}

fn validate_values(values: &[f64], n: usize, bounded: bool) -> PyResult<()> {
    if bounded {
        return validate_scores(values, n);
    }
    if values.len() != n {
        return Err(PyValueError::new_err(
            "weak prediction must have one value per row",
        ));
    }
    if let Some(row) = values.iter().position(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "invalid weak prediction at row {row}: {}; expected a finite value",
            values[row]
        )));
    }
    Ok(())
}

fn panic_error(payload: Box<dyn std::any::Any + Send>) -> PyErr {
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("unknown Rust panic");
    PyRuntimeError::new_err(format!("MiniBoosts failed: {message}"))
}

/// Algorithm-specific values read from the booster after training; `None`
/// for algorithms that do not define them.
#[derive(Default)]
struct Diagnostics {
    remaining_time: Option<f64>,
    total_time: Option<f64>,
    sparsification_error: Option<f64>,
    sparsification_within_tolerance: Option<bool>,
    adaboostv_rounds: Option<usize>,
    adaboostv_tolerance: Option<f64>,
    n_selected: Option<usize>,
    n_rounds: Option<usize>,
    decided_early: Option<bool>,
    intercept: Option<f64>,
    coefficients: Option<Vec<f64>>,
}

#[pyclass]
struct Model {
    model: BoxedClassifier,
    scores: Scores,
    count: usize,
    bounded: bool,
    #[pyo3(get)]
    n_iterations: usize,
    #[pyo3(get)]
    stop_reason: String,
    #[pyo3(get)]
    remaining_time: Option<f64>,
    #[pyo3(get)]
    total_time: Option<f64>,
    #[pyo3(get)]
    sparsification_error: Option<f64>,
    #[pyo3(get)]
    sparsification_within_tolerance: Option<bool>,
    #[pyo3(get)]
    adaboostv_rounds: Option<usize>,
    #[pyo3(get)]
    adaboostv_tolerance: Option<f64>,
    #[pyo3(get)]
    n_selected: Option<usize>,
    #[pyo3(get)]
    n_rounds: Option<usize>,
    #[pyo3(get)]
    decided_early: Option<bool>,
    #[pyo3(get)]
    intercept: Option<f64>,
    #[pyo3(get)]
    coefficients: Option<Vec<f64>>,
}
#[pymethods]
impl Model {
    fn decision_function(
        &mut self,
        predictions: Vec<Vec<f64>>,
        n_rows: usize,
    ) -> PyResult<Vec<f64>> {
        if n_rows == 0 || predictions.len() != self.count {
            return Err(PyValueError::new_err(
                "prediction matrix has incorrect dimensions",
            ));
        }
        for values in &predictions {
            validate_values(values, n_rows, self.bounded)?;
        }
        let result = catch_unwind(AssertUnwindSafe(|| {
            *self.scores.write().expect("confidence cache poisoned") = predictions;
            self.model.confidence_all(&Sample::dummy(n_rows))
        }))
        .map_err(panic_error)?;
        if result.iter().any(|v| !v.is_finite()) {
            return Err(PyRuntimeError::new_err(
                "ensemble produced non-finite confidence",
            ));
        }
        Ok(result)
    }
}

type Fitted = (BoxedClassifier, &'static str, Diagnostics);

/// Runs at most `limit` boosting steps; returns whether the algorithm stopped
/// by itself (`ControlFlow::Break`) rather than at the limit.
fn boost_loop<B: Booster<Hypothesis>>(
    booster: &mut B,
    learner: &CallbackLearner,
    limit: usize,
) -> bool {
    booster.preprocess();
    for round in 1..=limit {
        learner.round.set(round);
        if booster.boost(learner, round).is_break() {
            return true;
        }
    }
    false
}

fn stop_name(algorithm_stop: bool) -> &'static str {
    if algorithm_stop {
        "algorithm_stop"
    } else {
        "max_iterations"
    }
}

fn plain<B>(mut booster: B, learner: &CallbackLearner, limit: usize) -> PyResult<Fitted>
where
    B: Booster<Hypothesis>,
    B::Output: Classifier + Send + Sync + 'static,
{
    let stop = stop_name(boost_loop(&mut booster, learner, limit));
    Ok((
        Box::new(booster.postprocess()),
        stop,
        Diagnostics::default(),
    ))
}

#[pyfunction]
#[pyo3(signature = (
    algorithm, x, y, callback, max_iterations, tolerance=None, nu=1.0, gamma=0.25,
    kappa=0.1, update_rule="default", edge_threshold=1e-3, n_hypotheses=None, seed=0,
    theta=0.0, sigma_f=0.1, loss="logistic", shrinkage=1.0, max_response=4.0,
))]
#[allow(clippy::too_many_arguments)]
fn train(
    algorithm: &str,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    callback: Py<PyAny>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    nu: f64,
    gamma: f64,
    kappa: f64,
    update_rule: &str,
    edge_threshold: f64,
    n_hypotheses: Option<usize>,
    seed: u64,
    theta: f64,
    sigma_f: f64,
    loss: &str,
    shrinkage: f64,
    max_response: f64,
) -> PyResult<Model> {
    let sample = Sample::from_dense(&x, Some(&y)).map_err(PyValueError::new_err)?;
    if max_iterations == Some(0) {
        return Err(PyValueError::new_err("max_iterations must be positive"));
    }
    // Only SparsiBoost and BBM derive their own round counts.
    if max_iterations.is_none() && !matches!(algorithm, "sparsiboost" | "bbm") {
        return Err(PyValueError::new_err("max_iterations is required"));
    }
    check_labels(&y)?;
    let rule = match update_rule {
        "default" => None,
        "classic" => Some(FwUpdateRule::Classic),
        "short_step" => Some(FwUpdateRule::ShortStep),
        "line_search" => Some(FwUpdateRule::LineSearch),
        _ => return Err(PyValueError::new_err("unknown update_rule")),
    };
    let loss = match loss {
        "logistic" => GbmLoss::Logistic,
        "exponential" => GbmLoss::Exponential,
        "squared" => GbmLoss::Squared,
        _ => return Err(PyValueError::new_err("unknown loss")),
    };
    let bounded = !matches!(algorithm, "gradient_boosting" | "logitboost");
    let learner = CallbackLearner {
        callback,
        scores: Arc::new(RwLock::new(Vec::new())),
        error: RefCell::new(None),
        round: Cell::new(0),
        bounded,
    };
    let result = catch_unwind(AssertUnwindSafe(|| -> PyResult<Fitted> {
        let s = &sample;
        let l = &learner;
        let tol = tolerance.unwrap_or(0.01);
        let n = max_iterations.unwrap_or(0);
        match algorithm {
            "adaboost" => plain(AdaBoost::init(s).tolerance(tol), l, n),
            "adaboostv" => plain(AdaBoostV::init(s).tolerance(tol), l, n),
            "madaboost" => plain(MadaBoost::init(s).tolerance(tol), l, n),
            "lpboost" => plain(LpBoost::init(s).nu(nu).tolerance(tol), l, n),
            "erlpboost" => plain(ErlpBoost::init(s).nu(nu).tolerance(tol), l, n),
            "corrective_erlpboost" => {
                let b = CorrectiveErlpBoost::init(s).nu(nu).tolerance(tol);
                plain(
                    if let Some(rule) = rule {
                        b.update_rule(rule)
                    } else {
                        b
                    },
                    l,
                    n,
                )
            }
            "mlpboost" => {
                let b = MlpBoost::init(s).nu(nu).tolerance(tol);
                plain(
                    if let Some(rule) = rule {
                        b.update_rule(rule)
                    } else {
                        b
                    },
                    l,
                    n,
                )
            }
            "smoothboost" => {
                let mut b = SmoothBoost::init(s).gamma(gamma).kappa(kappa);
                // The paper's round bound without the stopping test firing
                // is not an algorithm stop with a guarantee.
                let stop = match boost_loop(&mut b, l, n) {
                    true if b.reached_round_budget() => "round_budget",
                    stopped => stop_name(stopped),
                };
                Ok((Box::new(b.postprocess()), stop, Diagnostics::default()))
            }
            "softboost" => {
                let mut b = SoftBoost::init(s).nu(nu).tolerance(tol);
                let stop = stop_name(boost_loop(&mut b, l, n));
                if let Some(reason) = b.numerical_stop_reason() {
                    return Err(PyRuntimeError::new_err(reason.to_owned()));
                }
                Ok((Box::new(b.postprocess()), stop, Diagnostics::default()))
            }
            "totalboost" => {
                let mut b = TotalBoost::init(s).tolerance(tol);
                let stop = stop_name(boost_loop(&mut b, l, n));
                if let Some(reason) = b.numerical_stop_reason() {
                    return Err(PyRuntimeError::new_err(reason.to_owned()));
                }
                Ok((Box::new(b.postprocess()), stop, Diagnostics::default()))
            }
            "brownboost" => {
                let mut b = BrownBoost::init(s)
                    .tolerance(tol)
                    .edge_threshold(edge_threshold);
                let mut stop = stop_name(boost_loop(&mut b, l, n));
                match b.numerical_stop_reason() {
                    None => {}
                    Some(BROWNBOOST_EDGE_STOP) => stop = "edge_threshold",
                    Some(reason) => {
                        return Err(PyRuntimeError::new_err(format!("BrownBoost: {reason}")));
                    }
                }
                let diagnostics = Diagnostics {
                    remaining_time: b.remaining_time(),
                    total_time: Some(b.total_time()),
                    ..Diagnostics::default()
                };
                Ok((Box::new(b.postprocess()), stop, diagnostics))
            }
            "sparsiboost" => {
                if n_hypotheses.is_none() && tolerance.is_none() {
                    return Err(PyValueError::new_err(
                        "SparsiBoost needs n_hypotheses, tolerance, or both",
                    ));
                }
                let mut b = SparsiBoost::init(s).seed(seed);
                if let Some(t) = n_hypotheses {
                    b = b.n_hypotheses(t);
                }
                if let Some(v) = tolerance {
                    b = b.tolerance(v);
                }
                let rounds = b.max_loop();
                let nu = b.adaboostv_tolerance();
                // By default, allow the terminal call in which AdaBoostV
                // reports that its `rounds` iterations are done.
                let limit = max_iterations.unwrap_or(rounds.saturating_add(1));
                let stop = stop_name(boost_loop(&mut b, l, limit));
                let f = b.postprocess();
                let diagnostics = Diagnostics {
                    sparsification_error: b.sparsification_error(),
                    sparsification_within_tolerance: b.sparsification_within_tolerance(),
                    adaboostv_rounds: Some(rounds),
                    adaboostv_tolerance: Some(nu),
                    // `from_slices` drops the hypotheses with zero weight.
                    n_selected: Some(f.hypotheses.len()),
                    ..Diagnostics::default()
                };
                Ok((Box::new(f), stop, diagnostics))
            }
            "arcgv" => {
                // The round limit is required (the guarantee is asymptotic);
                // the loop below enforces it, so Δ_k = 0 is the only Break.
                plain(ArcGv::init(s).force_quit_at(n), l, n)
            }
            "bbm" => {
                let mut b = BoostByMajority::init(s).gamma(gamma).tolerance(tol);
                let rounds = b.max_loop();
                // By default, allow the terminal call after the k rounds.
                let limit = max_iterations.unwrap_or(rounds.saturating_add(1));
                let stop = stop_name(boost_loop(&mut b, l, limit));
                let diagnostics = Diagnostics {
                    n_rounds: Some(rounds),
                    decided_early: Some(b.decided_early()),
                    ..Diagnostics::default()
                };
                Ok((Box::new(b.postprocess()), stop, diagnostics))
            }
            "robustboost" => {
                let mut b = RobustBoost::init(s)
                    .tolerance(tol)
                    .theta(theta)
                    .sigma_f(sigma_f);
                let mut stop = stop_name(boost_loop(&mut b, l, n));
                match b.numerical_stop_reason() {
                    None => {}
                    Some(ROBUSTBOOST_EDGE_STOP) => stop = "nonpositive_edge",
                    Some(reason) => {
                        return Err(PyRuntimeError::new_err(format!("RobustBoost: {reason}")));
                    }
                }
                let diagnostics = Diagnostics {
                    remaining_time: b.remaining_time(),
                    ..Diagnostics::default()
                };
                Ok((Box::new(b.postprocess()), stop, diagnostics))
            }
            "gradient_boosting" => {
                let mut b = GradientBoosting::init(s)
                    .loss(loss)
                    .shrinkage(shrinkage)
                    .force_quit_at(n);
                let mut stop = stop_name(boost_loop(&mut b, l, n));
                match b.numerical_stop_reason() {
                    None => {}
                    Some(GBM_ZERO_STOP) => stop = "zero_hypothesis",
                    Some(GBM_NO_MINIMIZER_STOP) => stop = "no_minimizer",
                    Some(reason) => {
                        return Err(PyRuntimeError::new_err(format!(
                            "gradient boosting: {reason}"
                        )));
                    }
                }
                let f = b.postprocess();
                let diagnostics = Diagnostics {
                    intercept: Some(f.intercept),
                    coefficients: Some(f.coefficients.clone()),
                    ..Diagnostics::default()
                };
                Ok((Box::new(f), stop, diagnostics))
            }
            "logitboost" => {
                let mut b = LogitBoost::init(s)
                    .max_response(max_response)
                    .force_quit_at(n);
                let mut stop = stop_name(boost_loop(&mut b, l, n));
                match b.numerical_stop_reason() {
                    None => {}
                    Some(LOGITBOOST_UNDERFLOW_STOP) => stop = "weights_underflow",
                    Some(reason) => {
                        return Err(PyRuntimeError::new_err(format!("LogitBoost: {reason}")));
                    }
                }
                let f = b.postprocess();
                let diagnostics = Diagnostics {
                    intercept: Some(f.intercept),
                    coefficients: Some(f.coefficients.clone()),
                    ..Diagnostics::default()
                };
                Ok((Box::new(f), stop, diagnostics))
            }
            "graph_separation" => plain(GraphSeparationBoosting::init(s), l, n),
            _ => Err(PyValueError::new_err("unknown boosting algorithm")),
        }
    }));
    if let Some(error) = learner.error.into_inner() {
        return Err(error);
    }
    let (model, stop_reason, diagnostics) = result.map_err(panic_error)??;
    let count = learner
        .scores
        .read()
        .expect("confidence cache poisoned")
        .len();
    Ok(Model {
        model,
        scores: learner.scores,
        count,
        bounded,
        n_iterations: count,
        stop_reason: stop_reason.to_string(),
        remaining_time: diagnostics.remaining_time,
        total_time: diagnostics.total_time,
        sparsification_error: diagnostics.sparsification_error,
        sparsification_within_tolerance: diagnostics.sparsification_within_tolerance,
        adaboostv_rounds: diagnostics.adaboostv_rounds,
        adaboostv_tolerance: diagnostics.adaboostv_tolerance,
        n_selected: diagnostics.n_selected,
        n_rounds: diagnostics.n_rounds,
        decided_early: diagnostics.decided_early,
        intercept: diagnostics.intercept,
        coefficients: diagnostics.coefficients,
    })
}

fn check_labels(y: &[f64]) -> PyResult<()> {
    if y.iter().any(|v| *v != -1.0 && *v != 1.0) || !y.contains(&-1.0) || !y.contains(&1.0) {
        return Err(PyValueError::new_err(
            "training requires both -1 and +1 labels",
        ));
    }
    Ok(())
}

fn check_distribution(weights: &[f64], n: usize) -> PyResult<()> {
    if weights.len() != n
        || weights.iter().any(|v| !v.is_finite() || *v < 0.0)
        || (weights.iter().sum::<f64>() - 1.0).abs() > 1e-8
    {
        return Err(PyValueError::new_err(
            "weights must be a finite probability distribution",
        ));
    }
    Ok(())
}

/// A fitted native weak hypothesis. `index` and `sign` identify the chosen
/// member `sign · h_index` for learners over a finite class, and
/// `assumption_violated` says whether no member reached the requested edge
/// under an adversarial selection rule.
#[pyclass]
struct NativeClassifier {
    model: BoxedClassifier,
    n_features: usize,
    #[pyo3(get)]
    index: Option<usize>,
    #[pyo3(get)]
    sign: Option<f64>,
    #[pyo3(get)]
    assumption_violated: Option<bool>,
}
#[pymethods]
impl NativeClassifier {
    fn decision_function(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let sample = Sample::from_dense(&x, None).map_err(PyValueError::new_err)?;
        if sample.shape().1 != self.n_features {
            return Err(PyValueError::new_err("incorrect feature count"));
        }
        catch_unwind(AssertUnwindSafe(|| self.model.confidence_all(&sample))).map_err(panic_error)
    }
}

/// Validates the training data, then fits without the GIL. Rust panics
/// (invalid parameters, non-convergence) become `RuntimeError`.
fn fit_native<H: Send>(
    py: Python<'_>,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Vec<f64>,
    fit: impl FnOnce(&Sample, &[f64]) -> H + Send,
) -> PyResult<(H, usize)> {
    let sample = Sample::from_dense(&x, Some(&y)).map_err(PyValueError::new_err)?;
    check_labels(&y)?;
    check_distribution(&weights, y.len())?;
    let n_features = sample.shape().1;
    let h = py
        .detach(|| catch_unwind(AssertUnwindSafe(|| fit(&sample, &weights))))
        .map_err(panic_error)?;
    Ok((h, n_features))
}

fn native(model: impl Classifier + Send + Sync + 'static, n_features: usize) -> NativeClassifier {
    NativeClassifier {
        model: Box::new(model),
        n_features,
        index: None,
        sign: None,
        assumption_violated: None,
    }
}

/// A finite class's selection rule: `"largest"` (the default rule),
/// `"weakest"` (`FiniteClass::weakest_with_edge`), or `"exactly"`
/// (`FiniteClass::edge_exactly`), the last two with edge `gamma`.
#[derive(Clone, Copy)]
enum Selection {
    Largest,
    Weakest(f64),
    Exactly(f64),
}

impl Selection {
    fn parse(selection: &str, gamma: Option<f64>) -> PyResult<Self> {
        let gamma = || match gamma {
            Some(g) if g > 0.0 && g <= 1.0 => Ok(g),
            _ => Err(PyValueError::new_err(
                "gamma must satisfy 0 < gamma <= 1 for this selection",
            )),
        };
        match selection {
            "largest" => Ok(Self::Largest),
            "weakest" => Ok(Self::Weakest(gamma()?)),
            "exactly" => Ok(Self::Exactly(gamma()?)),
            _ => Err(PyValueError::new_err(
                "selection must be largest, weakest, or exactly",
            )),
        }
    }

    fn apply<H>(self, class: FiniteClass<H>) -> FiniteClass<H> {
        match self {
            Self::Largest => class,
            Self::Weakest(gamma) => class.weakest_with_edge(gamma),
            Self::Exactly(gamma) => class.edge_exactly(gamma),
        }
    }
}

#[pyfunction]
#[pyo3(signature = (x, y, weights, max_depth, split_by="entropy", exact_splits=false, leaf_confidence="purity"))]
#[allow(clippy::too_many_arguments)]
fn fit_tree(
    py: Python<'_>,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Vec<f64>,
    max_depth: usize,
    split_by: &str,
    exact_splits: bool,
    leaf_confidence: &str,
) -> PyResult<NativeClassifier> {
    if max_depth == 0 {
        return Err(PyValueError::new_err("tree needs positive depth"));
    }
    let split_by = match split_by {
        "entropy" => SplitBy::Entropy,
        "edge" => SplitBy::Edge,
        "gini" => SplitBy::Gini,
        "twoing" => SplitBy::Twoing,
        _ => {
            return Err(PyValueError::new_err(
                "split_by must be entropy, edge, gini, or twoing",
            ));
        }
    };
    let leaf_confidence = match leaf_confidence {
        "purity" => LeafConfidence::Purity,
        "sign" => LeafConfidence::Sign,
        _ => {
            return Err(PyValueError::new_err(
                "leaf_confidence must be purity or sign",
            ));
        }
    };
    let (h, d) = fit_native(py, x, y, weights, |s, w| {
        let builder = DecisionTreeBuilder::new(s)
            .max_depth(max_depth)
            .split_by(split_by)
            .leaf_confidence(leaf_confidence);
        if exact_splits {
            builder.exact_splits()
        } else {
            builder
        }
        .build()
        .produce(s, w)
    })?;
    Ok(native(h, d))
}

#[pyfunction]
fn fit_bounded_linear(
    py: Python<'_>,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Vec<f64>,
) -> PyResult<NativeClassifier> {
    let (h, d) = fit_native(py, x, y, weights, |s, w| BoundedLinear::new().produce(s, w))?;
    Ok(native(h, d))
}

#[pyfunction]
fn fit_logistic_regression(
    py: Python<'_>,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Vec<f64>,
    regularization: f64,
    output: &str,
) -> PyResult<NativeClassifier> {
    let output = match output {
        "sign" => LinearOutput::Sign,
        "tanh" => LinearOutput::Tanh,
        _ => return Err(PyValueError::new_err("output must be sign or tanh")),
    };
    let (h, d) = fit_native(py, x, y, weights, |s, w| {
        LogisticRegression::new()
            .regularization(regularization)
            .output(output)
            .produce(s, w)
    })?;
    Ok(native(h, d))
}

#[pyfunction]
fn fit_naive_bayes(
    py: Python<'_>,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Vec<f64>,
    var_smoothing: f64,
) -> PyResult<NativeClassifier> {
    let (h, d) = fit_native(py, x, y, weights, |s, w| {
        NaiveBayes::new().var_smoothing(var_smoothing).produce(s, w)
    })?;
    Ok(native(h, d))
}

#[pyfunction]
#[pyo3(signature = (x, y, weights, bandwidth, centers=None, selection="largest", gamma=None))]
#[allow(clippy::too_many_arguments)]
fn fit_rbf(
    py: Python<'_>,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Vec<f64>,
    bandwidth: f64,
    centers: Option<Vec<usize>>,
    selection: &str,
    gamma: Option<f64>,
) -> PyResult<NativeClassifier> {
    let selection = Selection::parse(selection, gamma)?;
    let (result, d) = fit_native(py, x, y, weights, |s, w| {
        let builder = RbfClassBuilder::new(s).bandwidth(bandwidth);
        let class = match centers {
            Some(rows) => builder.centers(rows),
            None => builder,
        }
        .build();
        let class = selection.apply(class);
        let h = class.produce(s, w);
        (h, class.assumption_violations() > 0)
    })?;
    let (h, violated) = result;
    let (index, sign) = (h.index, h.sign);
    Ok(NativeClassifier {
        index: Some(index),
        sign: Some(sign),
        assumption_violated: Some(violated),
        ..native(h, d)
    })
}

/// Returns `(j, sign, violated)` for the member `sign · columns[j]` that
/// `FiniteClass::produce` chooses under `weights` with the given selection
/// rule (see `Selection`); `violated` is true if no member reached
/// `gamma`.
#[pyfunction]
#[pyo3(signature = (columns, y, weights, with_negations, selection="largest", gamma=None))]
fn select_finite_class(
    columns: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Vec<f64>,
    with_negations: bool,
    selection: &str,
    gamma: Option<f64>,
) -> PyResult<(usize, f64, bool)> {
    // The table ignores features; one constant column gives the sample shape.
    let rows = vec![vec![0.0]; y.len()];
    let sample = Sample::from_dense(&rows, Some(&y)).map_err(PyValueError::new_err)?;
    check_labels(&y)?;
    check_distribution(&weights, y.len())?;
    if columns.is_empty() {
        return Err(PyValueError::new_err(
            "the finite class needs at least one hypothesis",
        ));
    }
    for column in &columns {
        validate_scores(column, y.len())?;
    }
    let class = catch_unwind(AssertUnwindSafe(|| {
        let class = FiniteClass::from_table(&sample, columns);
        if with_negations {
            class.with_negations()
        } else {
            class
        }
    }))
    .map_err(panic_error)?;
    let class = Selection::parse(selection, gamma)?.apply(class);
    catch_unwind(AssertUnwindSafe(|| {
        let member = class.produce(&sample, &weights);
        (member.index, member.sign, class.assumption_violations() > 0)
    }))
    .map_err(panic_error)
}

/// Row indices of the parts used by `MajorityOfFive` on `n_samples` rows.
#[pyfunction]
fn majority_of_five_parts(
    n_samples: usize,
    n_parts: usize,
    seed: u64,
) -> PyResult<Vec<Vec<usize>>> {
    if n_parts == 0 || n_parts > n_samples {
        return Err(PyValueError::new_err(
            "n_parts must lie between 1 and the number of training examples",
        ));
    }
    let sample = Sample::dummy(n_samples);
    Ok(MajorityOfFive::init(&sample)
        .n_parts(n_parts)
        .seed(seed)
        .parts())
}

/// Row indices of the sub-samples used by `LarsenRitzert` on `n_samples`
/// rows.
#[pyfunction]
#[pyo3(signature = (n_samples, seed, max_subsamples=None))]
fn larsen_ritzert_subsamples(
    n_samples: usize,
    seed: u64,
    max_subsamples: Option<usize>,
) -> PyResult<Vec<Vec<usize>>> {
    let sample = Sample::dummy(n_samples);
    let learner = LarsenRitzert::init(&sample).seed(seed);
    let learner = match max_subsamples {
        Some(0) => {
            return Err(PyValueError::new_err(
                "max_subsamples must be a positive integer",
            ));
        }
        Some(count) => learner.max_subsamples(count),
        None => learner,
    };
    Ok(learner.subsamples())
}

#[pyclass]
struct NativeRegressor {
    model: RegressionTreeRegressor,
    n_features: usize,
}
#[pymethods]
impl NativeRegressor {
    fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let sample = Sample::from_dense(&x, None).map_err(PyValueError::new_err)?;
        if sample.shape().1 != self.n_features {
            return Err(PyValueError::new_err("incorrect feature count"));
        }
        catch_unwind(AssertUnwindSafe(|| self.model.predict_all(&sample))).map_err(panic_error)
    }
}

#[pyfunction]
#[pyo3(signature = (x, y, weights, max_depth, n_bins=None))]
fn fit_regression_tree(
    py: Python<'_>,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Vec<f64>,
    max_depth: usize,
    n_bins: Option<usize>,
) -> PyResult<NativeRegressor> {
    // Targets are real values, so only finiteness is checked (by from_dense).
    let sample = Sample::from_dense(&x, Some(&y)).map_err(PyValueError::new_err)?;
    check_distribution(&weights, y.len())?;
    if n_bins == Some(0) {
        return Err(PyValueError::new_err("n_bins must be positive"));
    }
    let n_features = sample.shape().1;
    let model = py
        .detach(|| {
            catch_unwind(AssertUnwindSafe(|| {
                let builder = RegressionTreeBuilder::new(&sample).max_depth(max_depth);
                match n_bins {
                    Some(n) => builder.n_bins(n),
                    None => builder,
                }
                .build()
                .produce(&sample, &weights)
            }))
        })
        .map_err(panic_error)?;
    Ok(NativeRegressor { model, n_features })
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_function(wrap_pyfunction!(fit_tree, m)?)?;
    m.add_function(wrap_pyfunction!(fit_bounded_linear, m)?)?;
    m.add_function(wrap_pyfunction!(fit_logistic_regression, m)?)?;
    m.add_function(wrap_pyfunction!(fit_naive_bayes, m)?)?;
    m.add_function(wrap_pyfunction!(fit_rbf, m)?)?;
    m.add_function(wrap_pyfunction!(select_finite_class, m)?)?;
    m.add_function(wrap_pyfunction!(fit_regression_tree, m)?)?;
    m.add_function(wrap_pyfunction!(majority_of_five_parts, m)?)?;
    m.add_function(wrap_pyfunction!(larsen_ritzert_subsamples, m)?)?;
    m.add_class::<Model>()?;
    m.add_class::<NativeClassifier>()?;
    m.add_class::<NativeRegressor>()?;
    Ok(())
}
