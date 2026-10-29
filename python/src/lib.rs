//! Thin Python boundary: Rust owns boosting and aggregation; Python supplies
//! independently fitted weak hypotheses as batches of confidence values.
use miniboosts::{Classifier, FwUpdateRule, Sample, prelude::*};
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

struct CallbackLearner {
    callback: Py<PyAny>,
    scores: Scores,
    error: RefCell<Option<PyErr>>,
    round: Cell<usize>,
}
impl WeakLearner for CallbackLearner {
    type Hypothesis = Hypothesis;
    fn produce(&self, sample: &Sample, dist: &[f64]) -> Hypothesis {
        let result = Python::attach(|py| {
            py.check_signals()?;
            let values: Vec<f64> = self
                .callback
                .call1(py, (dist.to_vec(), self.round.get()))?
                .extract(py)?;
            validate_scores(&values, sample.shape().0)?;
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

fn panic_error(payload: Box<dyn std::any::Any + Send>) -> PyErr {
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("unknown Rust panic");
    PyRuntimeError::new_err(format!("MiniBoosts failed: {message}"))
}

#[pyclass]
struct Model {
    model: Box<dyn Classifier + Send + Sync>,
    scores: Scores,
    count: usize,
    #[pyo3(get)]
    n_iterations: usize,
    #[pyo3(get)]
    stop_reason: String,
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
            validate_scores(values, n_rows)?;
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

fn run<B>(
    mut booster: B,
    learner: &CallbackLearner,
    limit: usize,
) -> (Box<dyn Classifier + Send + Sync>, String)
where
    B: Booster<Hypothesis>,
    B::Output: Classifier + Send + Sync + 'static,
{
    booster.preprocess();
    let mut stop = "max_iterations".to_string();
    for round in 1..=limit {
        learner.round.set(round);
        if booster.boost(learner, round).is_break() {
            stop = "algorithm_stop".into();
            break;
        }
    }
    (Box::new(booster.postprocess()), stop)
}

#[pyfunction]
#[pyo3(signature = (algorithm, x, y, callback, max_iterations, tolerance=0.01, nu=1.0, gamma=0.25, kappa=0.1, update_rule="default"))]
#[allow(clippy::too_many_arguments)]
fn train(
    algorithm: &str,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    callback: Py<PyAny>,
    max_iterations: usize,
    tolerance: f64,
    nu: f64,
    gamma: f64,
    kappa: f64,
    update_rule: &str,
) -> PyResult<Model> {
    let sample = Sample::from_dense(&x, Some(&y)).map_err(PyValueError::new_err)?;
    if max_iterations == 0 {
        return Err(PyValueError::new_err("max_iterations must be positive"));
    }
    if y.iter().any(|v| *v != -1.0 && *v != 1.0) || !y.contains(&-1.0) || !y.contains(&1.0) {
        return Err(PyValueError::new_err(
            "training requires both -1 and +1 labels",
        ));
    }
    let rule = match update_rule {
        "default" => None,
        "classic" => Some(FwUpdateRule::Classic),
        "short_step" => Some(FwUpdateRule::ShortStep),
        "line_search" => Some(FwUpdateRule::LineSearch),
        _ => return Err(PyValueError::new_err("unknown update_rule")),
    };
    let learner = CallbackLearner {
        callback,
        scores: Arc::new(RwLock::new(Vec::new())),
        error: RefCell::new(None),
        round: Cell::new(0),
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let s = &sample;
        let l = &learner;
        let n = max_iterations;
        let value = match algorithm {
            "adaboost" => run(AdaBoost::init(s).tolerance(tolerance), l, n),
            "adaboostv" => run(AdaBoostV::init(s).tolerance(tolerance), l, n),
            "madaboost" => run(MadaBoost::init(s).tolerance(tolerance), l, n),
            "lpboost" => run(LpBoost::init(s).nu(nu).tolerance(tolerance), l, n),
            "erlpboost" => run(ErlpBoost::init(s).nu(nu).tolerance(tolerance), l, n),
            "corrective_erlpboost" => {
                let b = CorrectiveErlpBoost::init(s).nu(nu).tolerance(tolerance);
                run(
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
                let b = MlpBoost::init(s).nu(nu).tolerance(tolerance);
                run(
                    if let Some(rule) = rule {
                        b.update_rule(rule)
                    } else {
                        b
                    },
                    l,
                    n,
                )
            }
            "smoothboost" => run(SmoothBoost::init(s).gamma(gamma).kappa(kappa), l, n),
            "softboost" => {
                let mut b = SoftBoost::init(s).nu(nu).tolerance(tolerance);
                b.preprocess();
                let mut stop = "max_iterations".to_string();
                for round in 1..=n {
                    l.round.set(round);
                    if b.boost(l, round).is_break() {
                        stop = "algorithm_stop".into();
                        break;
                    }
                }
                if let Some(reason) = b.numerical_stop_reason() {
                    return Err(PyRuntimeError::new_err(reason.to_owned()));
                }
                (
                    Box::new(b.postprocess()) as Box<dyn Classifier + Send + Sync>,
                    stop,
                )
            }
            "totalboost" => {
                let mut b = TotalBoost::init(s).tolerance(tolerance);
                b.preprocess();
                let mut stop = "max_iterations".to_string();
                for round in 1..=n {
                    l.round.set(round);
                    if b.boost(l, round).is_break() {
                        stop = "algorithm_stop".into();
                        break;
                    }
                }
                if let Some(reason) = b.numerical_stop_reason() {
                    return Err(PyRuntimeError::new_err(reason.to_owned()));
                }
                (
                    Box::new(b.postprocess()) as Box<dyn Classifier + Send + Sync>,
                    stop,
                )
            }
            "graph_separation" => run(GraphSeparationBoosting::init(s), l, n),
            _ => return Err(PyValueError::new_err("unknown boosting algorithm")),
        };
        Ok(value)
    }));
    if let Some(error) = learner.error.into_inner() {
        return Err(error);
    }
    let (model, stop_reason) = result.map_err(panic_error)??;
    let count = learner
        .scores
        .read()
        .expect("confidence cache poisoned")
        .len();
    Ok(Model {
        model,
        scores: learner.scores,
        count,
        n_iterations: count,
        stop_reason,
    })
}

#[pyclass]
struct Tree {
    model: miniboosts::DecisionTreeClassifier,
    n_features: usize,
}
#[pymethods]
impl Tree {
    fn decision_function(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let sample = Sample::from_dense(&x, None).map_err(PyValueError::new_err)?;
        if sample.shape().1 != self.n_features {
            return Err(PyValueError::new_err("incorrect feature count"));
        }
        catch_unwind(AssertUnwindSafe(|| self.model.confidence_all(&sample))).map_err(panic_error)
    }
}
#[pyfunction]
fn fit_tree(
    py: Python<'_>,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Vec<f64>,
    max_depth: usize,
) -> PyResult<Tree> {
    let sample = Sample::from_dense(&x, Some(&y)).map_err(PyValueError::new_err)?;
    if max_depth == 0
        || y.iter().any(|v| *v != -1.0 && *v != 1.0)
        || !y.contains(&-1.0)
        || !y.contains(&1.0)
    {
        return Err(PyValueError::new_err(
            "tree needs positive depth and both -1/+1 classes",
        ));
    }
    if weights.len() != y.len()
        || weights.iter().any(|v| !v.is_finite() || *v < 0.0)
        || (weights.iter().sum::<f64>() - 1.0).abs() > 1e-8
    {
        return Err(PyValueError::new_err(
            "tree weights must be a finite probability distribution",
        ));
    }
    let n_features = sample.shape().1;
    let model = py
        .detach(|| {
            catch_unwind(AssertUnwindSafe(|| {
                DecisionTreeBuilder::new(&sample)
                    .max_depth(max_depth)
                    .split_by(SplitBy::Entropy)
                    .build()
                    .produce(&sample, &weights)
            }))
        })
        .map_err(panic_error)?;
    Ok(Tree { model, n_features })
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_function(wrap_pyfunction!(fit_tree, m)?)?;
    m.add_class::<Model>()?;
    m.add_class::<Tree>()?;
    Ok(())
}
