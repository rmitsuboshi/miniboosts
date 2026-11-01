//! Thin WebAssembly boundary for the interactive demo: JavaScript supplies
//! labeled 2-D points, Rust trains a booster with decision trees and returns
//! the ensemble's confidence on a square grid over `[-1, 1]^2`.
use miniboosts::{Classifier, Sample, prelude::*};
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
    /// `"algorithm_stop"`, `"max_iterations"`, or, for SmoothBoost,
    /// `"round_budget"` (the paper's round bound was reached without the
    /// stopping test firing, so its margin guarantee does not apply).
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

/// Trains `algorithm` on the points `(xs[i], ys[i])` with labels
/// `labels[i] ∈ {-1, +1}` and evaluates the result on a
/// `resolution × resolution` grid.
///
/// `nu` is the capping parameter of the soft-margin boosters
/// (`1 ≤ nu ≤ n`), `gamma` and `kappa` are SmoothBoost's parameters, and
/// `n_hypotheses` (`T`, `0` for none) and `gap` (`v`, NaN for none) are
/// SparsiBoost's targets, at least one of which must be set.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn train(
    algorithm: &str,
    xs: &[f64],
    ys: &[f64],
    labels: &[f64],
    max_depth: usize,
    max_iterations: usize,
    tolerance: f64,
    nu: f64,
    gamma: f64,
    kappa: f64,
    n_hypotheses: usize,
    gap: f64,
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
    if max_depth == 0 || max_iterations == 0 || !(2..=400).contains(&resolution) {
        return Err(JsError::new(
            "depth, iterations, and resolution must be positive",
        ));
    }
    if !(tolerance.is_finite() && tolerance > 0.0) {
        return Err(JsError::new("tolerance must be positive"));
    }
    if !(1.0..=n as f64).contains(&nu) {
        return Err(JsError::new(
            "nu must lie between 1 and the number of points",
        ));
    }
    if !(0.0 < gamma && gamma < 0.5) || !(0.0 < kappa && kappa < 1.0) {
        return Err(JsError::new(
            "gamma must lie in (0, 0.5) and kappa in (0, 1)",
        ));
    }
    if !(gap.is_nan() || (gap.is_finite() && gap > 0.0)) {
        return Err(JsError::new("the target gap v must be positive"));
    }
    if algorithm == "sparsiboost" && n_hypotheses == 0 && gap.is_nan() {
        return Err(JsError::new(
            "SparsiBoost needs a target size T, a target gap v, or both",
        ));
    }

    let rows = xs
        .iter()
        .zip(ys)
        .map(|(&x, &y)| vec![x, y])
        .collect::<Vec<_>>();
    let sample = Sample::from_dense(&rows, Some(labels)).map_err(|e| JsError::new(&e))?;
    let tree = DecisionTreeBuilder::new(&sample)
        .max_depth(max_depth)
        .split_by(SplitBy::Entropy)
        .build();

    let s = &sample;
    let (model, rounds, stop_reason, note): (Box<dyn Classifier>, _, _, _) = match algorithm {
        "adaboost" => run(
            AdaBoost::init(s).tolerance(tolerance),
            &tree,
            max_iterations,
            no_check,
        )?
        .boxed(),
        "adaboostv" => run(
            AdaBoostV::init(s).tolerance(tolerance),
            &tree,
            max_iterations,
            no_check,
        )?
        .boxed(),
        "madaboost" => run(
            MadaBoost::init(s).tolerance(tolerance),
            &tree,
            max_iterations,
            no_check,
        )?
        .boxed(),
        "smoothboost" => run(
            SmoothBoost::init(s).gamma(gamma).kappa(kappa),
            &tree,
            max_iterations,
            |b: &SmoothBoost<_>| Ok(b.reached_round_budget().then_some("round_budget")),
        )?
        .boxed(),
        "lpboost" => run(
            LpBoost::init(s).nu(nu).tolerance(tolerance),
            &tree,
            max_iterations,
            no_check,
        )?
        .boxed(),
        "erlpboost" => run(
            ErlpBoost::init(s).nu(nu).tolerance(tolerance),
            &tree,
            max_iterations,
            no_check,
        )?
        .boxed(),
        "corrective_erlpboost" => run(
            CorrectiveErlpBoost::init(s).nu(nu).tolerance(tolerance),
            &tree,
            max_iterations,
            no_check,
        )?
        .boxed(),
        "mlpboost" => run(
            MlpBoost::init(s).nu(nu).tolerance(tolerance),
            &tree,
            max_iterations,
            no_check,
        )?
        .boxed(),
        "softboost" => run(
            SoftBoost::init(s).nu(nu).tolerance(tolerance),
            &tree,
            max_iterations,
            |b: &SoftBoost<_>| {
                b.numerical_stop_reason()
                    .map_or(Ok(None), |r| Err(r.to_owned()))
            },
        )?
        .boxed(),
        "totalboost" => run(
            TotalBoost::init(s).tolerance(tolerance),
            &tree,
            max_iterations,
            |b: &TotalBoost<_>| {
                b.numerical_stop_reason()
                    .map_or(Ok(None), |r| Err(r.to_owned()))
            },
        )?
        .boxed(),
        "graph_separation" => run(
            GraphSeparationBoosting::init(s),
            &tree,
            max_iterations,
            no_check,
        )?
        .boxed(),
        "sparsiboost" => {
            let mut booster = SparsiBoost::init(s);
            if n_hypotheses > 0 {
                booster = booster.n_hypotheses(n_hypotheses);
            }
            if !gap.is_nan() {
                booster = booster.tolerance(gap);
            }
            let planned = booster.max_loop();
            let t = run(booster, &tree, max_iterations, no_check)?;
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
        _ => return Err(JsError::new("unknown boosting algorithm")),
    };

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
    fn boxed(self) -> (Box<dyn Classifier>, usize, &'static str, String) {
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
