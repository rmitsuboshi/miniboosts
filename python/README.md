# MiniBoosts for Python

Python bindings for the Rust MiniBoosts library, with algorithm-specific,
scikit-learn-style binary classifiers. Boosting and aggregation use the Rust
implementations; weak learners can be Rust decision trees or Python estimators.

## Install from this repository

Requires Rust 1.85+, Python 3.10+, NumPy, and scikit-learn 1.6+ (below 2).
From the repository root, with a virtual environment activated:

```sh
python -m pip install ./python
```

This builds a native extension locally. The Python package has not been
published to PyPI. The existing Rust crate remains usable without Python or
PyO3; `python/` is a separate, non-published Cargo package.

For development:

```sh
python -m pip install 'maturin>=1.9,<2' 'pytest>=8' 'numpy>=1.24' 'scikit-learn>=1.6,<2'
maturin develop --manifest-path python/Cargo.toml --release
python -m pytest python/tests
```

## Train and predict

```python
import numpy as np
from miniboosts import AdaBoostClassifier, DecisionTreeClassifier

X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
y = np.array(["cat", "cat", "dog", "dog"])
model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    tolerance=0.01,
    max_iterations=100,
)
model.fit(X, y)  # warns: 'cat' -> -1, 'dog' -> +1
labels = model.predict(X)  # original labels
scores = model.decision_function(X)
```

`fit(X, y)` returns the same estimator and always starts a fresh fit. Predictions
before a successful fit raise `sklearn.exceptions.NotFittedError`. A failed
refit discards the previous model. Constructors only store configuration.

Inputs must be dense, finite real numeric matrices and one-dimensional binary
labels (finite numbers, booleans, or strings; do not mix text and numbers). `classes_` gives the sorted classes:
`classes_[0]` maps to -1 and `classes_[1]` to +1. Conversion emits one
`UserWarning` per fit showing the mapping; labels already equal to -1/+1 need
no warning. Missing labels, one class, and more than two classes are errors.
A zero ensemble score predicts `classes_[1]`.

`decision_function` preserves the Rust algorithm's aggregation and scale. It
is not a probability, and the ensemble has no imposed [-1, +1] bound.
`get_params`, `set_params`, `sklearn.base.clone`, `Pipeline`, and sequential
`GridSearchCV` are supported. Full sklearn estimator-check compliance is not
claimed.

## Choose a weak learner

The default `estimator=None` uses `DecisionTreeClassifier(max_depth=1)`
(`RegressionTreeRegressor()` for the regression boosters below).
A custom learner should implement sklearn's estimator cloning convention
(e.g. inherit `BaseEstimator` and expose constructor parameters), and:

- `fit(X, y, sample_weight=weights)` trains on encoded -1/+1 labels and the
  boosting distribution. It must actually use those sample weights.
- `decision_function(X)` returns a numeric vector of shape `(len(X),)`.
  Every value must be finite and within **[-1, +1]**. Positive confidence means
  the +1 class. If the learner exposes `classes_`, it must be `[-1, +1]`.

`GradientBoostingClassifier` and `LogitBoostClassifier` instead need a
**regression** weak learner, such as sklearn's `DecisionTreeRegressor` or the
native `RegressionTreeRegressor`:

- `fit(X, z, sample_weight=weights)` fits the real-valued working response `z`
  (pseudo-residuals for gradient boosting, the clipped Newton response for
  LogitBoost) under the given weights (uniform for gradient boosting).
- `predict(X)` returns a finite numeric vector of shape `(len(X),)`; the values
  are unbounded.

Each boosting round fits an independent clone. The original estimator is not
modified. Predictions must be deterministic for a fitted learner and fixed
inputs: Rust caches training scores. Set random seeds on stochastic learners
for reproducible training.

Invalid confidence raises an error during training or prediction, with the
learner class, round, row (zero-based), and value for range/nonfinite failures.
There is no clipping or automatic score/probability transformation. Exceptions
raised by custom learners propagate to the caller. Rust algorithm failures and
numerical solver stops of SoftBoost, TotalBoost, BrownBoost, RobustBoost,
gradient boosting, and LogitBoost raise `RuntimeError`.

Many sklearn estimators have unbounded `decision_function` outputs or only
provide `predict`. To explicitly use **hard predictions** as -1/+1 weak scores:

```python
from sklearn.tree import DecisionTreeClassifier as SklearnTree
from miniboosts import AdaBoostClassifier, PredictionAdapter

model = AdaBoostClassifier(
    estimator=PredictionAdapter(SklearnTree(max_depth=2, random_state=0)),
    max_iterations=100,
)
model.fit(X, y)
```

`PredictionAdapter` discards confidence magnitude. Its wrapped estimator must
accept `sample_weight`. It does not silently rescue invalid confidence from a
learner passed directly to a booster.

### Native weak learners

These wrap the Rust weak learners. Each can also be used on its own: its `fit`
accepts `sample_weight` (normalized to a distribution), and labels are encoded
as for the boosters. Rust failures (e.g. non-convergence) raise `RuntimeError`.

| Python class | Parameters (defaults) | Hypothesis |
| --- | --- | --- |
| `DecisionTreeClassifier` | `max_depth=1` | Entropy-split tree with bounded leaf confidence |
| `BoundedLinearClassifier` | none | Edge-maximizing `(w·x + b) / S`, `‖(w, b)‖₂ ≤ 1`, `S = √(R² + 1)` |
| `LogisticRegressionClassifier` | `regularization=1e-4`, `output="sign"` | Weighted L2-regularized logistic regression |
| `NaiveBayesClassifier` | `var_smoothing=1e-9` | Weighted Gaussian naive Bayes, `P(+1∣x) − P(−1∣x)` |
| `RBFClassifier` | `bandwidth` (required), `centers=None` | Edge-maximizing Gaussian bump `±exp(−‖x − c‖² / (2σ²))` |
| `FiniteClassClassifier` | `hypotheses` (required), `with_negations=False` | Edge-maximizing member of a fixed list |
| `RegressionTreeRegressor` | `max_depth=3`, `n_bins=None` | Weighted least-squares regression tree (for regression boosters) |

`RegressionTreeRegressor` considers every split between consecutive distinct
values of a feature (threshold at the midpoint) when the feature has at most
`n_bins` distinct values (default cap 255); above the cap it uses `n_bins`
equal-width bins, and the best split may be missed.

- `BoundedLinearClassifier` exactly maximizes the edge over its class; `R` is
  the largest training-row norm, and confidences outside that radius are
  clamped to [-1, +1]. Boosting it yields a single linear function. The class
  depends on feature scale.
- `LogisticRegressionClassifier` minimizes a surrogate loss and need not
  maximize the edge (it can return a zero-edge hypothesis on symmetric data).
  `output="sign"` gives a halfspace; `"tanh"` gives `tanh((w·x + b)/2)`.
  The bias is regularized too.
- `NaiveBayesClassifier` uses weighted means, variances, and priors; it has no
  edge guarantee. With uniform weights it matches sklearn's `GaussianNB`.
- `RBFClassifier` chooses among bumps centered at the training rows indexed by
  `centers` (all rows by default). After fitting, `center_index_`, `center_`,
  and `sign_` describe the chosen bump. The class has no bias term, and each
  fit evaluates every bump on every row (`O(n k d)` for `k` centers).
- `FiniteClassClassifier` takes already fitted objects with
  `decision_function` values in [-1, +1] and selects the one with the largest
  weighted edge (`-h` too with `with_negations=True`; ties go to the smaller
  index, then to `+h`). `index_`, `sign_`, and `hypothesis_` describe the
  choice. `clone` keeps the same hypothesis objects instead of refitting
  copies; do not mutate them.

## Algorithms and parameters

All boosters accept `estimator=None` and `max_iterations=100` (SparsiBoost and
boost-by-majority: `None`, see below). The latter is a hard cap on calls to the Rust boosting
step; the algorithm may stop earlier.

| Python class | Additional parameters (defaults) |
| --- | --- |
| `AdaBoostClassifier` | `tolerance=0.01` |
| `AdaBoostVClassifier` | `tolerance=0.01` |
| `ArcGvClassifier` | none |
| `BoostByMajorityClassifier` | `gamma` (required), `tolerance=0.01`, `max_iterations=None` |
| `BrownBoostClassifier` | `tolerance=0.1`, `edge_threshold=1e-3` |
| `GradientBoostingClassifier` | `loss="logistic"`, `shrinkage=1.0` (regression weak learner) |
| `LogitBoostClassifier` | `max_response=4.0` (regression weak learner) |
| `MadaBoostClassifier` | `tolerance=0.01` |
| `LPBoostClassifier` | `nu=1.0`, `tolerance=0.01` |
| `ERLPBoostClassifier` | `nu=1.0`, `tolerance=0.01` |
| `CorrectiveERLPBoostClassifier` | `nu=1.0`, `tolerance=0.01`, `update_rule="short_step"` |
| `MLPBoostClassifier` | `nu=1.0`, `tolerance=0.01`, `update_rule="line_search"` |
| `RobustBoostClassifier` | `tolerance=0.1`, `theta=0.0`, `sigma_f=0.1` |
| `SmoothBoostClassifier` | `gamma=0.25`, `kappa=0.1` |
| `SoftBoostClassifier` | `nu=1.0`, `tolerance=0.01` |
| `SparsiBoostClassifier` | `n_hypotheses=None`, `tolerance=None`, `seed=0`, `max_iterations=None` |
| `TotalBoostClassifier` | `tolerance=0.01` |
| `GraphSeparationBoostingClassifier` | none |

`tolerance` has the original algorithm-specific meaning: an iteration-budget
parameter for AdaBoost/MadaBoost, a margin-update and budget parameter for
AdaBoostV, or an optimization stopping tolerance for the LP-based methods.
Equal values do not imply equal accuracy. `nu` lies in [1, number of examples]
and caps example weights at `1/nu`. SmoothBoost requires `0 < gamma < 0.5` and
`0 < kappa < 1`; its weak-learning assumption must hold on each distribution.
`update_rule` accepts `classic`, `short_step`, or `line_search`.

BrownBoost's `tolerance` is the target training error `ε` in (0, 1), which
sets the total time `c = (erfc⁻¹(ε))²`; `edge_threshold` is `ν` in [0, 1).
BrownBoost has no iteration bound, so `max_iterations` matters. After fitting,
`total_time_` is `c` and `remaining_time_` is the time left: `≤ 0` means the
loop exited, which is when Theorem 2 of Freund (2001) bounds the training error
by `ε`. A positive value means training stopped early and the bound does not
apply.

RobustBoost (Freund, 2009) is a BrownBoost variant for label noise.
`tolerance` is the error goal `ε` in (0, 1), `theta ≥ 0` the goal margin (on the
scale of the unnormalized scores; choose it by cross-validation), and
`sigma_f > 0` the final width. `remaining_time_` is `1 - t`: `0` means the
clock reached 1, so at most a fraction `2ε` of the training scores `y H(x)` are
at most `theta` (and so at most `2ε` of the training examples are
misclassified). A positive value means training stopped early.

Arc-gv (Breiman, 1999) converges to the best minimal margin only
asymptotically, so `max_iterations` is the round limit. `"algorithm_stop"`
means a step of size 0: the weak hypothesis does not beat the current minimal
margin, which certifies the optimum only if the weak learner minimizes the
weighted loss over its class.

Boost-by-majority (Freund, 1995) is not adaptive: `gamma` in (0, 1/2) is the
assumed advantage (weighted error at most `1/2 - gamma`) and `tolerance` the
target training error. It combines `k = ⌈ln(1/tolerance) / (2 gamma²)⌉`
hypotheses (`n_rounds_`) by an unweighted vote of their predicted signs, so
`decision_function` is the mean vote. With `max_iterations=None` all `k` rounds
can run; `decided_early_` is `True` if training stopped because every
example's final vote was already decided. If the advantage assumption fails,
the bound `exp(-2 gamma² k) ≤ tolerance` on the training error does not apply.

Gradient boosting (Friedman, 2001) fits each regression weak learner to the
pseudo-residuals with uniform weights, line-searches the step, and multiplies
it by `shrinkage` in (0, 1]. `loss` is `"logistic"` (binomial deviance),
`"exponential"`, or `"squared"`. LogitBoost (Friedman, Hastie, and Tibshirani,
2000) takes Newton steps on the binomial log-likelihood, fitting the working
response clipped to `[-max_response, max_response]` (`max_response ≥ 1`) by
weighted least squares. Both run exactly `max_iterations` rounds unless they
stop early, and give no training-error bound. After fitting, `intercept_` and
`coefficients_` give the additive score `F = intercept_ + Σ coefficients_[t]
h_t(x)` over the kept hypotheses; `decision_function` is `tanh(F)` (`F`
clamped to [-1, +1] for the squared loss), the class-probability difference
when `F` is half the log-odds.

SparsiBoost needs `n_hypotheses` (at most `T` output hypotheses), `tolerance`
(target gap `v`), or both; the size cap takes precedence. It runs AdaBoostV
for a round count derived from the target, then sparsifies the ensemble with a
seeded randomized discrepancy step (equal seeds give equal results). With
`max_iterations=None` all of these rounds run; a smaller cap truncates
AdaBoostV and voids the guarantee. After fitting, `adaboostv_rounds_` and
`adaboostv_tolerance_` give the AdaBoostV stage, `n_selected_` the number of
hypotheses with nonzero weight (possibly fewer than `T`),
`sparsification_error_` the largest change of a training margin, and
`sparsification_within_tolerance_` whether it is at most `v/2` (`None` without
a `tolerance`). `estimators_` holds all AdaBoostV-round learners, including
those the sparsification dropped.

See the repository README and Rust documentation for references and guarantees.

After fitting, `n_iter_` records the number of successfully produced weak
learners, excluding terminal calls that do not train a learner. `stop_reason_` is `"max_iterations"`,
`"algorithm_stop"`, or one of the algorithm-specific values below. `"algorithm_stop"` is an algorithm stop, not a generic
convergence certificate. `"round_budget"` means SmoothBoost reached its
theoretical round bound without its stopping test firing, so its margin
guarantee does not apply. `"edge_threshold"` means a BrownBoost weak hypothesis
had weighted correlation at most `edge_threshold`; it was discarded, training
stopped before the time ran out, and Theorem 2 does not apply.
`"nonpositive_edge"` is the same for a RobustBoost weak hypothesis with
nonpositive weighted correlation. `"zero_hypothesis"` means a gradient
boosting weak learner predicted 0 on every training row, so the step is
undefined. `"no_minimizer"` means that, for the logistic or exponential loss,
the weak hypothesis ordered every training example correctly
(`y h(x) ≥ 0`), so the loss decreases for every step size and the line search
has no minimizer; on small or separable data a deep tree can cause this in
round 1, leaving only the intercept (use a shallower tree or the squared loss). `"weights_underflow"` means every LogitBoost weight `p(1 - p)`
underflowed to 0 (the training scores have saturated); it is detected before
the weak learner is called. In the other cases the candidate of the last
round is discarded.

`estimators_` and `estimator_rounds_` expose all successfully
produced candidates, including candidates discarded by the Rust algorithm.
Do not mutate these fitted learners or lists.

## Current limits

This first binding supports dense binary classification. It does not yet
provide sparse inputs, multiclass, model persistence/pickling, `predict_proba`,
or an interactive UI. Do not share a fitted model across concurrent prediction
calls. Process-based workflows that serialize fitted native models are not
supported. Native weak-learner fitting releases the Python GIL; the entire boosting
loop does not. Array conversion and callback copies incur overhead, so include
them when reporting Python-facing timings. Training caches use O(nT) memory
for n examples and T candidate learners, in addition to algorithm-specific
storage. No wheels for other platforms are bundled with this source tree.
