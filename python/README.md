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

The default `estimator=None` uses `DecisionTreeClassifier(max_depth=1)`.
A custom learner should implement sklearn's estimator cloning convention
(e.g. inherit `BaseEstimator` and expose constructor parameters), and:

- `fit(X, y, sample_weight=weights)` trains on encoded -1/+1 labels and the
  boosting distribution. It must actually use those sample weights.
- `decision_function(X)` returns a numeric vector of shape `(len(X),)`.
  Every value must be finite and within **[-1, +1]**. Positive confidence means
  the +1 class. If the learner exposes `classes_`, it must be `[-1, +1]`.

Each boosting round fits an independent clone. The original estimator is not
modified. Predictions must be deterministic for a fitted learner and fixed
inputs: Rust caches training scores. Set random seeds on stochastic learners
for reproducible training.

Invalid confidence raises an error during training or prediction, with the
learner class, round, row (zero-based), and value for range/nonfinite failures.
There is no clipping or automatic score/probability transformation. Exceptions
raised by custom learners propagate to the caller. Rust algorithm failures and
SoftBoost/TotalBoost numerical solver stops raise `RuntimeError`.

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
learner passed directly to a booster. The native `DecisionTreeClassifier` can
also be used on its own and accepts `sample_weight` in its own `fit` method.

## Algorithms and parameters

All boosters accept `estimator=None` and `max_iterations=100`. The latter is a
hard cap on calls to the Rust boosting step; the algorithm may stop earlier.

| Python class | Additional parameters (defaults) |
| --- | --- |
| `AdaBoostClassifier` | `tolerance=0.01` |
| `AdaBoostVClassifier` | `tolerance=0.01` |
| `MadaBoostClassifier` | `tolerance=0.01` |
| `LPBoostClassifier` | `nu=1.0`, `tolerance=0.01` |
| `ERLPBoostClassifier` | `nu=1.0`, `tolerance=0.01` |
| `CorrectiveERLPBoostClassifier` | `nu=1.0`, `tolerance=0.01`, `update_rule="short_step"` |
| `MLPBoostClassifier` | `nu=1.0`, `tolerance=0.01`, `update_rule="line_search"` |
| `SmoothBoostClassifier` | `gamma=0.25`, `kappa=0.1` |
| `SoftBoostClassifier` | `nu=1.0`, `tolerance=0.01` |
| `TotalBoostClassifier` | `tolerance=0.01` |
| `GraphSeparationBoostingClassifier` | none |

`tolerance` has the original algorithm-specific meaning: an iteration-budget
parameter for AdaBoost/MadaBoost, a margin-update and budget parameter for
AdaBoostV, or an optimization stopping tolerance for the LP-based methods.
Equal values do not imply equal accuracy. `nu` lies in [1, number of examples]
and caps example weights at `1/nu`. SmoothBoost requires `0 < gamma < 0.5` and
`0 < kappa < 1`; its weak-learning assumption must hold on each distribution.
`update_rule` accepts `classic`, `short_step`, or `line_search`.
See the repository README and Rust documentation for references and guarantees.

After fitting, `n_iter_` records the number of successfully produced weak
learners, excluding terminal calls that do not train a learner. `stop_reason_` is `"max_iterations"` or
`"algorithm_stop"`; the latter is an algorithm stop, not a generic convergence
certificate. `estimators_` and `estimator_rounds_` expose all successfully
produced candidates, including candidates discarded by the Rust algorithm.
Do not mutate these fitted learners or lists.

## Current limits

This first binding supports dense binary classification. It does not yet
provide sparse inputs, multiclass, model persistence/pickling, `predict_proba`,
or an interactive UI. Do not share a fitted model across concurrent prediction
calls. Process-based workflows that serialize fitted native models are not
supported. Native tree fitting releases the Python GIL; the entire boosting
loop does not. Array conversion and callback copies incur overhead, so include
them when reporting Python-facing timings. Training caches use O(nT) memory
for n examples and T candidate learners, in addition to algorithm-specific
storage. No wheels for other platforms are bundled with this source tree.
