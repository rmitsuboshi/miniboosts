"""Exact stumps, adversarial finite classes, and sub-sample majorities."""
import numpy as np
import pytest
from sklearn.base import clone

import miniboosts as mb
from miniboosts import _native


def _edge(scores, y, w):
    return float(np.dot(w * y, scores))


def _best_stump_edge(X, y, w):
    # Constants +-1 and stumps +-sign(x_f - t) at every midpoint.
    best = abs(_edge(np.ones(len(y)), y, w))
    for f in range(X.shape[1]):
        values = np.unique(X[:, f])
        for a, b in zip(values[:-1], values[1:]):
            t = a / 2 + b / 2
            best = max(best, abs(_edge(np.where(X[:, f] < t, -1.0, 1.0), y, w)))
    return best


def test_exact_sign_stump_maximizes_the_edge():
    rng = np.random.default_rng(3)
    for _ in range(10):
        X = np.c_[rng.integers(0, 5, 30), rng.normal(size=30)].astype(float)
        y = np.where(rng.random(30) < 0.5, -1.0, 1.0)
        w = rng.random(30)
        w /= w.sum()
        stump = mb.DecisionTreeClassifier(max_depth=1, split_by='edge', exact_splits=True,
                                          leaf_confidence='sign').fit(X, y, sample_weight=w)
        scores = stump.decision_function(X)
        assert set(np.abs(scores)) == {1.0}
        assert abs(_edge(scores, y, w) - _best_stump_edge(X, y, w)) < 1e-12


@pytest.mark.parametrize('params', [{'split_by': 'random'}, {'exact_splits': 1},
                                    {'leaf_confidence': 'probability'}])
def test_tree_rejects_invalid_parameters(params):
    X, y = np.array([[0.0], [1.0]]), np.array([-1, 1])
    with pytest.raises(ValueError):
        mb.DecisionTreeClassifier(**params).fit(X, y)


class _Fixed:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def decision_function(self, X):
        return self.values[:len(X)]


def test_finite_class_selection_rules():
    X = np.zeros((4, 1))
    y = np.array([1, 1, -1, -1])
    # Edges under the uniform weights: 1, 0.5, 0.
    hypotheses = [_Fixed([1, 1, -1, -1]), _Fixed([1, 1, -1, 1]), _Fixed([1, 1, 1, 1])]
    largest = mb.FiniteClassClassifier(hypotheses).fit(X, y)
    assert (largest.index_, largest.sign_, largest.assumption_violated_) == (0, 1.0, False)
    weakest = mb.FiniteClassClassifier(hypotheses, selection='weakest', gamma=0.25).fit(X, y)
    assert (weakest.index_, weakest.sign_) == (1, 1.0)
    exactly = mb.FiniteClassClassifier(hypotheses, selection='exactly', gamma=0.25).fit(X, y)
    assert (exactly.index_, exactly.sign_) == (1, 0.5)
    assert _edge(exactly.decision_function(X), np.where(y > 0, 1.0, -1.0), np.full(4, 0.25)) == 0.25
    missing = mb.FiniteClassClassifier(hypotheses[1:], selection='weakest', gamma=0.75).fit(X, y)
    assert (missing.index_, missing.assumption_violated_) == (0, True)
    assert clone(exactly).get_params()['selection'] == 'exactly'
    for params in [{'selection': 'smallest'}, {'selection': 'weakest'},
                   {'selection': 'exactly', 'gamma': 1.5}]:
        with pytest.raises(ValueError):
            mb.FiniteClassClassifier(hypotheses, **params).fit(X, y)


def test_rbf_selection_reports_violations():
    X = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    y = np.array([-1, -1, 1, 1])
    h = mb.RBFClassifier(bandwidth=0.5, selection='weakest', gamma=0.99).fit(X, y)
    assert h.assumption_violated_
    h = mb.RBFClassifier(bandwidth=0.5, selection='weakest', gamma=0.01).fit(X, y)
    assert not h.assumption_violated_


def _separable(n):
    X = np.r_[np.linspace(-3, -1, n // 2), np.linspace(1, 3, n - n // 2)][:, None]
    y = np.r_[np.full(n // 2, 'a'), np.full(n - n // 2, 'b')]
    return X, y


def test_majority_of_five():
    parts = _native.majority_of_five_parts(23, 5, 3)
    assert sorted(np.concatenate(parts).tolist()) == list(range(23))
    assert {len(p) for p in parts} == {4, 5}
    X, y = _separable(40)
    booster = mb.AdaBoostClassifier(max_iterations=5)
    with pytest.warns(UserWarning, match='encoded'):
        model = mb.MajorityOfFiveClassifier(estimator=booster, seed=1).fit(X, y)
    assert len(model.estimators_) == 5
    assert (model.predict(X) == y).all()
    assert set(model.decision_function(X)) <= {-1.0, 1.0}
    with pytest.warns(UserWarning, match='encoded'), pytest.raises(ValueError):
        mb.MajorityOfFiveClassifier(n_parts=50).fit(X, y)
    with pytest.raises(ValueError):
        mb.MajorityOfFiveClassifier(n_parts=0).fit(X, y)


def test_larsen_ritzert():
    assert len(_native.larsen_ritzert_subsamples(16, 0)) == 9
    assert len(_native.larsen_ritzert_subsamples(64, 0, 5)) == 5
    X, y = _separable(16)
    with pytest.warns(UserWarning, match='encoded'):
        model = mb.LarsenRitzertClassifier(estimator=mb.AdaBoostClassifier(max_iterations=5)).fit(X, y)
    assert len(model.estimators_) == 9
    assert (model.predict(X) == y).all()
    with pytest.raises(ValueError):
        mb.LarsenRitzertClassifier(max_subsamples=0).fit(X, y)


def test_failed_voter_fit_names_the_subsample():
    # One part gets only one class: the booster's fit fails, and the error
    # says which sub-sample it was.
    X = np.arange(10, dtype=float)[:, None]
    y = np.r_[np.full(9, -1), 1]
    with pytest.raises(ValueError) as error:
        mb.MajorityOfFiveClassifier().fit(X, y)
    assert any('sub-sample' in note for note in getattr(error.value, '__notes__', []))
