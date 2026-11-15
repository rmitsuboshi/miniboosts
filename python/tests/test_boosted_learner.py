"""BoostedLearnerClassifier: a booster as a weak learner, by resampling."""
import numpy as np
import pytest
from scipy import sparse
import miniboosts as mb
from miniboosts import _native


def _interval(n=60):
    # Positive inside [15, 35): no single stump is right on more than 2/3 of the rows.
    X = np.arange(n, dtype=float)[:, None]
    y = np.where((X[:, 0] >= 15) & (X[:, 0] < 35), 1.0, -1.0)
    return X, y


def _inner(**kwargs):
    stump = mb.DecisionTreeClassifier(max_depth=1)
    return mb.BoostedLearnerClassifier(
        estimator=mb.AdaBoostClassifier(estimator=stump, max_iterations=15), **kwargs)


def test_native_rows_and_their_validation():
    # The distribution of the draws is tested in Rust; here, the boundary.
    uniform = [0.02] * 50
    rows = _native.boosted_learner_rows(uniform, 1)
    assert len(rows) == 50 and rows == sorted(rows)
    assert len(_native.boosted_learner_rows(uniform, 1, 80)) == 80
    assert set(_native.boosted_learner_rows([0.5, 0.0, 0.5], 0, 30)) <= {0, 2}
    for bad, size in [([], None), ([0.0, 0.0], None), ([0.5, -0.5], None),
                      ([0.5, float('nan')], None), ([0.5, 0.5], 0)]:
        with pytest.raises(ValueError):
            _native.boosted_learner_rows(bad, 0, size)


def test_fit_uses_the_drawn_rows():
    X, y = _interval()
    weights = np.linspace(1.0, 3.0, len(y))
    learner = _inner(seed=4, sample_size=80).fit(X, y, sample_weight=weights)
    expected = _native.boosted_learner_rows((weights / weights.max() / (weights / weights.max()).sum()).tolist(), 4, 80)
    np.testing.assert_array_equal(learner.rows_, expected)
    np.testing.assert_array_equal(learner.classes_, [-1, 1])
    scores = learner.decision_function(X)
    assert np.all(np.abs(scores) <= 1 + 1e-12)
    # The same weights give the same hypothesis.
    again = _inner(seed=4, sample_size=80).fit(X, y, sample_weight=weights)
    np.testing.assert_array_equal(again.decision_function(X), scores)


def test_sign_returns_predictions():
    X, y = _interval()
    real = _inner(seed=1).fit(X, y)
    signed = _inner(seed=1, sign=True).fit(X, y)
    scores = signed.decision_function(X)
    assert set(scores) <= {-1.0, 1.0}
    np.testing.assert_array_equal(scores, np.where(real.decision_function(X) >= 0, 1.0, -1.0))


def test_single_label_draw_gives_a_constant():
    X, y = _interval()
    weights = np.zeros(len(y))
    weights[20] = 1.0
    learner = _inner().fit(X, y, sample_weight=weights)
    assert learner.estimator_ is None and learner.constant_ == 1.0
    np.testing.assert_array_equal(learner.decision_function(X), np.ones(len(y)))
    np.testing.assert_array_equal(learner.predict(X), np.ones(len(y)))


@pytest.mark.parametrize('outer', [
    lambda inner: mb.AdaBoostClassifier(estimator=inner, max_iterations=8),
    lambda inner: mb.LPBoostClassifier(estimator=inner, max_iterations=20),
])
def test_boosters_run_over_a_boosted_learner(outer):
    X, y = _interval()
    model = outer(_inner(seed=2)).fit(X, y)
    assert (model.predict(X) == y).mean() >= 0.9
    # Sparse input reaches the inner booster as well.
    sparse_model = outer(_inner(seed=2)).fit(sparse.csr_matrix(X), y)
    np.testing.assert_allclose(sparse_model.decision_function(sparse.csr_matrix(X)),
                               model.decision_function(X), rtol=0, atol=1e-9)


def test_invalid_parameters():
    X, y = _interval()
    for kwargs in [dict(seed=-1), dict(sample_size=0), dict(sample_size=1.5), dict(sign='yes')]:
        learner = mb.BoostedLearnerClassifier(**kwargs)
        with pytest.raises(ValueError):
            learner.fit(X, y)
        assert not hasattr(learner, 'rows_')
    with pytest.raises(TypeError):
        mb.BoostedLearnerClassifier(estimator=object()).fit(X, y)
